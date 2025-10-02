import json
import os
import random
import numpy as np
import pyroomacoustics as pa
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
import sys

# my_moduleが提供されていることを前提とします
from mymodule import const, rec_config as rec_conf, rec_utility as rec_util
from mymodule import my_func, reverbe_feater as rev_feat


def create_reverb_dataset_final(
		target_dir: Path,
		noise_path: Path,
		output_dir: Path,
		num_rooms: int,
		num_files_per_room: int,
		snr: float,
		channel: int = 1
):
	"""
	自己教師あり学習用のデータセットを生成する関数。
	- 複数の「部屋（ドメイン）」をシミュレートする。
	- 各シミュレーションのメタデータ（ドメインID、物理的特徴量）を保存する。
	- 目的信号と雑音信号の両方に個別に残響を付加し、結合する。
	"""
	output_dir.mkdir(parents=True, exist_ok=True)
	metadata_path = output_dir / "metadata.json"
	metadata = {}

	# 目的信号のファイルリストを取得
	speech_files = list(target_dir.rglob("*.wav"))
	if not speech_files:
		print(f"❌ エラー: 目的信号ファイルが見つかりません: {target_dir}", file=sys.stderr)
		return

	print(f"✅ 目的信号ファイルリストの取得完了。{len(speech_files)} 個のファイルを処理します。")

	# 雑音信号の読み込み (一度だけ)
	try:
		noise_signal_orig, fs_noise = sf.read(noise_path)
	except FileNotFoundError:
		print(f"❌ エラー: 雑音ファイルが見つかりません: {noise_path}", file=sys.stderr)
		return

	if noise_signal_orig.ndim > 1:
		noise_signal_orig = noise_signal_orig.mean(axis=1)

	# 部屋ごとにループ
	for room_id in range(num_rooms):
		print(f"\n--- Simulating Room (Domain) ID: {room_id} ---")

		# ランダムな部屋のパラメータを生成 (サイズと吸音率は適宜調整してください)
		room_dim = np.array([random.uniform(3, 8), random.uniform(3, 8), random.uniform(2.5, 4)])
		# Sabineの残響式から吸収率と反射上限回数を決定
		rt60_target = random.uniform(0.1, 1.0)
		e_absorption, max_order = pa.inverse_sabine(rt60_target, room_dim)

		# 部屋のメタデータに情報を記録
		room_metadata = {
			"room_id": room_id,
			"room_dim": room_dim.tolist(),
			"target_rt60": rt60_target,
			"absorption": e_absorption,
			"max_order": max_order,
			"files": []
		}

		# 部屋の作成とマイクの設置
		room = pa.ShoeBox(room_dim, fs=rec_conf.sampling_rate, max_order=max_order, materials=pa.Material(e_absorption))
		mic_center = room_dim / 2
		mic_coordinate = rec_util.set_mic_coordinate(center=mic_center, num_channels=channel, distance=0.1)
		room.add_microphone_array(pa.MicrophoneArray(mic_coordinate, fs=room.fs))

		# 音源の位置をランダムに設定（壁から離す）
		source_pos_signal = np.array([
			random.uniform(0.5, room_dim[0] - 0.5),
			random.uniform(0.5, room_dim[1] - 0.5),
			random.uniform(0.5, room_dim[2] - 0.5)
		])
		source_pos_noise = np.array([
			random.uniform(0.5, room_dim[0] - 0.5),
			random.uniform(0.5, room_dim[1] - 0.5),
			random.uniform(0.5, room_dim[2] - 0.5)
		])

		# 音源の追加
		room.add_source(source_pos_signal)
		room.add_source(source_pos_noise)

		# RIRを計算
		room.compute_rir()
		rir_signal = room.rir[0][0]  # 目的信号のRIR
		rir_noise = room.rir[0][1]  # 雑音信号のRIR

		# 物理的特徴量（RT60, C50, D50）を計算
		# rirが2次元配列（マイク, ソース）で返される可能性があるため、最初のRIRを使用
		rt60 = room.measure_rt60()[0][0]
		c50 = rev_feat.calculate_c50(rir_signal)
		d50 = rev_feat.calculate_d50(rir_signal)

		# 各部屋で指定された数のファイルを生成
		selected_speech_files = random.sample(speech_files, k=num_files_per_room)
		for clean_filepath in tqdm(selected_speech_files, desc=f"Generating files for room {room_id}"):
			try:
				# クリーン音声信号の読み込みと前処理
				clean_signal, fs_clean = sf.read(clean_filepath)
				if clean_signal.ndim > 1:
					clean_signal = clean_signal.mean(axis=1)

				# 雑音信号を切り出し
				start_noise = random.randint(0, len(noise_signal_orig) - len(clean_signal))
				noise_segment_orig = noise_signal_orig[start_noise: start_noise + len(clean_signal)]

				# RIRで畳み込み、残響付き信号を生成
				reverb_signal = np.convolve(clean_signal, rir_signal, mode='full')[:len(clean_signal)]
				reverb_noise = np.convolve(noise_segment_orig, rir_noise, mode='full')[:len(noise_segment_orig)]

				# SNRを調整して結合
				scaled_noise = rec_util.get_scale_noise(reverb_signal, reverb_noise, snr)
				mixed_signal = reverb_signal + scaled_noise

				# ファイル名を生成
				base_filename = clean_filepath.stem
				output_filename = f"{base_filename}_room{room_id:03}_rt{int(rt60 * 10):03}_snr{snr:02}.wav"

				# 出力ディレクトリに保存
				output_sub_dir = output_dir / f"room_{room_id}"
				output_sub_dir.mkdir(parents=True, exist_ok=True)

				output_path = output_sub_dir / output_filename
				sf.write(output_path, mixed_signal, rec_conf.sampling_rate)

				# メタデータにファイル情報を追加
				file_metadata = {
					"filename": output_filename,
					"clean_source_file": clean_filepath.name,
					"rt60": rt60,
					"c50": c50,
					"d50": d50,
					"snr": snr
				}
				room_metadata["files"].append(file_metadata)

			except Exception as e:
				tqdm.write(f"❌ ファイル処理中にエラーが発生しました: {clean_filepath.name} ({e})", file=sys.stderr)

		metadata[f"room_{room_id}"] = room_metadata

	# 全メタデータファイルを保存
	with open(metadata_path, "w") as f:
		json.dump(metadata, f, indent=4)

	print("\n🎉 データセットの生成が完了しました。")


if __name__ == "__main__":
	# 使用例
	speech_type = "subset_DEMAND"
	noise_type = "hoth"

	# `mymodule/const.py`に定義されたパスを基に設定
	try:
		sample_data_dir = Path(const.SAMPLE_DATA_DIR)
		mix_data_dir = Path(const.MIX_DATA_DIR)
	except NameError:
		print("const.pyのパス設定が読み込めません。手動でパスを設定します。")
		sample_data_dir = Path("./sound_data/sample_data")
		mix_data_dir = Path("./sound_data/mix_data")

	# `train/`ディレクトリ内の音声ファイルを使用
	data_type = "test"
	target_dir = sample_data_dir / "speech" / speech_type / data_type
	# `noise/`ディレクトリ内の雑音ファイルを使用
	noise_path = sample_data_dir / "noise" / f"{noise_type}.wav"
	output_dir = mix_data_dir / "reverb_encoder_dataset" / data_type

	create_reverb_dataset_final(
		target_dir=target_dir,
		noise_path=noise_path,
		output_dir=output_dir,
		num_rooms=10,
		num_files_per_room=20,
		snr=10,
		channel=1
	)