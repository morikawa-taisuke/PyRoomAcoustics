import json
import os
import random
import numpy as np
import pyroomacoustics as pa
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
import sys
import argparse  # --- ▼ 修正箇所 ▼ --- (argparseをインポート)

# my_moduleが提供されていることを前提とします
from mymodule import const, rec_config as rec_conf, rec_utility as rec_util
# reverbe_feater は rec_util 側で import される
from mymodule import my_func


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
	(mymodule/rec_utility.py の関数を利用するようにリファクタリング)
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

		# --- (ステップ4.1でリファクタリング済み) ---
		# ランダムな部屋のパラメータを生成
		room, room_dim, rt60_target, e_absorption, max_order = \
			rec_util.create_random_room_shoebox(
				room_dim_range=((3, 8), (3, 8), (2.5, 4)),
				rt60_range=(0.1, 1.0),
				fs=rec_conf.sampling_rate
			)
		# ---

		# 部屋のメタデータに情報を記録
		room_metadata = {
			"room_id": room_id,
			"room_dim": room_dim.tolist(),
			"target_rt60": rt60_target,
			"absorption": e_absorption,
			"max_order": max_order,
			"files": []
		}

		# --- (ステップ4.1でリファクタリング済み) ---
		# マイクの設置 (※ここは `channel` 引数を反映)
		mic_center = room_dim / 2
		mic_coordinate = rec_util.set_mic_coordinate(center=mic_center, num_channels=channel, distance=0.1)

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

		# RIRを計算し、特徴量を取得
		rir_signal, rir_noise, rt60, c50, d50 = \
			rec_util.compute_rir_and_features(
				room,
				mic_coordinate,
				source_pos_signal,
				source_pos_noise
			)
		# ---

		# 各部屋で指定された数のファイルを生成
		if len(speech_files) < num_files_per_room:
			print(
				f"警告: 要求されたファイル数({num_files_per_room})が利用可能なファイル数({len(speech_files)})より多いため、利用可能な全ファイルを使用します。")
			selected_speech_files = speech_files
		else:
			selected_speech_files = random.sample(speech_files, k=num_files_per_room)

		for clean_filepath in tqdm(selected_speech_files, desc=f"Generating files for room {room_id}"):
			try:
				# クリーン音声信号の読み込みと前処理
				clean_signal, fs_clean = sf.read(clean_filepath)
				if clean_signal.ndim > 1:
					clean_signal = clean_signal.mean(axis=1)

				# 雑音信号を切り出し (クリーン音声より雑音が短い場合に対応)
				if len(noise_signal_orig) <= len(clean_signal):
					repeat_times = int(np.ceil(len(clean_signal) / len(noise_signal_orig)))
					noise_signal_tiled = np.tile(noise_signal_orig, repeat_times)
				else:
					noise_signal_tiled = noise_signal_orig

				start_noise = random.randint(0, len(noise_signal_tiled) - len(clean_signal))
				noise_segment_orig = noise_signal_tiled[start_noise: start_noise + len(clean_signal)]

				# --- (ステップ4.1でリファクタリング済み) ---
				# (畳み込みと混合を rec_utility.py に移動)
				mixed_signal = rec_util.convolve_and_mix(
					clean_signal,
					noise_segment_orig,
					rir_signal,
					rir_noise,
					snr
				)
				# ---

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

	print(f"\n🎉 {output_dir} へのデータセット生成が完了しました。")


# --- ▼ ステップ 4.2: __main__ をJSON設定ファイル駆動に修正 ▼ ---
if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='ドメイン（部屋）ごとのデータセットを一括生成します')
	parser.add_argument('--config', type=str, required=True,
	                    help='処理設定が記述されたJSONファイルのパス')
	args = parser.parse_args()

	# 1. 設定ファイルを読み込む
	try:
		with open(args.config, 'r', encoding='utf-8') as f:
			config = json.load(f)
	except FileNotFoundError:
		print(f"エラー: 設定ファイルが見つかりません: {args.config}", file=sys.stderr)
		sys.exit(1)
	except json.JSONDecodeError:
		print(f"エラー: 設定ファイル({args.config})のJSON形式が正しくありません。", file=sys.stderr)
		sys.exit(1)

	print(f"設定ファイル {args.config} を読み込みました。")

	# 2. パスを変数に展開 (const.py のパスを上書き可能にする)
	base_paths = config.get('base_paths', {})

	# const.py のパスをデフォルトとし、JSONで上書き
	default_sample_dir = Path(const.SAMPLE_DATA_DIR if 'const' in globals() else './sound_data/sample_data')
	default_mix_dir = Path(const.MIX_DATA_DIR if 'const' in globals() else './sound_data/mix_data')

	speech_root = Path(base_paths.get('speech_data_root', default_sample_dir / "speech"))
	noise_root = Path(base_paths.get('noise_data_root', default_sample_dir / "noise"))
	output_root = Path(base_paths.get('output_data_root', default_mix_dir))

	# B案（ドメイン生成）用の設定を読み込む
	domain_config = config.get('domain_generation_settings', {})

	splits = config.get('splits', [])  # "train", "test" など

	# 3. ループ処理
	for split in splits:
		print(f"\n--- \"{split}\" の処理を開始 ---")

		# JSON内の "speech_type" (例: "subset_DEMAND") を使用
		speech_type = domain_config.get('speech_type', 'subset_DEMAND')
		target_dir = speech_root / speech_type / split

		# JSON内の "noise_type" (例: "hoth.wav") を使用
		noise_path = noise_root / domain_config.get('noise_type', 'hoth.wav')

		# JSON内の "output_name" (例: "reverb_encoder_dataset") を使用
		output_dir = output_root / domain_config.get('output_name', 'reverb_encoder_dataset') / split

		# ディレクトリが存在するか確認
		if not target_dir.exists():
			print(f"警告: 目的信号ディレクトリが見つかりません: {target_dir}", file=sys.stderr)
			continue
		if not noise_path.exists():
			print(f"警告: 雑音ファイルが見つかりません: {noise_path}", file=sys.stderr)
			continue

		create_reverb_dataset_final(
			target_dir=target_dir,
			noise_path=noise_path,
			output_dir=output_dir,
			num_rooms=domain_config.get('num_rooms', 10),
			num_files_per_room=domain_config.get('num_files_per_room', 20),
			snr=domain_config.get('snr', 10),
			channel=domain_config.get('channel', 1)
		)

	print("\nすべての処理が完了しました。")
# --- ▲ ステップ 4.2: 修正完了 ▲ ---