import argparse
import json
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from pathlib import Path
import random
from tqdm import tqdm
import sys

from mymodule import const
from mymodule import rec_utility as rec_util

# ===================================================================
# ▼▼▼ 設定項目 ▼▼▼
# ===================================================================

# --- 入力設定 ---
# 前のステップで作成したJSONファイルのパス
DEFAULT_JSON_PATH = "vctk_split_file_list.json"

# --- 出力設定 ---
# 生成したデータセットを保存する親ディレクトリ
DEFAULT_OUTPUT_DIR = Path("C:/Users/kataoka-lab/Desktop/sound_data/mix_data/vctk_reverb_noise")

# --- 雑音設定 ---
# 使用する雑音ファイルのパス (ご自身の環境に合わせて変更してください)
# この雑音ファイルは、シミュレーション中にランダムな箇所が切り取られて使用されます。
DEFAULT_NOISE_PATH = Path(f"{const.SAMPLE_DATA_DIR}\\noise\\hoth.wav")

# --- シミュレーションのランダムパラメータ範囲 ---
# 部屋の大きさ [x, y, z] (メートル)
ROOM_DIM_RANGE = {
	'x': (3, 3),
	'y': (3, 3),
	'z': (3, 3)
}

# 残響時間 RT60 (秒)
RT60_RANGE = (0.5, 0.5)

# 信号対雑音比 SNR (dB)
SNR_RANGE = (5, 5)


# ===================================================================
# ▲▲▲ 設定項目 ▲▲▲
# ===================================================================

def create_augmented_dataset(json_path: Path, output_dir: Path, noise_path: Path):
	"""
	JSONファイルに基づき、音声にランダムな残響と雑音を付与してデータセットを生成する。
	"""
	# ---- 1. 入力ファイルのチェック ----
	if not json_path.is_file():
		print(f"❌ エラー: 入力JSONファイルが見つかりません: {json_path}", file=sys.stderr)
		sys.exit(1)
	if not noise_path.is_file():
		print(f"❌ エラー: 雑音ファイルが見つかりません: {noise_path}", file=sys.stderr)
		sys.exit(1)

	print("✅ 入力ファイルのチェック完了。")
	print(f"📖 JSON入力: {json_path}")
	print(f"🔊 雑音ファイル: {noise_path}")
	print(f"💾 出力先: {output_dir}")

	# ---- 2. データの読み込み ----
	with open(json_path, 'r', encoding='utf-8') as f:
		all_splits_info = json.load(f)

	fs_noise, noise_signal = wavfile.read(noise_path)
	# モノラルに変換
	if noise_signal.ndim > 1:
		noise_signal = noise_signal.mean(axis=1)

	# ---- 3. データセット生成ループ ----
	for split_name, speakers_data in all_splits_info.items():
		print(f"\n======== {split_name.upper()} セットの処理を開始 ========")

		# tqdmを使って進捗バーを表示
		file_list = []
		for speaker_id, data in speakers_data.items():
			for filepath in data["filepaths"]:
				file_list.append((speaker_id, Path(filepath)))

		for speaker_id, clean_filepath in tqdm(file_list, desc=f"Processing {split_name}"):
			try:
				# --- 3a. 音響環境をランダムに設定 ---
				room_dim = [
					random.uniform(*ROOM_DIM_RANGE['x']),
					random.uniform(*ROOM_DIM_RANGE['y']),
					random.uniform(*ROOM_DIM_RANGE['z'])
				]
				rt60_target = random.uniform(*RT60_RANGE)
				snr_target = random.uniform(*SNR_RANGE)

				# Sabineの式から壁の吸収率と最大反射回数を計算
				e_absorption, max_order = pra.inverse_sabine(rt60_target, room_dim)

				# --- 3b. 部屋を作成 ---
				room = pra.ShoeBox(
					room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order
				)

				# --- 3c. 音源とマイクを配置 ---
				fs_clean, clean_signal = wavfile.read(clean_filepath)
				# サンプリングレートを部屋と合わせる（必要ならリサンプリング）
				if fs_clean != room.fs:
					# ここでは簡単化のためエラーとするが、実際はリサンプリング処理が望ましい
					tqdm.write(f"⚠️  サンプリングレートが異なります: {clean_filepath.name} ({fs_clean}Hz)。スキップします。")
					continue

				# モノラルに変換
				if clean_signal.ndim > 1:
					clean_signal = clean_signal.mean(axis=1)

				# 音源とマイクの位置をランダムに設定 (壁から20cmは離す)
				mic_pos = room_dim // 2
				doas = np.array([
					[np.pi / 2., np.pi / 2],
					[np.pi / 2., 0]
				])  # 音源の方向[仰角, 方位角](ラジアン)
				distance = [0.5, 0.7]  # 音源とマイクの距離(m)
				source_pos = rec_util.set_souces_coordinate2(doas, distance, mic_pos)

				room.add_source(source_pos, signal=clean_signal)
				room.add_microphone_array(pra.MicrophoneArray(mic_pos.reshape(-1, 1), room.fs))

				# --- 3d. 雑音を追加 ---
				# 雑音信号からランダムな箇所を切り出して使用
				start = random.randint(0, len(noise_signal) - len(clean_signal))
				noise_segment = noise_signal[start: start + len(clean_signal)]

				room.add_source(source_pos, signal=noise_segment, snr=snr_target)

				# --- 3e. シミュレーション実行 ---
				# anechoic(clean), reverb, noisy(mic_array)のシグナルを分離して計算
				room.sources[1].power = 0.  # 一時的にノイズをオフ
				room.simulate(snr=None)  # snr=None でないと古い挙動になる
				reverb_signal = room.mic_array.signals[0, :len(clean_signal)]

				room.sources[1].power = 1.  # ノイズをオンに戻す
				room.simulate(snr=snr_target)
				noisy_signal = room.mic_array.signals[0, :len(clean_signal)]

				# --- 3f. ファイルを保存 ---
				# 出力ディレクトリを作成
				output_sub_dir = output_dir / split_name / speaker_id
				output_sub_dir.mkdir(parents=True, exist_ok=True)

				base_filename = clean_filepath.stem

				# 各ファイルを正規化して保存
				def save_wav(path, signal, fs):
					# floatを16-bit intに変換
					signal_norm = signal / np.max(np.abs(signal)) * 0.9
					wavfile.write(path, fs, (signal_norm * 32767).astype(np.int16))

				save_wav(output_sub_dir / f"{base_filename}_clean.wav", clean_signal, fs_clean)
				save_wav(output_sub_dir / f"{base_filename}_reverb.wav", reverb_signal, fs_clean)
				save_wav(output_sub_dir / f"{base_filename}_noisy.wav", noisy_signal, fs_clean)

			except Exception as e:
				tqdm.write(f"❌ エラーが発生しました: {clean_filepath.name} ({e})")

	print("\n🎉 全ての処理が完了しました。")


if __name__ == "__main__":
	# コマンドライン引数の設定
	parser = argparse.ArgumentParser(description="JSONファイルに基づき、音声に残響と雑音を付与するデータセットを生成します。")
	parser.add_argument(
		"--json_path", type=Path, default=DEFAULT_JSON_PATH,
		help=f"入力となるJSONファイルのパス (デフォルト: {DEFAULT_JSON_PATH})"
	)
	parser.add_argument(
		"--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
		help=f"生成したデータセットを保存するディレクトリ (デフォルト: {DEFAULT_OUTPUT_DIR})"
	)
	parser.add_argument(
		"--noise_path", type=Path, default=DEFAULT_NOISE_PATH,
		help=f"シミュレーションで使用する雑音ファイルのパス (デフォルト: {DEFAULT_NOISE_PATH})"
	)

	args = parser.parse_args()

	create_augmented_dataset(
		json_path=args.json_path,
		output_dir=args.output_dir,
		noise_path=args.noise_path
	)