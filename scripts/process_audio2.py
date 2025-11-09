import os
import argparse
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from tqdm import tqdm
from tqdm.contrib import tzip
import random
import json  # JSONモジュールをインポート
from src import my_func


def load_wav(filepath):
	"""
    soundfileを使用してWAVファイルを読み込む
    (モノラル: (N,), マルチチャンネル: (N, C))
    """
	data, sr = sf.read(filepath, dtype='float32')  # 浮動小数点数で読み込む
	return data, sr


def save_wav(filepath, data, sr):
	"""
    soundfileを使用してWAVファイルを保存する
    (モノラル: (N,), マルチチャンネル: (N, C) に対応)
    """
	sf.write(filepath, data, sr)


def random_crop(noise, target_length):
	"""
    ノイズを指定した長さにランダムに切り出す
    """
	if len(noise) <= target_length:
		# 長さが足りない場合はループして埋める
		repeat_times = int(np.ceil(target_length / len(noise)))
		noise = np.tile(noise, repeat_times)
	start = np.random.randint(0, len(noise) - target_length + 1)
	return noise[start:start + target_length]


# --- ▼ ステップ 3.1: apply_ir 関数をマルチチャンネル対応に修正 ▼ ---
def apply_ir(signal, ir):
	"""
    モノラル信号 (N,) に
    モノラルRIR (M,) または マルチチャンネルRIR (M, C) を畳み込む

    Args:
        signal (np.ndarray): モノラル音声信号 (N,)
        ir (np.ndarray): RIR (M,) または (M, C)

    Returns:
        np.ndarray: 畳み込み後信号 (N,) または (N, C)
    """

	# 畳み込み対象のRIRがマルチチャンネル(2D)の場合
	if ir.ndim > 1:
		# モノラル信号 (N,) を (N, 1) にリシェイプ (ブロードキャストのため)
		if signal.ndim == 1:
			signal_reshaped = signal[:, np.newaxis]
		else:
			signal_reshaped = signal  # 既に2D以上ならそのまま

		# fftconvolveで、畳み込みの軸を 0 (時間軸) に指定
		# (N, 1) と (M, C) がブロードキャストされ、(L, C) の結果が得られる
		convolved = fftconvolve(signal_reshaped, ir, mode='full', axes=0)

	# RIRがモノラル(1D)の場合
	else:
		convolved = fftconvolve(signal, ir, mode='full')

	# 元の信号の長さ (N) に切り詰める
	return convolved[:len(signal)]


# --- ▲ ステップ 3.1: 修正完了 ▲ ---

# --- ▼ ステップ 3.2: mix_snr 関数をマルチチャンネル/無音に対応するよう修正 ▼ ---
def mix_snr(speech, noise, snr_db):
	"""
    音声(speech)と雑音(noise)をSNR(dB)に基づいて混合する
    (N,) , (N, 1), (N, C) のNumpy配列に対応
    """
	# パワー計算 (時間軸とチャンネル軸で平均)
	speech_power = np.mean(speech ** 2)
	noise_power = np.mean(noise ** 2)

	# 雑音がほぼ無音(1e-10未満)の場合は、ゲイン計算でのゼロ除算を避ける
	if noise_power < 1e-10:
		return speech

	# 目的の雑音パワー
	target_noise_power = speech_power / (10 ** (snr_db / 10))

	# 雑音にかけるゲイン（スケーリング係数）
	noise_scale = np.sqrt(target_noise_power / noise_power)

	return speech + noise * noise_scale


# --- ▲ ステップ 3.2: 修正完了 ▲ ---


def clean(speech_dir, ir_path, output_dir):
	my_func.exists_dir(output_dir)

	speech_files = my_func.get_file_list(speech_dir)

	print("-" * 32)
	print(f"speech_type({len(speech_files)}): {speech_dir}")
	print(f"ir_dir: {ir_path}")
	print(f"output_dir: {output_dir}")
	print("-" * 32)

	for speech_file in tqdm(speech_files):
		# 読み込み
		speech, sr = load_wav(speech_file)
		speech_ir, _ = load_wav(ir_path[0])  # speech_ir は (M,) または (M, C)

		# IR畳み込み (apply_ir が (N,) または (N, C) を返す)
		speech_reverb = apply_ir(speech, speech_ir)

		# 保存
		out_name = f"{my_func.get_fname(speech_file)[0]}.wav"
		out_path = os.path.join(output_dir, out_name)
		save_wav(out_path, speech_reverb, sr)


def noise_reverbe(speech_dir, noise_dir, ir_path, output_dir):
	print("-" * 32)
	print(f"speech_type: {speech_dir}")
	print(f"noise_dir: {noise_dir}")
	print(f"ir_dir: {ir_path}")
	print(f"output_dir: {output_dir}")
	print("-" * 32)

	my_func.exists_dir(output_dir)

	speech_files = my_func.get_file_list(speech_dir)

	# noise_dir からランダムに選択
	noise_file_list = my_func.get_file_list(noise_dir)
	if not noise_file_list:
		print(f"エラー: 雑音ファイルが見つかりません: {noise_dir}")
		return
	noise_files = [random.choice(noise_file_list) for _ in range(len(speech_files))]

	snr = 0.0  # SNRは0に設定
	reverbe = 0.5  # reverberation timeは0.5秒に設定 (ファイル名用)

	for (speech_file, noise_file) in tzip(speech_files, noise_files):
		# ファイルパス生成
		speech_ir_path = ir_path[0]
		noise_ir_path = ir_path[1]

		# 読み込み
		speech, sr = load_wav(speech_file)  # (N,)
		noise, _ = load_wav(noise_file)  # (N_noise,)
		speech_ir, _ = load_wav(speech_ir_path)  # (M,) or (M, C)
		noise_ir, _ = load_wav(noise_ir_path)  # (M,) or (M, C)

		# noise切り出し
		noise = random_crop(noise, len(speech))  # (N,)

		# IR畳み込み
		speech_reverb = apply_ir(speech, speech_ir)  # (N,) or (N, C)
		noise_reverb = apply_ir(noise, noise_ir)  # (N,) or (N, C)

		# SNR調整
		mixed = mix_snr(speech_reverb, noise_reverb, snr)  # (N,) or (N, C)

		# 保存
		out_name = f"{my_func.get_fname(speech_file)[0]}_{my_func.get_fname(noise_file)[0]}_{int(snr * 10):03}dB_{int(reverbe * 1000)}msec.wav"
		out_path = os.path.join(output_dir, out_name)
		save_wav(out_path, mixed, sr)


def reverbe_only(speech_dir, ir_path, output_dir):
	print("-" * 32)
	print(f"speech_type: {speech_dir}")
	print(f"ir_dir: {ir_path}")
	print(f"output_dir: {output_dir}")
	print("-" * 32)

	my_func.exists_dir(output_dir)

	speech_files = my_func.get_file_list(speech_dir)

	reverbe = 0.5  # reverberation timeは0.5秒に設定 (ファイル名用)
	for speech_file in tqdm(speech_files):
		# 読み込み
		speech, sr = load_wav(speech_file)
		speech_ir, _ = load_wav(ir_path[0])  # (M,) or (M, C)

		# IR畳み込み
		speech_reverb = apply_ir(speech, speech_ir)  # (N,) or (N, C)

		# 保存
		out_name = f"{my_func.get_fname(speech_file)[0]}_{int(reverbe * 1000)}msec.wav"
		out_path = os.path.join(output_dir, out_name)
		save_wav(out_path, speech_reverb, sr)


def noise_only(speech_dir, noise_dir, ir_path, output_dir):
	print("-" * 32)
	print(f"speech_type: {speech_dir}")
	print(f"noise_dir: {noise_dir}")
	print(f"ir_dir: {ir_path}")
	print(f"output_dir: {output_dir}")
	print("-" * 32)

	my_func.exists_dir(output_dir)

	speech_files = my_func.get_file_list(speech_dir)

	# noise_dir からランダムに選択
	noise_file_list = my_func.get_file_list(noise_dir)
	if not noise_file_list:
		print(f"エラー: 雑音ファイルが見つかりません: {noise_dir}")
		return
	noise_files = [random.choice(noise_file_list) for _ in range(len(speech_files))]

	snr = 0.0  # SNRは0に設定
	for (speech_file, noise_file) in tzip(speech_files, noise_files):
		# ファイルパス生成
		speech_ir_path = ir_path[0]
		noise_ir_path = ir_path[1]

		# 読み込み
		speech, sr = load_wav(speech_file)
		noise, _ = load_wav(noise_file)
		speech_ir, _ = load_wav(speech_ir_path)
		noise_ir, _ = load_wav(noise_ir_path)

		# noise切り出し
		noise = random_crop(noise, len(speech))

		# IR畳み込み
		speech_reverb = apply_ir(speech, speech_ir)
		noise_reverb = apply_ir(noise, noise_ir)

		# SNR調整
		# ※ noise_only だが、オリジナルのコード では speech_reverb と noise_reverb を mix_snr している
		mixed = mix_snr(speech_reverb, noise_reverb, snr)

		# 保存
		out_name = f"{my_func.get_fname(speech_file)[0]}_{my_func.get_fname(noise_file)[0]}_{int(snr * 10):03}dB.wav"
		out_path = os.path.join(output_dir, out_name)
		save_wav(out_path, mixed, sr)


# --- ▼ ステップ 3.2: argparse と __main__ をJSON設定ファイル駆動に修正 ▼ ---
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='設定ファイル(JSON)に従い音声データを処理')

	# argparse を --config のみを必須引数とするように変更
	parser.add_argument('--config', type=str, required=True,
	                    help='処理設定が記述されたJSONファイルのパス')

	args = parser.parse_args()

	# 1. 設定ファイルを読み込む
	try:
		with open(args.config, 'r', encoding='utf-8') as f:
			config = json.load(f)
	except FileNotFoundError:
		print(f"エラー: 設定ファイルが見つかりません: {args.config}")
		exit()
	except json.JSONDecodeError:
		print(f"エラー: 設定ファイル({args.config})のJSON形式が正しくありません。")
		exit()

	# 2. パスを変数に展開
	base_paths = config.get('base_paths', {})
	speech_root = base_paths.get('speech_data_root', '')
	noise_root = base_paths.get('noise_data_root', '')
	ir_root = base_paths.get('ir_data_root', '')
	output_root = base_paths.get('output_data_root', '')

	splits = config.get('splits', [])
	tasks = config.get('tasks', {})

	print(f"設定ファイル {args.config} を読み込みました。")
	print(f"処理対象: {splits}")

	# 3. ループ処理
	for split in splits:
		print(f"\n--- \"{split}\" の処理を開始 ---")
		speech_dir = os.path.join(speech_root, split)

		# --- clean ---
		task_cfg = tasks.get('clean', {})
		if task_cfg.get('enabled', False):
			print("  タスク: clean")
			output_dir = os.path.join(output_root, split, 'clean')
			# IRパスはリストとして渡す (clean関数は ir_path[0] を参照する)
			ir_paths = [
				os.path.join(ir_root, task_cfg.get('speech_ir_path', ''))
			]
			clean(speech_dir, ir_paths, output_dir)

		# --- reverbe_only ---
		task_cfg = tasks.get('reverbe_only', {})
		if task_cfg.get('enabled', False):
			print("  タスク: reverbe_only")
			output_dir = os.path.join(output_root, split, 'reverbe_only')
			ir_paths = [
				os.path.join(ir_root, task_cfg.get('speech_ir_path', ''))
			]
			reverbe_only(speech_dir, ir_paths, output_dir)

		# --- noise_only ---
		task_cfg = tasks.get('noise_only', {})
		if task_cfg.get('enabled', False):
			print("  タスク: noise_only")
			output_dir = os.path.join(output_root, split, 'noise_only')
			noise_dir = os.path.join(noise_root, task_cfg.get('noise_type', ''))
			ir_paths = [
				os.path.join(ir_root, task_cfg.get('speech_ir_path', '')),
				os.path.join(ir_root, task_cfg.get('noise_ir_path', ''))
			]
			noise_only(speech_dir, noise_dir, ir_paths, output_dir)

		# --- noise_reverbe ---
		task_cfg = tasks.get('noise_reverbe', {})
		if task_cfg.get('enabled', False):
			print("  タスク: noise_reverbe")
			output_dir = os.path.join(output_root, split, 'noise_reverbe')
			noise_dir = os.path.join(noise_root, task_cfg.get('noise_type', ''))
			ir_paths = [
				os.path.join(ir_root, task_cfg.get('speech_ir_path', '')),
				os.path.join(ir_root, task_cfg.get('noise_ir_path', ''))
			]
			noise_reverbe(speech_dir, noise_dir, ir_paths, output_dir)

	print("\nすべての処理が完了しました。")
# --- ▲ ステップ 3.2: 修正完了 ▲ ---