# src/mymodule/rec_utility.py

import numpy as np
import pyroomacoustics as pa
import soundfile as sf
import yaml
import random
from pathlib import Path

# --- (旧: rec_config.py の内容) ---
# simulation.py または constants.py に移動することを推奨
SAMPLING_RATE = 16000


# --- (旧: reverb_feater.py の内容) ---
# audio.py に移動することを推奨
def calculate_c50(rir, fs=SAMPLING_RATE):
	t_50ms = int(0.050 * fs)
	energy = rir ** 2
	e_early = np.sum(energy[:t_50ms])
	e_late = np.sum(energy[t_50ms:])
	if e_late > 0:
		c50 = 10 * np.log10(e_early / e_late)
	else:
		c50 = np.inf
	return c50


def calculate_d50(rir, fs=SAMPLING_RATE):
	t_50ms = int(0.050 * fs)
	energy = rir ** 2
	e_early = np.sum(energy[:t_50ms])
	e_total = np.sum(energy)
	if e_total > 0:
		d50 = (e_early / e_total) * 100
	else:
		d50 = 0.0
	return d50


# ===================================================================
# === ▼▼▼ 新しいワークフローのための関数群 ▼▼▼ ===
# ===================================================================

# --- 1. 設定ファイル・I/O 関連 ---

def load_yaml_config(config_path):
	"""YAML設定ファイルを読み込む"""
	with open(config_path, 'r', encoding='utf-8') as f:
		config = yaml.safe_load(f)
	return config


def load_wav(filepath, sr=SAMPLING_RATE):
	"""
    soundfileを使用してWAVファイルを読み込む
    (リサンプリングとモノラル化も行う)
    """
	data, loaded_sr = sf.read(filepath, dtype='float32', always_2d=True)

	# モノラルに変換
	if data.shape[1] > 1:
		data = np.mean(data, axis=1)
	else:
		data = data[:, 0]

	# リサンプリング (TODO: 必要ならresampyなどを追加)
	if loaded_sr != sr:
		raise NotImplementedError(
			f"Resampling required: {filepath} ({loaded_sr}Hz) != target ({sr}Hz)"
		)

	return data, sr


def save_wav(filepath: Path, data: np.ndarray, sr=SAMPLING_RATE):
	"""
    soundfileを使用してWAVファイルを保存する (マルチチャンネル対応)
    (N,) または (N, C) のNumpy配列を受け取る
    """
	filepath.parent.mkdir(parents=True, exist_ok=True)
	sf.write(filepath, data, sr)


def get_file_list(dir_path: Path, ext: str = '.wav') -> list[Path]:
	"""
    指定したディレクトリ内の全ての .wav ファイルを再帰的に検索する
    (my_func.py の rglob 版)
    """
	if not dir_path.is_dir():
		raise FileNotFoundError(f"ディレクトリが見つかりません: {dir_path}")
	return sorted(list(dir_path.rglob(f"*{ext}")))


# --- 2. 座標計算・ランダム化 関連 ---

def get_random_value(param):
	"""
    [min, max] のリストからランダムな値を取得する
    min == max の場合は固定値として返す
    """
	if isinstance(param, (list, tuple)) and len(param) == 2:
		return random.uniform(param[0], param[1])
	# スカラー値の場合はそのまま返す
	elif isinstance(param, (int, float)):
		return param
	else:
		raise ValueError(f"不正な範囲指定です: {param}")


def spherical_to_cartesian(azimuth_deg, elevation_deg, distance):
	"""
    極座標（度数法）をデカルト座標に変換する
    (pyroomacousticsの規約: azimuthはX軸からY軸方向, elevationはXY平面からZ軸方向)
    """
	azimuth_rad = np.deg2rad(azimuth_deg)
	elevation_rad = np.deg2rad(elevation_deg)

	x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
	y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
	z = distance * np.sin(elevation_rad)
	return np.array([x, y, z])


def get_mic_array(mic_config, room_center):
	"""
    YAML設定からマイクアレイの座標を生成する
    """
	shape = mic_config['array']['shape']
	channels = mic_config['array']['channels']

	# 1. アレイ中心位置の決定
	if mic_config['position_strategy'] == 'fixed_cartesian':
		center_pos = np.array(mic_config['position_coords'])
	elif mic_config['position_strategy'] == 'center':
		center_pos = room_center
	else:
		# TODO: 'random_area' などの実装
		raise NotImplementedError(f"未実装のマイク配置: {mic_config['position_strategy']}")

	# 2. アレイ形状の生成
	if shape == 'single':
		if channels != 1:
			print(f"警告: shape='single' のため、channels=1 に強制します。")
		return center_pos.reshape(3, 1)  # (3, 1) の形状

	elif shape == 'linear':
		spacing = mic_config['array']['spacing']
		# (旧: rec_utility.py の set_mic_coordinate)
		mic_coords = pa.linear_2D_array(
			center=[center_pos[0], center_pos[1]],
			M=channels,
			phi=0,  # X軸に平行
			d=spacing
		)
		# 3Dに拡張
		mic_coords_3d = np.vstack([mic_coords, np.full(channels, center_pos[2])])
		return mic_coords_3d  # (3, M) の形状

	elif shape == 'circular':
		diameter = mic_config['array']['diameter']
		radius = diameter / 2.0
		# (旧: rec_utility.py の set_circular_mic_coordinate)
		mic_coords = pa.circular_2D_array(
			center=[center_pos[0], center_pos[1]],
			M=channels,
			phi0=0,
			radius=radius
		)
		# 3Dに拡張
		mic_coords_3d = np.vstack([mic_coords, np.full(channels, center_pos[2])])
		return mic_coords_3d  # (3, M) の形状

	else:
		raise ValueError(f"未対応のアレイ形状: {shape}")


def get_source_positions(source_config, mic_center):
	"""
    YAML設定から音源の座標リストを生成する
    """
	strategy = source_config['position_strategy']
	count = get_random_value(source_config.get('count_range', [1, 1]))

	positions = []
	for _ in range(int(count)):
		if strategy == 'fixed_spherical':
			params = source_config['params']
			pos_relative = spherical_to_cartesian(
				params['azimuth'], params['elevation'], params['distance']
			)

		elif strategy == 'random_spherical_range':
			params = source_config['params']
			azimuth = get_random_value(params['azimuth_range'])
			elevation = get_random_value(params['elevation_range'])
			distance = get_random_value(params['distance_range'])
			pos_relative = spherical_to_cartesian(azimuth, elevation, distance)

		elif strategy == 'random_area':
			# (旧: new_signal_noise.py のロジック)
			# TODO: room_dim を引数で受け取る必要がある
			raise NotImplementedError("random_area は room_dim が必要なため未実装")

		else:
			raise ValueError(f"未対応の音源配置: {strategy}")

		# マイク中心からの相対座標 -> 部屋の絶対座標
		positions.append(mic_center + pos_relative)

	return positions  # [np.array([x,y,z]), ...] のリスト


# --- 3. Pyroomacoustics 処理 関連 ---

def compute_rirs(room: pa.ShoeBox, mic_coords: np.ndarray,
                 speech_pos_list: list, noise_pos_list: list):
	"""
    Roomオブジェクトにマイクと音源を配置し、RIRを計算する

    Returns:
        dict: {'rir_speech': (M, C, N), 'rir_noise': (M, C, N)}
              M=音源数, C=チャンネル数, N=サンプル数
    """
	# マイクを設置
	room.add_microphone_array(mic_coords)

	# 音源を設置
	all_sources = speech_pos_list + noise_pos_list
	for pos in all_sources:
		room.add_source(pos)

	# RIR計算
	room.compute_rir()

	# RIRを分離
	num_speech = len(speech_pos_list)

	# room.rir は [(C, N), (C, N), ...] (音源数Mのリスト)
	# (M, C, N) の形状にスタックする
	all_rirs = np.array(room.rir)

	rir_dict = {
		'rir_speech': all_rirs[:num_speech],
		'rir_noise': all_rirs[num_speech:]
	}

	return rir_dict


def get_wave_power(wave_data):
	"""音源のパワーを計算する (旧: rec_utility.py)"""
	# チャンネル全体で平均パワーを計算
	return np.mean(wave_data ** 2)


def get_scale_noise(signal_data, noise_data, snr_db):
	"""指定したSNRに雑音の大きさを調整 (旧: rec_utility.py)"""
	signal_power = get_wave_power(signal_data)
	noise_power = get_wave_power(noise_data)

	if noise_power < 1e-10:
		return np.zeros_like(noise_data)

	target_noise_power = signal_power / (10 ** (snr_db / 10))
	noise_scale = np.sqrt(target_noise_power / noise_power)

	return noise_data * noise_scale


def convolve_and_mix(
		clean_signal: np.ndarray,
		noise_signal: np.ndarray,
		rir_speech: np.ndarray,
		rir_noise: np.ndarray,
		snr_db: float
) -> dict:
	"""
    各信号とRIRを畳み込み、指定したSNRで混合する
    (B案拡張: 複数の信号を辞書で返す)

    Args:
        clean_signal (N,): モノラルクリーン音声
        noise_signal (N_noise,): モノラルノイズ
        rir_speech (C, N_rir): 話者用RIR (チャンネル, サンプル)
        rir_noise (C, N_rir): ノイズ用RIR (チャンネル, サンプル)
        snr_db (float): 混合SNR

    Returns:
        dict: 各種信号 ( (N_out, C) のNumpy配列 )
    """

	# 1. 信号の長さを決定
	target_len = len(clean_signal)

	# 2. 雑音の長さを調整 (ランダムクロップ)
	if len(noise_signal) <= target_len:
		repeat_times = int(np.ceil(target_len / len(noise_signal)))
		noise_signal_tiled = np.tile(noise_signal, repeat_times)
	else:
		noise_signal_tiled = noise_signal

	start_noise = random.randint(0, len(noise_signal_tiled) - target_len)
	noise_segment = noise_signal_tiled[start_noise: start_noise + target_len]

	# 3. 畳み込み (scipy.signal.fftconvolve を推奨するが、
	#    process_audio2.py 同様の実装)

	# (N,) -> (N, 1)
	clean_signal_col = clean_signal[:, np.newaxis]
	noise_segment_col = noise_segment[:, np.newaxis]

	# (C, N_rir) -> (N_rir, C)
	rir_speech_col = rir_speech.T
	rir_noise_col = rir_noise.T

	# 畳み込み (fftconvolveが望ましいが、簡易的に np.convolve を使う)
	# (N_out, C) の形状になる
	from scipy.signal import fftconvolve
	reverb_signal = fftconvolve(clean_signal_col, rir_speech_col, mode='full', axes=0)[:target_len]
	reverb_noise = fftconvolve(noise_segment_col, rir_noise_col, mode='full', axes=0)[:target_len]

	# 4. SNR調整
	scaled_noise = get_scale_noise(reverb_signal, reverb_noise, snr_db)

	# 5. 混合
	mixed_signal = reverb_signal + scaled_noise

	# 6. 教師データ（クリーン）もチャンネル数と長さを合わせる
	# (N,) -> (N, C)
	num_channels = reverb_signal.shape[1]
	clean_speech_target = np.tile(clean_signal_col, (1, num_channels))

	return {
		"mixture": mixed_signal,  # (N_out, C)
		"reverberant_speech": reverb_signal,  # (N_out, C)
		"reverberant_noise": scaled_noise,  # (N_out, C)
		"clean_speech": clean_speech_target  # (N_out, C)
	}