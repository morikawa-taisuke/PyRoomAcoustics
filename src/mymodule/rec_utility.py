# src/mymodule/rec_utility.py
import json
import random
from pathlib import Path

import numpy as np
import pyroomacoustics as pa
import soundfile as sf
import yaml

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


def load_json_config(config_path):
	"""JSONファイルを読み込む"""
	with open(config_path, 'r', encoding='utf-8') as f:
		config = json.load(f)
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


def get_file_list(dir_path: Path, ext: str = '.wav'):
	"""
    指定したディレクトリ内の全ての .wav ファイルを再帰的に検索する
    (my_func.py の rglob 版)
    """

	if dir_path.is_file():
		return [dir_path]
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
	array_type = mic_config['array_type']
	channels = mic_config['channel']

	# 1. アレイ中心位置の決定
	if mic_config['position_strategy'] == 'fixed_cartesian':
		center_pos = np.array(mic_config['position_coords'])
	elif mic_config['position_strategy'] == 'center':
		center_pos = room_center
	else:
		# TODO: 'random_area' などの実装
		raise NotImplementedError(f"未実装のマイク配置: {mic_config['position_strategy']}")

	# 2. アレイ形状の生成
	if channels == 1:
		return center_pos.reshape(3, 1)  # (3, 1) の形状
	else:
		if array_type == 'linear':  # 線形アレイ
			distance = mic_config['D'] * 0.01
			mic_coords = pa.linear_2D_array(
				center=[center_pos[0], center_pos[1]],
				M=channels,
				phi=0,  # X軸に平行
				d=distance
			)
			# 3Dに拡張
			mic_coords_3d = np.vstack([mic_coords, np.full(channels, center_pos[2])])
			return mic_coords_3d  # (3, M) の形状
		elif array_type == 'circular':  # 円形アレイ
			diameter = mic_config['D'] * 0.01
			radius = diameter / 2.0
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
			raise ValueError(f"未対応のアレイ形状: {array_type}")


def get_source_positions(source_config, mic_center):
	"""
    YAML設定から音源の座標リストを生成する。
    config/sample/sample.yml の 'source' セクションの構造に対応し、
    マイク中心からの相対座標で音源位置を決定する。
    """
	positions = []

	# 音源の数を決定 (デフォルトは1)
	# config/sample/sample.yml には 'count_range' がないため、デフォルト1とする
	count = get_random_value(source_config.get('count_range', [1, 1]))

	for _ in range(int(count)):
		pos_relative = None

		# 1. ランダムな球面座標範囲が指定されているかチェック (例: horizon_range: [0, 360])
		if all(k + '_range' in source_config for k in ['horizon', 'elevate', 'distance']):
			azimuth = get_random_value(source_config['horizon_range'])
			elevation = get_random_value(source_config['elevate_range'])
			distance = get_random_value(source_config['distance_range'])
			pos_relative = spherical_to_cartesian(azimuth, elevation, distance)

		# 2. 固定の球面座標が指定されているかチェック (例: horizon: 90)
		elif all(k in source_config for k in ['horizon', 'elevate', 'distance']):
			azimuth = source_config['horizon']
			elevation = source_config['elevate']
			distance = source_config['distance']
			pos_relative = spherical_to_cartesian(azimuth, elevation, distance)

		# 3. 未対応の音源配置設定
		if pos_relative is None:
			raise ValueError(
				f"未対応の音源配置設定です。'horizon', 'elevate', 'distance' またはその範囲指定が必要です: {source_config}")

		# マイク中心からの相対座標 -> 部屋の絶対座標
		positions.append(mic_center + pos_relative)

	return positions  # [np.array([x,y,z]), ...] のリスト


# --- 3. Pyroomacoustics 処理 関連 ---

def compute_rirs(room: pa.ShoeBox, mic_coords: np.ndarray,
                 speech_pos_list: list, noise_pos_list: list):
	"""
	Roomオブジェクトにマイクと音源を配置し、RIRを計算する
	(修正版: room.rir が [Mic][Source] の入れ子リストであると確定)
	"""

	# 部屋にマイクと音源を追加
	room.add_microphone_array(mic_coords)

	all_sources = speech_pos_list + noise_pos_list
	for pos in all_sources:
		room.add_source(pos)
	# RIRの計算
	room.compute_rir()

	# room.rir は [Mic][Source] の入れ子リスト
	all_rirs_list_by_mic = room.rir

	num_mics = len(all_rirs_list_by_mic)
	num_speech = len(speech_pos_list)
	num_noise = len(noise_pos_list)

	# 話者RIRの転置
	rir_speech_list = []
	# (音源 0 から num_speech-1 までループ)
	for s_idx in range(num_speech):
		# この音源 (s_idx) のRIRを、全マイクから収集
		rirs_for_this_source = []
		max_len = 0
		for m_idx in range(num_mics):
			rir = all_rirs_list_by_mic[m_idx][s_idx]  # [Mic][Source] でアクセス
			rirs_for_this_source.append(rir)
			if len(rir) > max_len:
				max_len = len(rir)

		# (長さが異なる場合があるため、最長のRIRに合わせてパディング)
		for rir in rirs_for_this_source:
			padded = np.zeros(max_len, dtype=rir.dtype)
			padded[:len(rir)] = rir
			rir_speech_list.append(padded)

		# 全マイク (C) のRIRをスタック -> (C, N) のNumpy配列
		# rir_speech_list.append(np.array(padded_rirs))

	# ノイズRIRの転置
	rir_noise_list = []
	# (音源 num_speech から最後までループ)
	for n_idx in range(num_noise):
		s_idx = num_speech + n_idx  # 元のリストでのインデックス

		rirs_for_this_source = []
		max_len = 0
		for m_idx in range(num_mics):
			rir = all_rirs_list_by_mic[m_idx][s_idx]  # [Mic][Source] でアクセス
			rirs_for_this_source.append(rir)
			if len(rir) > max_len:
				max_len = len(rir)

		for rir in rirs_for_this_source:
			padded = np.zeros(max_len, dtype=rir.dtype)
			padded[:len(rir)] = rir
			rir_noise_list.append(padded)

		# rir_noise_list.append(np.array(padded_rirs))

	rir_dict = {
		# 'rir_speech' は [ array(C, N_s0), array(C, N_s1), ... ] のリスト
		'rir_speech': np.array(rir_speech_list),
		# 'rir_noise' は [ array(C, N_n0), array(C, N_n1), ... ] のリスト
		'rir_noise': np.array(rir_noise_list)
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


def convolve_and_mix(clean_signal: np.ndarray, noise_signal: np.ndarray, rir_speech: np.ndarray, rir_noise: np.ndarray, snr_db: float) -> dict:
	"""
	各信号とRIRを畳み込み、指定したSNRで混合する
	(修正版: ユーザー定義に基づき "noise_only" (clean + dry_noise) を生成)

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

	# 3. 畳み込み (scipy.signal.fftconvolve)

	# (N,) -> (N, 1)
	clean_signal_col = clean_signal[:, np.newaxis]
	noise_segment_col = noise_segment[:, np.newaxis]

	# (C, N_rir) -> (N_rir, C)
	rir_speech_col = rir_speech.T
	rir_noise_col = rir_noise.T
	num_channels = rir_speech_col.shape[1]

	clean_signal_multi = np.tile(clean_signal_col, (1, num_channels))
	noise_segment_multi = np.tile(noise_segment_col, (1, num_channels))

	# print(rir_speech_col.shape, clean_signal_multi.shape)
	# print(rir_noise_col.shape, noise_segment_multi.shape)

	# 畳み込み
	# (N_out, C) の形状になる
	from scipy.signal import fftconvolve
	reverb_signal = fftconvolve(clean_signal_multi, rir_speech_col, mode='full', axes=0)[:target_len]
	reverb_noise = fftconvolve(noise_segment_multi, rir_noise_col, mode='full', axes=0)[:target_len]

	# 4. SNR調整
	scaled_reverb_noise = get_scale_noise(reverb_signal, reverb_noise, snr_db)  # noise_reverb用
	scaled_dry_noise = get_scale_noise(clean_signal, noise_segment, snr_db) # noise_only用

	# 5. 混合 (残響あり + 残響ありノイズ)
	mixed_signal = reverb_signal + scaled_reverb_noise  # noise_reverb

	# (N,) + (N,) = (N,)
	noise_only_signal_mono = clean_signal + scaled_dry_noise    # noise_only

	# 6. 出力形式 (N, C) に整形
	num_channels = reverb_signal.shape[1]

	# (N,) -> (N, C)
	clean_speech_target = np.tile(clean_signal_col, (1, num_channels))  # clean

	# (N,) -> (N, 1) -> (N, C)
	noise_only_signal_col = noise_only_signal_mono[:, np.newaxis]
	noise_only_target = np.tile(noise_only_signal_col, (1, num_channels))

	# 6. 音量調整
	max = np.max(np.abs(clean_signal))
	mixed_signal = mixed_signal / np.max(np.abs(mixed_signal)) * max
	reverb_signal = reverb_signal / np.max(np.abs(reverb_signal)) * max
	noise_only_target = noise_only_target / np.max(np.abs(noise_only_target)) * max
	clean_speech_target = clean_speech_target / np.max(np.abs(clean_speech_target)) * max


	return {
		"noise_reverb": mixed_signal,  # (N_out, C)
		"reverb_only": reverb_signal,  # (N_out, C)
		"noise_only": noise_only_target,  # (N_out, C) <- NEW
		"clean_speech": clean_speech_target  # (N_out, C)
	}
