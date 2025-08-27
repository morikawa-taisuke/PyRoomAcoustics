import numpy as np
import soundfile as sf
from GPURIR import pyroomacoustics as gpura


def generate_reverberant_audio(
		audio_path: str,
		noise_path: str,
		output_path: str,
		room_params: dict,
		source_params: dict,
		fs: int = 16000,
		snr_db: float = 10.0
):
	"""
	音声信号に指定された残響と雑音を付加し、マルチチャンネル音声を作成する関数。

	Args:
		audio_path (str): 元の音声ファイルのパス。
		noise_path (str): 元の雑音ファイルのパス。
		output_path (str): 出力ファイルの保存パス。
		room_params (dict): 部屋のパラメータ（'dim', 'rt60', 'rir_length'）。
		source_params (dict): 音源と雑音源の位置（'speech', 'noise'）。
		fs (int, optional): サンプリング周波数。Defaults to 16000.
		snr_db (float, optional): 信号対雑音比（SNR）。Defaults to 10.0.
	"""
	try:
		clean_speech, _ = sf.read(audio_path, dtype='float32')
		clean_noise, _ = sf.read(noise_path, dtype='float32')
	except FileNotFoundError as e:
		print(f"Error: {e}. Please check file paths.")
		return

	# 部屋と音源の設定
	room_dim = room_params['dim']
	rt60 = room_params['rt60']
	rir_length = room_params['rir_length']
	speech_pos = source_params['speech']
	noise_pos = source_params['noise']
	mic_pos = np.array(source_params['mics'])
	nb_mics = mic_pos.shape[0]

	# 音声信号用のRIRを生成
	rir_speech = gpura.generate_rir(
		room_sz=room_dim,
		pos_src=[speech_pos],
		pos_rcv=mic_pos,
		Tmax=rir_length,
		fs=fs,
		rt60=rt60
	)

	# 雑音信号用のRIRを生成
	rir_noise = gpura.generate_rir(
		room_sz=room_dim,
		pos_src=[noise_pos],
		pos_rcv=mic_pos,
		Tmax=rir_length,
		fs=fs,
		rt60=rt60
	)

	# 音声信号と雑音信号にそれぞれ残響を付加
	reverberant_speech = np.zeros((nb_mics, rir_speech.shape[1] + clean_speech.shape[0] - 1))
	reverberant_noise = np.zeros((nb_mics, rir_noise.shape[1] + clean_noise.shape[0] - 1))

	for i in range(nb_mics):
		reverberant_speech[i, :] = np.convolve(clean_speech, rir_speech[i], mode='full')
		reverberant_noise[i, :] = np.convolve(clean_noise, rir_noise[i], mode='full')

	# 信号の長さを揃える
	max_len = max(reverberant_speech.shape[1], reverberant_noise.shape[1])
	reverberant_speech = np.pad(reverberant_speech, ((0, 0), (0, max_len - reverberant_speech.shape[1])))
	reverberant_noise = np.pad(reverberant_noise, ((0, 0), (0, max_len - reverberant_noise.shape[1])))

	# SNRに合わせて雑音の振幅を調整
	power_speech = np.sum(reverberant_speech ** 2, axis=1) / max_len
	power_noise = np.sum(reverberant_noise ** 2, axis=1) / max_len

	snr_linear = 10 ** (snr_db / 10.0)

	# 振幅を調整し、音声と雑音を合成
	mixed_audio = np.zeros_like(reverberant_speech)
	for i in range(nb_mics):
		noise_gain = np.sqrt(power_speech[i] / (power_noise[i] * snr_linear))
		mixed_audio[i, :] = reverberant_speech[i, :] + reverberant_noise[i, :] * noise_gain

	# ファイル保存
	sf.write(output_path, mixed_audio.T, fs)
	print(f"混合音声が正常に生成され、'{output_path}'に保存されました。")


# --- 使用例 ---
if __name__ == "__main__":
	# パラメータ設定
	room_parameters = {
		'dim': [6.0, 7.0, 4.0],
		'rt60': 0.5,
		'rir_length': 1.0
	}
	source_positions = {
		'speech': [3, 3, 1.5],
		'noise': [4.0, 3.5, 1.5],
		'mics': [
			[3.0, 3.5, 1.5]  # マイク1
			]
	}

	# 関数を呼び出して音声生成
	generate_reverberant_audio(
		audio_path='/Users/a/Documents/python/PyRoomAcoustics/mymodule/p257_433.wav',
		noise_path='/Users/a/Documents/sound_data/sample_data/noise/hoth.wav',
		output_path='mixed_output.wav',
		room_params=room_parameters,
		source_params=source_positions,
		snr_db=5.0
	)