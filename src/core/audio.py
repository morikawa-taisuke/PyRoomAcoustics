"""
【役割】
音声ファイルの読み書きや変換など、基本的な音声データ処理を行うモジュール
"""
import soundfile as sf
import numpy as np


def load_wav(filepath):
    """soundfileを使用してWAVファイルを読み込み、データとサンプリングレートを返す"""
    data, sr = sf.read(filepath, dtype='float32')
    return data, sr


def save_wav(filepath, data, sr):
    """soundfileを使用してWAVファイルを保存する"""
    sf.write(filepath, data, sr)


def random_crop(noise, target_length):
    """対象の長さに合わせてノイズをランダムにクロップ（不足分は繰り返し）する"""
    if len(noise) <= target_length:
        repeat_times = int(np.ceil(target_length / len(noise)))
        noise = np.tile(noise, repeat_times)
    start = np.random.randint(0, len(noise) - target_length + 1)
    return noise[start:start + target_length]


def mix_snr(speech, noise, snr_db):
    """指定されたSNR(dB)になるよう、ノイズのスケールを調整して重畳する"""
    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power < 1e-10:
        return speech

    target_noise_power = speech_power / (10 ** (snr_db / 10))
    noise_scale = np.sqrt(target_noise_power / noise_power)
    
    return speech + noise * noise_scale


def normalize_audio(audio, max_amplitude=0.9):
    """最大振幅を max_amplitude に収め、音割れを防ぐ（ノーマライズ）"""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio * (max_amplitude / max_val)
    return audio


def get_scale_noise(signal_data, noise_data, snr_db):
    '''目的信号に対するSNR(dB)に合わせて雑音データのスケールを調整する'''
    signal_power = np.mean(signal_data ** 2)
    noise_power = np.mean(noise_data ** 2)

    if noise_power < 1e-10:
        return np.zeros_like(noise_data)

    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise_scale = np.sqrt(target_noise_power / noise_power)

    return noise_data * noise_scale
