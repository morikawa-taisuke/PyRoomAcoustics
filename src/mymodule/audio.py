import soundfile as sf
import numpy as np
import math

from mymodule import rec_config as rec_conf
from . import utility

def get_wave_sample(wave_path):
    """wave_pathの中で最も長い音声長を返す
 
    :param wave_path: 音源のパス
    :return　num_samples: 最長の音声長
    """
    num_samples = 0
    for wave_file in wave_path:
        info = sf.info(wave_file)
        if num_samples < info.frames:
            num_samples = info.frames
    return num_samples


def load_wave_data(wave_path):
    """音源を読み込む
 
    :param wave_path: 音源のパス
    :return wave_data: 読み込んだwaveデータ [音声長,]
    """
    wave_data, _ = sf.read(wave_path, dtype='float32')
    return wave_data


def save_wave(signal, file_name, sample_rate=rec_conf.sampling_rate):
    """wavfileの書き込み
 
    :param signal: wavデータ
    :param file_name: ファイル名
    :param sample_rate: サンプリングレート
    """
    """ 出力先のディレクトリの確認 """
    utility.exists_dir(utility.get_dir_name(file_name))
    """ データを書き込み """
    sf.write(file_name, signal, sample_rate, subtype='PCM_16')


def get_wave_power(wave_data):
    """音源のパワーを計算する
 
    :param wave:
    :return power:
    """
    power = sum(wave_data ** 2)
    return power


def get_snr(signal_pawer, noise_pawer):
    """引数のSNRを計算する
 
    :param signal_pawer: 目的信号
    :param noise_pawer:  雑音信号
    :return snr: SNR
    """
    snr = 10 * math.log10(signal_pawer / noise_pawer)
    return snr


def get_scale_noise(signal_data, noise_data, snr):
    """指定したSNRに雑音の大きさを調整
 
    :param signal_data: 目的信号
    :param noise_data: 雑音
    :param snr: SNR
    :return scale_noise_data: 調整後の雑音 [1,音声長]
    """
    signal_pawer = get_wave_power(signal_data)  # 目的信号のパワーを計算
    noise_pawer = get_wave_power(noise_data)  # 雑音信号のパワーを計算
    ten_pow = pow(10, snr / 10)  # 10^(snr/10)　10の(snr/10)乗を計算
    squared_alpah = signal_pawer / (noise_pawer * ten_pow)  # α^2を計算
    alpha = math.sqrt(squared_alpah)  # αを計算

    scale_noise_data = alpha * noise_data  # 雑音信号の大きさを調整
    after_snr = round(get_snr(signal_pawer, get_wave_power(scale_noise_data)))

    if after_snr != snr:
        print(f"not:{after_snr},{snr}")
    return scale_noise_data
