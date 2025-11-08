import wave as wave
import numpy as np
import os
import itertools
import math
import random
import pyroomacoustics as pa

from mymodule import my_func
from mymodule import rec_config as rec_conf


def set_mic_coordinate(center, num_channels, distance):
    """ アレイマイクの各マイクの座標を決める (線形アレイ)
    :param center: マイクの中心点
    :param num_channels: チャンネル数
    :param distance: マイク間の距離
    :return coordinate: マイクの座標
    """
    # マイクロホンアレイのマイク配置
    if center.ndim == 2:
        mic_alignments = np.array([[0.0 + distance * (i + (1 - num_channels) / 2), 0.0] for i in range(num_channels)])
    else:
        mic_alignments = np.array([[0.0 + distance * (i + (1 - num_channels) / 2), 0.0, 0.0] for i in range(num_channels)])
    # print(mic_alignments)
    # マイクロホンアレイの座標
    coordinate = mic_alignments.T + center[:, None]
    # print(coordinate)

    return coordinate

def set_circular_mic_coordinate(center, num_channels:int, radius, rotate:bool=False):
    """ アレイマイクの各マイクの座標を決める (円形アレイ)
    :param center: マイクの中心点
    :param num_channels: チャンネル数
    :param radius: アレイマイクの半径
    :param rotate: 回転の有無 回転しない場合,話者に対して十字に配置, した場合,Xのように配置する_
    :return coordinate: マイクの座標
    """
    if not rotate:
        angle_list = np.linspace(0, 2*np.pi, num_channels, endpoint=False)	# 回転なし
    else:
        angle_list = np.linspace(0+np.pi/4, 2*np.pi+np.pi/4, num_channels, endpoint=False)	# 45°回転
    # angle_list = np.linspace(0, 2*np.pi, num_channels, endpoint=False)	# 回転なし
    # angle_list = np.linspace(0 + np.pi / 4, 2 * np.pi + np.pi / 4, num_channels, endpoint=False)  # 45°回転
    # print(angle_list)
    if len(center) == 2:
        x_points = center[0] + radius * np.cos(angle_list)
        y_points = center[1] + radius * np.sin(angle_list)
        coordinate = np.array([x_points.tolist(), y_points.tolist()])
    else:
        x_points = center[0] + radius * np.cos(angle_list)
        y_points = center[1] + radius * np.sin(angle_list)
        z_points = np.full(num_channels, center[2])
        coordinate = np.array([x_points.tolist(), y_points.tolist(), z_points.tolist()])

    # print(coordinate)

    return coordinate


def set_souces_coordinate(doas, distance, mic_center):
    """音源の座標を計算する
 
    :param doas: 音源の到来方向
    :param distance: 音源からアレイマイクの中心点までの距離
    :param mic_center: アレイマイクの中心点
    :return :
    """
    # print("set_souces")
    souces_coordinate = np.zeros((3, doas.shape[0]), dtype=doas.dtype)
    souces_coordinate[0, :] = np.cos(doas[:, 1]) * np.sin(doas[:, 0])
    souces_coordinate[1, :] = np.sin(doas[:, 1]) * np.sin(doas[:, 0])
    souces_coordinate[2, :] = np.cos(doas[:, 0])
    souces_coordinate *= distance
    souces_coordinate += mic_center[:, None]
    return souces_coordinate


def set_souces_coordinate2(doas, distance, mic_center):
    """音源の座標を計算する
 
    :param doas: 音源の到来方向 [2,音源数]
    :param distance: 音源からアレイマイクの中心点までの距離
    :param mic_center: アレイマイクの中心点
    :return :
    """
    # print("set_souces")
    souces_coordinate = np.zeros((3, doas.shape[0]), dtype=doas.dtype)
    souces_coordinate[0, :] = np.cos(doas[:, 1]) * np.sin(doas[:, 0])  # x
    souces_coordinate[1, :] = np.sin(doas[:, 1]) * np.sin(doas[:, 0])  # y
    souces_coordinate[2, :] = np.cos(doas[:, 0])  # z
    # print(f"souces_coordinate.shape:{souces_coordinate.shape}")
    # print(f"np.array(distance).shape:{np.array(distance).shape}")
    for idx in range(doas.shape[0]):
        souces_coordinate[:, idx] *= distance[idx]
    souces_coordinate += mic_center[:, None]
    return souces_coordinate


def get_wave_sample(wave_path):
    """wave_pathの中で最も長い音声長を返す
 
    :param wave_path: 音源のパス
    :return　num_samples: 最長の音声長
    """
    num_samples = 0
    for wave_file in wave_path:
        wav = wave.open(wave_file)
        if num_samples < wav.getnframes():
            num_samples = wav.getnframes()
        wav.close()
        return num_samples


def load_wave_data(wave_path):
    """音源を読み込む
 
    :param wave_path: 音源のパス
    :return wave_data: 読み込んだwaveデータ [音源数,音声長]
    """
    # wav = wave.open(wave_path,"r")
    # wave_data = wav.readframes(wav.getnframes())
    # wave_data = np.frombuffer(wave_data, dtype=np.int16)
    # wave_data = wave_data / np.iinfo(np.int16).max
    # wav.close()
    with wave.open(wave_path, "r") as wav:
        wave_data = wav.readframes(wav.getnframes())
        wave_data = np.frombuffer(wave_data, dtype=np.int16)
        wave_data = wave_data / np.iinfo(np.int16).max
    return wave_data


def save_wave(signal, file_name, sample_rate=rec_conf.sampling_rate):
    """wavfileの書き込み
 
    :param signal: wavデータ
    :param file_name: ファイル名
    :param sample_rate: サンプリングレート
    """
    """ 出力先のディレクトリの確認 """
    my_func.exists_dir(my_func.get_dirname(file_name))
    """ 2バイトのデータに変換 """
    signal = signal.astype(np.int16)
    # """ 出力ファイルを書き込み専用で開く """
    # wave_out = wave.open(file_name, "w")
    # """ 出力の設定 """
    # wave_out.setnchannels(1)  # モノラル:1、ステレオ:2
    # wave_out.setsampwidth(2)  # サンプルサイズ2byte
    # wave_out.setframerate(sample_rate)  # サンプリング周波数
    # """ データを書き込み """
    # wave_out.writeframes(signal)
    # """ ファイルを閉じる """
    # wave_out.close()
    with wave.open(file_name,"w") as wave_out:
        wave_out.setnchannels(1)  # モノラル:1、ステレオ:2
        wave_out.setsampwidth(2)  # サンプルサイズ2byte
        wave_out.setframerate(sample_rate)  # サンプリング周波数
        """ データを書き込み """
        wave_out.writeframes(signal)


def get_wave_power(wave_data):
    """音源のパワーを計算する
 
    :param wave:
    :return power:
    """
    power = sum(wave_data ** 2)
    # print(f"type(power):{type(power)}") # 確認用
    # print(f"power.shape:{power.shape}") # 確認用
    # print(power)                        # 確認用
    # data = np.array([1,2,3])
    # squea_data = data**2
    # print(data)
    # print(squea_data)
    # sum_data = sum(data**2)
    # print(sum_data)
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
    # print(f"snr:{snr}")
    # print(f"befor_snr:{get_snr(signal_pawer,noise_pawer)}")
    # print(f"after_snr:{after_snr}")

    if after_snr != snr:
        print(f"not:{after_snr},{snr}")
    return scale_noise_data


def nantoka(room_dim):
    # print("inverse_sabine")
    volume = np.prod(room_dim)  # 室内体積(volume)を求める
    # print(f"volume:{volume}")

    # 部屋の総面積を求める
    edgs_combination = itertools.combinations(room_dim, 2)  # 重複アリの組み合わせ [len(配列),2]
    # edgs_combination = np.array(list(edgs_combination))
    # print(f"edgs_combination:{edgs_combination}")
    area = [l1 * l2 for l1, l2 in edgs_combination]  # 各面積(area)を求める
    # print(f"area:{area}")
    sphere = 2 * np.sum(area)  # 室内総面積(sphere)を求める
    sab_coef = 24


def get_file_name(file_path: str) -> str:
    """ ファイル名を取得する
 
    Parameters
    ----------
    file_path:ファイル名を取得するパス
 
    Returns
    -------
    file_name: ファイル名
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    return file_name


def get_dir_name(dir_path):
    dir_name = os.path.dirname(dir_path)
    return dir_name


def exists_dir(dir_path):
    _, ext = os.path.splitext(dir_path)
    # print("util : exists_dir", _, ext)
    if len(ext) == 0 and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    elif not len(ext) == 0 and not os.path.exists(get_dir_name(dir_path)):
        os.makedirs(get_dir_name(dir_path))
    return ext


def search_reverb_sec(reverbe_sec, channel=1, angle=np.pi):
    reverbe = reverbe_sec
    cnt = 0
    room_dim = np.r_[5.0, 5.0, 5.0]
    """ 音源の読み込み """
    target_data = load_wave_data(f"./mymodule/JA01F049.wav")
    noise_data = target_data
    wave_data = []  # 1つの配列に格納
    wave_data.append(target_data)
    wave_data.append(noise_data)
    mic_center = room_dim / 2  # アレイマイクの中心[x,y,z](m)
    num_channels = channel  # マイクの個数(チャンネル数)
    distance = 0.1  # 各マイクの間隔(m)
    mic_codinate = set_mic_coordinate(center=mic_center,
                                      num_channels=num_channels,
                                      distance=distance)  # 各マイクの座標
    doas = np.array([
        [np.pi / 2., np.pi / 2],
        [np.pi / 2., angle]
    ])  # 音源の方向[仰角, 方位角](ラジアン)
    distance = [0.5, 0.7]  # 音源とマイクの距離(m)

    max_order = 0  # 初期化
    e_absorption = 0  # 初期化
    rt60 = 0  # 初期化
    while cnt < 100:  # 試行回数が100以上の時にループを抜ける
        e_absorption, max_order = pa.inverse_sabine(reverbe, room_dim)  # Sabineの残響式から壁の吸収率と反射上限回数を決定
        room = pa.ShoeBox(room_dim, fs=rec_conf.sampling_rate, max_order=max_order, absorption=e_absorption)  # 部屋の作成

        """ 部屋にマイクを設置 """
        room.add_microphone_array(pa.MicrophoneArray(mic_codinate, fs=room.fs))
        """ 各音源の座標 """
        source_codinate = set_souces_coordinate2(doas, distance, mic_center)
        """ 各音源を部屋に追加する """
        for idx in range(2):
            wave_data[idx] /= np.std(wave_data[idx])
            room.add_source(source_codinate[:, idx], signal=wave_data[idx])

        room.simulate()  # シミュレーション
        rt60 = room.measure_rt60()  # 残響時間の取得
        round_rt60 = round(np.mean(rt60), 3)  # 有効数字3桁で丸める
        if round_rt60 >= reverbe_sec:  #
            break
        cnt += 1
        reverbe += 0.01
    # print(f"[{cnt}]rt60:{np.mean(rt60)}")
    print(f"max_order:{max_order}\ne_absorption:{e_absorption}")
    print(f"rt60={np.mean(rt60)}")
    return e_absorption, max_order


if __name__ == "__main__":
    print("\nrec_utility")
    """print("main")
    room_dim=[10,20,30]
    print(f"room_dim:{room_dim}")
 
    nantoka(room_dim)"""
    # """
    target_dir = ["./wave/sample_data/speech/JA/training_JA01/JA01F049.wav"]
    noise_path = "./wave/sample_data/noise/hoth.wav"
    for target_path in target_dir:
        target_data = load_wave_data(target_path)
    noise_data = load_wave_data(noise_path)
    start = random.randint(0, len(noise_data) - len(target_data))  # スタート位置をランダムに決定
    noise_data = noise_data[start: start + len(target_data)]  # noise_dataを切り出す
    # print(f"len(target_data):{len(target_data)}")               # 確認用
    # print(f"len(noise_data):{len(noise_data)}")                 # 確認用
    scale_nosie = get_scale_noise(target_data, noise_data, 10)  # 雑音の調整
    # print(f"len(scale_noise):{len(scale_nosie)}")

    room_dim = np.r_[10, 10, 10]
    doas = np.array([
        [np.pi / 2., 0],
        [np.pi / 2., np.pi]
    ])  # 音源の方向[仰角, 方位角](ラジアン)
    distance = [1., 2.]
    mic_center = room_dim / 2.
    source_codinate = set_souces_coordinate2(doas, distance, mic_center=mic_center)
    print(f"source_codinate:{source_codinate}")
    # """

    print("rec_utility\n")
