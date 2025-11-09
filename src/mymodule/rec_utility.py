import numpy as np
import itertools
import random
import pyroomacoustics as pa

from mymodule import reverb_feater as rev_feat, rec_config as rec_conf
from . import audio

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
    # マイクロホンアレイの座標
    coordinate = mic_alignments.T + center[:, None]

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

    if len(center) == 2:
        x_points = center[0] + radius * np.cos(angle_list)
        y_points = center[1] + radius * np.sin(angle_list)
        coordinate = np.array([x_points.tolist(), y_points.tolist()])
    else:
        x_points = center[0] + radius * np.cos(angle_list)
        y_points = center[1] + radius * np.sin(angle_list)
        z_points = np.full(num_channels, center[2])
        coordinate = np.array([x_points.tolist(), y_points.tolist(), z_points.tolist()])

    return coordinate



def set_souces_coordinate(doas, distance, mic_center):
    """音源の座標を計算する
 
    :param doas: 音源の到来方向
    :param distance: 音源からアレイマイクの中心点までの距離
    :param mic_center: アレイマイクの中心点
    :return :
    """
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
    souces_coordinate = np.zeros((3, doas.shape[0]), dtype=doas.dtype)
    souces_coordinate[0, :] = np.cos(doas[:, 1]) * np.sin(doas[:, 0])  # x
    souces_coordinate[1, :] = np.sin(doas[:, 1]) * np.sin(doas[:, 0])  # y
    souces_coordinate[2, :] = np.cos(doas[:, 0])  # z
    for idx in range(doas.shape[0]):
        souces_coordinate[:, idx] *= distance[idx]
    souces_coordinate += mic_center[:, None]
    return souces_coordinate


def nantoka(room_dim):
    volume = np.prod(room_dim)
    edgs_combination = itertools.combinations(room_dim, 2)
    area = [l1 * l2 for l1, l2 in edgs_combination]
    sphere = 2 * np.sum(area)
    sab_coef = 24


def search_reverb_sec(reverb_sec, channel=1, angle=np.pi):
    reverb = reverb_sec
    cnt = 0
    room_dim = np.r_[10.0, 7.0, 3.0]
    """ 音源の読み込み """
    target_data = audio.load_wave_data(f"./sample_data/JA01F049.wav")
    noise_data = target_data
    wave_data = []  # 1つの配列に格納
    wave_data.append(target_data)
    wave_data.append(noise_data)
    mic_center = np.r_[3.0, 3.0, 1.2]  # アレイマイクの中心[x,y,z](m)
    num_channels = channel  # マイクの個数(チャンネル数)
    distance = 0.1  # 各マイクの間隔(m)
    mic_codinate = set_mic_coordinate(center=mic_center,
                                               num_channels=num_channels,
                                               distance=distance)  # 各マイクの座標
    doas = np.array([
        [np.pi/2., np.pi/2],
        [np.pi/2., angle]
    ])  # 音源の方向[仰角, 方位角](ラジアン)
    distance = [0.5, 0.7]  # 音源とマイクの距離(m)

    max_order = 0   # 初期化
    e_absorption = 0    # 初期化
    rt60 = 0    # 初期化
    while cnt < 100:   # 試行回数が100以上の時にループを抜ける
        e_absorption, max_order = pa.inverse_sabine(reverb, room_dim)  # Sabineの残響式から壁の吸収率と反射上限回数を決定
        room = pa.ShoeBox(room_dim, fs=rec_conf.sampling_rate, max_order=max_order, absorption=e_absorption)    # 部屋の作成

        """ 部屋にマイクを設置 """
        room.add_microphone_array(pa.MicrophoneArray(mic_codinate, fs=room.fs))
        """ 各音源の座標 """
        source_codinate = set_souces_coordinate2(doas, distance, mic_center)
        """ 各音源を部屋に追加する """
        for idx in range(2):
            wave_data[idx] /= np.std(wave_data[idx])
            room.add_source(source_codinate[:, idx], signal=wave_data[idx])

        room.simulate() # シミュレーション
        rt60 = room.measure_rt60()  # 残響時間の取得
        round_rt60 = round(np.mean(rt60), 3)    # 有効数字3桁で丸める
        if round_rt60 >= reverb_sec:   #
            break
        cnt += 1
        reverb += 0.01
    print(f"max_order:{max_order}\ne_absorption:{e_absorption}")
    print(f"rt60={np.mean(rt60)}")
    return e_absorption, max_order


def create_random_room_shoebox(
        room_dim_range=((3, 8), (3, 8), (2.5, 4)),
        rt60_range=(0.1, 1.0),
        fs=16000
):
    """
    ランダムなパラメータでShoeBoxルームを作成する
    (new_signal_noise.pyから切り出し)

    Args:
        room_dim_range (tuple): (x_range, y_range, z_range)
        rt60_range (tuple): (min, max)
        fs (int): サンプリング周波数

    Returns:
        tuple: (room, room_dim, rt60_target, e_absorption, max_order)
    """
    # ランダムな部屋のパラメータを生成
    room_dim = np.array([
        random.uniform(room_dim_range[0][0], room_dim_range[0][1]),
        random.uniform(room_dim_range[1][0], room_dim_range[1][1]),
        random.uniform(room_dim_range[2][0], room_dim_range[2][1])
    ])

    # Sabineの残響式から吸収率と反射上限回数を決定
    rt60_target = random.uniform(rt60_range[0], rt60_range[1])
    e_absorption, max_order = pa.inverse_sabine(rt60_target, room_dim)

    # 部屋の作成
    room = pa.ShoeBox(
        room_dim,
        fs=fs,
        max_order=max_order,
        materials=pa.Material(e_absorption)
    )

    return room, room_dim, rt60_target, e_absorption, max_order


def compute_rir_and_features(room, mic_coordinate, source_pos_signal, source_pos_noise):
    """
    部屋にマイクと音源を設置し、RIRと音響特徴量を計算する
    (new_signal_noise.pyから切り出し)

    Args:
        room (pa.ShoeBox): pyroomacoustics の room オブジェクト
        mic_coordinate (np.ndarray): マイク座標
        source_pos_signal (np.ndarray): 目的音源の座標
        source_pos_noise (np.ndarray): 雑音音源の座標

    Returns:
        tuple: (rir_signal, rir_noise, rt60, c50, d50)
    """
    # マイクの設置
    room.add_microphone_array(pa.MicrophoneArray(mic_coordinate, fs=room.fs))

    # 音源の追加
    room.add_source(source_pos_signal)
    room.add_source(source_pos_noise)

    # RIRを計算
    room.compute_rir()

    # RIRは (マイク, 音源) のリストで返る
    # マイク0, 音源0 (目的信号)
    rir_signal = room.rir[0][0]
    # マイク0, 音源1 (雑音)
    rir_noise = room.rir[0][1]

    # 物理的特徴量（RT60, C50, D50）を計算
    # 目的信号のRIR (マイク0, 音源0) を使用
    rt60 = room.measure_rt60()[0][0]
    c50 = rev_feat.calculate_c50(rir_signal, fs=room.fs)
    d50 = rev_feat.calculate_d50(rir_signal, fs=room.fs)

    return rir_signal, rir_noise, rt60, c50, d50


def convolve_and_mix(clean_signal, noise_segment, rir_signal, rir_noise, snr):
    """
    各信号とRIRを畳み込み、指定したSNRで混合する
    (new_signal_noise.pyから切り出し)

    Args:
        clean_signal (np.ndarray): モノラルクリーン音声 (N,)
        noise_segment (np.ndarray): モノラル雑音 (N,)
        rir_signal (np.ndarray): 目的信号用RIR
        rir_noise (np.ndarray): 雑音用RIR
        snr (float): 混合SNR [dB]

    Returns:
        np.ndarray: 混合後の音声信号 (M,)
    """
    # RIRで畳み込み、残響付き信号を生成
    reverb_signal = np.convolve(clean_signal, rir_signal, mode='full')[:len(clean_signal)]
    reverb_noise = np.convolve(noise_segment, rir_noise, mode='full')[:len(noise_segment)]

    # SNRを調整して結合
    scaled_noise = audio.get_scale_noise(reverb_signal, reverb_noise, snr)
    mixed_signal = reverb_signal + scaled_noise

    return mixed_signal


# ----------------------------------------------------


if __name__ == "__main__":
    print("\nrec_utility")

    target_dir = ["./wave/sample_data/speech/JA/training_JA01/JA01F049.wav"]
    noise_path = "./wave/sample_data/noise/hoth.wav"
    for target_path in target_dir:
        target_data = audio.load_wave_data(target_path)
    noise_data = audio.load_wave_data(noise_path)
    start = random.randint(0, len(noise_data) - len(target_data))
    noise_data = noise_data[start: start + len(target_data)]
    scale_nosie = audio.get_scale_noise(target_data, noise_data, 10)

    room_dim = np.r_[10, 10, 10]
    doas = np.array([
        [np.pi / 2., 0],
        [np.pi / 2., np.pi]
    ])
    distance = [1., 2.]
    mic_center = room_dim / 2.
    source_codinate = set_souces_coordinate2(doas, distance, mic_center=mic_center)
    print(f"source_codinate:{source_codinate}")

    print("rec_utility\n")
