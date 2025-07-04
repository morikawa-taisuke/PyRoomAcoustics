import json
import math
import os.path
from distutils.command.clean import clean

import numpy as np
import pyroomacoustics as pa
from tqdm import tqdm
import soundfile as sf

from mymodule import const, rec_config as rec_conf, rec_utility as rec_util
from mymodule import my_func

def IR_speech(out_dir, reverbe_sec, reverbe_par, channel=1, distance=0, is_line=False):
    """ シミュレーションを用いた録音 (部屋のパラメータを計算済み)

    Args:
        wave_files: シミュレーションで使用する音声([目的音声,雑音]) [0]:目的音声　[1]:雑音
        out_dir: 出力先のディレクトリ
        snr: 雑音と原音のSNR
        reverbe_par: serch_reverbe_secによって決定した部屋のパラメータ
        channel: チャンネル数
        is_split: Ture=チャンネルごとにファイルを分ける False=すべてのチャンネルを1つのファイルにまとめる

    Returns:
        None
    """
    # print(f"reverbe: {reverbe_sec: 03}")
    num_sources = 1

    """ 音源のパラメータ """
    sample_rate = rec_conf.sampling_rate  # サンプリング周波数
    """ シミュレーションのパラメータ """
    room_dim = np.r_[10.0, 7.0, 3.0]  # 部屋の大きさ[x,y,z](m)
    mic_center = np.r_[3.0, 3.0, 1.2]  # アレイマイクの中心[x,y,z](m)
    num_channels = channel  # マイクの個数(チャンネル数)
    distance = distance * 0.01  # 各マイクの間隔(m)
    if is_line:
        mic_coordinate = rec_util.set_mic_coordinate(center=mic_center, num_channels=num_channels,
                                                     distance=distance)  # 線形アレイの場合
    else:
        mic_coordinate = rec_util.set_circular_mic_coordinate(center=mic_center, num_channels=num_channels,
                                                              radius=distance)  # 円形アレイの場合

    doas = np.array([
        [np.pi / 2., np.pi / 2],
    ])  # 音源の方向[仰角, 方位角](ラジアン)
    distance = [0.5, 0.7]  # 音源とマイクの距離(m)

    """ 部屋の生成 """
    room_reverbe = pa.ShoeBox(room_dim, fs=sample_rate, max_order=reverbe_par[1], absorption=reverbe_par[0])  # 残響のみ
    room_clean = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0, absorption=1.0)  # 教師信号

    """ 部屋にマイクを設置 """
    room_reverbe.add_microphone_array(pa.MicrophoneArray(mic_coordinate, fs=room_reverbe.fs))
    room_clean.add_microphone_array(pa.MicrophoneArray(mic_coordinate, fs=room_clean.fs))

    """ 各音源の座標 """
    source_coordinate = rec_util.set_souces_coordinate2(doas, distance, mic_center)

    """ 各音源を部屋に追加する """
    room_reverbe.add_source(source_coordinate)
    room_clean.add_source(source_coordinate)

    """ インパルス応答を取得する [ チャンネル, マイク, サンプル ] """
    room_reverbe.compute_rir()
    room_clean.compute_rir()

    """ インパルス応答の波形データを保存 """
    ir_reverbe = room_reverbe.rir
    ir_clean = room_clean.rir
    # print("ir_reverbe.shape: ", len(ir_reverbe[0][0]))
    ir_reverbe = ir_reverbe[0][0]
    ir_clean = ir_clean[0][0]

    """ 正規化の確認 """
    ir_reverbe /= np.max(np.abs(ir_reverbe))  # 可視化のため正規化
    ir_clean /= np.max(np.abs(ir_clean))  # 可視化のため正規化

    """ 畳み込んだ波形をファイルに書き込む """
    """ チャンネルをまとめて保存 """
    """ reverberation_only """
    # print(f"ir_reverbe.shape:{ir_reverbe.shape}")               # 確認用
    reverbe_path = f"{out_dir}/reverbe_only/speech/{reverbe_sec:03}sec.wav"
    my_func.exists_dir(reverbe_path)
    sf.write(reverbe_path, ir_reverbe, sample_rate)
    """ clean """
    # print(f"ir_clean.shape:{ir_clean.shape}")               # 確認用
    clean_path = f"{out_dir}/clean/speech/{reverbe_sec:03}sec.wav"
    my_func.exists_dir(clean_path)
    sf.write(clean_path, ir_clean, sample_rate)

def IR_noise(out_dir, reverbe_sec, reverbe_par, channel=1, distance=0, angle=np.pi, angle_name: str = "None",
           is_line=False):
    """ 雑音を部屋に配置し，各マイクのインパルス応答を出力

    :param out_dir: 出力先 (推奨：絶対パス)
    :param reverbe_sec: 残響時間 ( Rt60 )
    :param reverbe_par: 部屋のパラメータ [壁の吸収率, 最大反射回数]
    :param channel: マイク数 (チャンネル数)
    :param distance: マイク間隔
    :param angle: 雑音の角度 (水平角)
    :param angle_name: 角度の名前 (出力ファイル名に仕様)
    :param is_line: マイク配置が線形(True) or 円形(False)
    :return:
    """
    # print("out_dir:", out_dir)

    """ 音源のパラメータ """
    sample_rate = rec_conf.sampling_rate  # サンプリング周波数
    """ シミュレーションのパラメータ """
    room_dim = np.r_[10.0, 7.0, 3.0]  # 部屋の大きさ[x,y,z](m)
    mic_center = np.r_[3.0, 3.0, 1.2]  # アレイマイクの中心[x,y,z](m)
    num_channels = channel  # マイクの個数(チャンネル数)
    distance = distance * 0.01  # 各マイクの間隔(m)
    if is_line:
        mic_coordinate = rec_util.set_mic_coordinate(center=mic_center, num_channels=num_channels,
                                                     distance=distance)  # 線形アレイの場合
    else:
        mic_coordinate = rec_util.set_circular_mic_coordinate(center=mic_center, num_channels=num_channels,
                                                              radius=distance)  # 円形アレイの場合

    doas = np.array([
        [np.pi / 2., angle]
    ])  # 音源の方向[仰角, 方位角](ラジアン)
    distance = [0.7]  # 音源とマイクの距離(m)

    """ 部屋の生成 """
    room_reverbe = pa.ShoeBox(room_dim, fs=sample_rate, max_order=reverbe_par[1], absorption=reverbe_par[0])  # 残響あり
    room_clean = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0, absorption=1.0)  # 残響なし

    """ 部屋にマイクを設置 """
    room_reverbe.add_microphone_array(pa.MicrophoneArray(mic_coordinate, fs=room_reverbe.fs))
    room_clean.add_microphone_array(pa.MicrophoneArray(mic_coordinate, fs=room_clean.fs))

    """ 各音源を部屋に追加する """
    source_coordinate = rec_util.set_souces_coordinate2(doas, distance, mic_center) # 音源の座標を計算
    room_reverbe.add_source(source_coordinate)
    room_clean.add_source(source_coordinate)

    """ インパルス応答を取得する [ 音源, マイク, サンプル ] """
    room_reverbe.compute_rir()
    room_clean.compute_rir()

    """ インパルス応答の波形データを保存 """
    ir_reverbe = room_reverbe.rir
    ir_clean = room_clean.rir
    # print("ir_reverbe.shape: ", ir_reverbe.shape)
    # print(ir_clean)
    ir_reverbe = ir_reverbe[0][0]
    ir_clean = ir_clean[0][0]


    """ 正規化 """
    ir_reverbe /= np.max(np.abs(ir_reverbe))
    ir_clean /= np.max(np.abs(ir_clean))

    """ 畳み込んだ波形をファイルに書き込む 1つの音声ファイルに全てのチャンネルを保存 """
    """ reverberation_only """
    # print(f"ir_reverbe.shape:{ir_reverbe.shape}")               # 確認用
    reverbe_path = f"{out_dir}/reverbe_only/noise/{reverbe_sec:03}sec_{angle_name}.wav"
    my_func.exists_dir(reverbe_path)
    sf.write(reverbe_path, ir_reverbe, sample_rate)
    """ clean """
    # print(f"ir_clean.shape:{ir_clean.shape}")               # 確認用
    clean_path = f"{out_dir}/clean/noise/{reverbe_sec:03}sec_{angle_name}.wav"
    my_func.exists_dir(clean_path)
    sf.write(clean_path, ir_clean, sample_rate)

def get_shape(data):
    if isinstance(data, list):
        return [len(data)] + get_shape(data[0])
    else:
        return []  # Assuming leaf elements are considered as a single column

def serch_reverbe_sec(reverbe_sec, channel=1, angle=np.pi):
    reverbe = reverbe_sec
    cnt = 0
    room_dim = np.r_[10.0, 7.0, 3.0]
    """ 音源の読み込み """
    target_data = rec_util.load_wave_data(f"./mymodule/JA01F049.wav")
    noise_data = target_data
    wave_data = []  # 1つの配列に格納
    wave_data.append(target_data)
    wave_data.append(noise_data)
    mic_center = np.r_[3.0, 3.0, 1.2]  # アレイマイクの中心[x,y,z](m)
    num_channels = channel  # マイクの個数(チャンネル数)
    distance = 0.1  # 各マイクの間隔(m)
    mic_codinate = rec_util.set_mic_coordinate(center=mic_center,
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
        e_absorption, max_order = pa.inverse_sabine(reverbe, room_dim)  # Sabineの残響式から壁の吸収率と反射上限回数を決定
        room = pa.ShoeBox(room_dim, fs=rec_conf.sampling_rate, max_order=max_order, absorption=e_absorption)    # 部屋の作成

        """ 部屋にマイクを設置 """
        room.add_microphone_array(pa.MicrophoneArray(mic_codinate, fs=room.fs))
        """ 各音源の座標 """
        source_codinate = rec_util.set_souces_coordinate2(doas, distance, mic_center)
        """ 各音源を部屋に追加する """
        for idx in range(2):
            wave_data[idx] /= np.std(wave_data[idx])
            room.add_source(source_codinate[:, idx], signal=wave_data[idx])

        room.simulate() # シミュレーション
        rt60 = room.measure_rt60()  # 残響時間の取得
        round_rt60 = round(np.mean(rt60), 3)    # 有効数字3桁で丸める
        if round_rt60 >= reverbe_sec:   #
            break
        cnt += 1
        reverbe += 0.01
    # print(f"[{cnt}]rt60:{np.mean(rt60)}")
    print(f"max_order:{max_order}\ne_absorption:{e_absorption}")
    print(f"rt60={np.mean(rt60)}")
    return e_absorption, max_order


if __name__ == "__main__":
    print("generate_IR")
    channel_list = [1]  # チャンネル数
    distance_list = [0]  # マイク間隔 cm
    is_line_list = [True]  # マイク配置が線形(True) or 円形(False)


    for reverbe_sec in tqdm(range(50, 50+1)):
        # print(f"reverbe_sec: ",reverbe_sec)
        # reverbe_sec = 50
        reverbe_par_json = f"{const.MIX_DATA_DIR}/reverbe_condition/{reverbe_sec*10}msec.json"
        # print("json_path:", reverbe_par_json)
        with open(reverbe_par_json, "r") as json_file:
            json_data = json.load(json_file)
            reverbe_par = json_data["reverbe_par"]
        for is_line in is_line_list:
            if is_line:
                liner_circular = "liner"
            else:
                liner_circular = "circular"
            for distance in distance_list:
                for channel in channel_list:
                    # out_dir = os.path.join("./", "IR",  f"{channel}ch_{distance}cm_{liner_circular}")
                    out_dir = os.path.join(const.MIX_DATA_DIR, "IR",  f"{channel}ch_{distance}cm_{liner_circular}")
                    IR_speech(out_dir, reverbe_sec, reverbe_par, channel=channel, distance=distance, is_line=is_line)
                    # out_dir = os.path.join(const.SAMPLE_DATA_DIR, "IR",  f"{channel}ch_{distance}cm_{liner_circular}")
                    for dig in range(0, 0+1, 1):
                        angle = math.radians(dig)   # rad ← °
                        angle_name = f"{dig:03}dig"
                        IR_noise(out_dir, reverbe_sec, reverbe_par, channel=channel, distance=distance, angle=angle, angle_name=angle_name, is_line=is_line)
