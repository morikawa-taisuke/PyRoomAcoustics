import json
import math
import os.path

import numpy as np
import pyroomacoustics as pa
from tqdm import tqdm
import soundfile as sf

from mymodule import const, my_func, rec_config as rec_conf, rec_utility as rec_util


def IR_speech(out_dir, reverb_sec, reverb_par, channel=1, distance=0, is_line=False):
    """ シミュレーションを用いた録音 (部屋のパラメータを計算済み)

    Args:
        wave_files: シミュレーションで使用する音声([目的音声,雑音]) [0]:目的音声　[1]:雑音
        out_dir: 出力先のディレクトリ
        snr: 雑音と原音のSNR
        reverb_par: serch_reverb_secによって決定した部屋のパラメータ
        channel: チャンネル数
        is_split: Ture=チャンネルごとにファイルを分ける False=すべてのチャンネルを1つのファイルにまとめる

    Returns:
        None
    """
    # print(f"reverb: {reverb_sec: 03}")
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
    room_reverb = pa.ShoeBox(room_dim, fs=sample_rate, max_order=reverb_par[1], absorption=reverb_par[0])  # 残響のみ
    room_clean = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0, absorption=1.0)  # 教師信号

    """ 部屋にマイクを設置 """
    room_reverb.add_microphone_array(pa.MicrophoneArray(mic_coordinate, fs=room_reverb.fs))
    room_clean.add_microphone_array(pa.MicrophoneArray(mic_coordinate, fs=room_clean.fs))

    """ 各音源の座標 """
    source_coordinate = rec_util.set_souces_coordinate2(doas, distance, mic_center)

    """ 各音源を部屋に追加する """
    room_reverb.add_source(source_coordinate)
    room_clean.add_source(source_coordinate)

    """ インパルス応答を取得する [ チャンネル, マイク, サンプル ] """
    room_reverb.compute_rir()
    room_clean.compute_rir()

    """ インパルス応答の波形データを保存 """
    ir_reverb = room_reverb.rir
    ir_clean = room_clean.rir
    # print("ir_reverb.shape: ", len(ir_reverb[0][0]))
    ir_reverb = ir_reverb[0][0]
    ir_clean = ir_clean[0][0]

    """ 正規化の確認 """
    ir_reverb /= np.max(np.abs(ir_reverb))  # 可視化のため正規化
    ir_clean /= np.max(np.abs(ir_clean))  # 可視化のため正規化

    """ 畳み込んだ波形をファイルに書き込む """
    """ チャンネルをまとめて保存 """
    """ reverbration_only """
    # print(f"ir_reverb.shape:{ir_reverb.shape}")               # 確認用
    reverb_path = f"{out_dir}/reverb_only/speech/{reverb_sec:03}sec.wav"
    my_func.exists_dir(reverb_path)
    sf.write(reverb_path, ir_reverb, sample_rate)
    """ clean """
    # print(f"ir_clean.shape:{ir_clean.shape}")               # 確認用
    clean_path = f"{out_dir}/clean/speech/{reverb_sec:03}sec.wav"
    my_func.exists_dir(clean_path)
    sf.write(clean_path, ir_clean, sample_rate)

def IR_noise(out_dir, reverb_sec, reverb_par, channel=1, distance=0, angle=np.pi, angle_name: str = "None",
           is_line=False):
    """ 雑音を部屋に配置し，各マイクのインパルス応答を出力

    :param out_dir: 出力先 (推奨：絶対パス)
    :param reverb_sec: 残響時間 ( Rt60 )
    :param reverb_par: 部屋のパラメータ [壁の吸収率, 最大反射回数]
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
    room_reverb = pa.ShoeBox(room_dim, fs=sample_rate, max_order=reverb_par[1], absorption=reverb_par[0])  # 残響あり
    room_clean = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0, absorption=1.0)  # 残響なし

    """ 部屋にマイクを設置 """
    room_reverb.add_microphone_array(pa.MicrophoneArray(mic_coordinate, fs=room_reverb.fs))
    room_clean.add_microphone_array(pa.MicrophoneArray(mic_coordinate, fs=room_clean.fs))

    """ 各音源を部屋に追加する """
    source_coordinate = rec_util.set_souces_coordinate2(doas, distance, mic_center) # 音源の座標を計算
    room_reverb.add_source(source_coordinate)
    room_clean.add_source(source_coordinate)

    """ インパルス応答を取得する [ 音源, マイク, サンプル ] """
    room_reverb.compute_rir()
    room_clean.compute_rir()

    """ インパルス応答の波形データを保存 """
    ir_reverb = room_reverb.rir
    ir_clean = room_clean.rir
    # print("ir_reverb.shape: ", ir_reverb.shape)
    # print(ir_clean)
    ir_reverb = ir_reverb[0][0]
    ir_clean = ir_clean[0][0]


    """ 正規化 """
    ir_reverb /= np.max(np.abs(ir_reverb))
    ir_clean /= np.max(np.abs(ir_clean))

    """ 畳み込んだ波形をファイルに書き込む 1つの音声ファイルに全てのチャンネルを保存 """
    """ reverbration_only """
    # print(f"ir_reverb.shape:{ir_reverb.shape}")               # 確認用
    reverb_path = f"{out_dir}/reverb_only/noise/{reverb_sec:03}sec_{angle_name}.wav"
    my_func.exists_dir(reverb_path)
    sf.write(reverb_path, ir_reverb, sample_rate)
    """ clean """
    # print(f"ir_clean.shape:{ir_clean.shape}")               # 確認用
    clean_path = f"{out_dir}/clean/noise/{reverb_sec:03}sec_{angle_name}.wav"
    my_func.exists_dir(clean_path)
    sf.write(clean_path, ir_clean, sample_rate)

def get_shape(data):
    if isinstance(data, list):
        return [len(data)] + get_shape(data[0])
    else:
        return []  # Assuming leaf elements are considered as a single column


if __name__ == "__main__":
    print("generate_IR")
    channel_list = [1]  # チャンネル数
    distance_list = [0]  # マイク間隔 cm
    is_line_list = [True]  # マイク配置が線形(True) or 円形(False)


    for reverb_sec in tqdm(range(50, 50+1)):
        # print(f"reverb_sec: ",reverb_sec)
        # reverb_sec = 50
        reverb_par_json = f"{const.MIX_DATA_DIR}/reverb_condition/{reverb_sec * 10}msec.json"
        # print("json_path:", reverb_par_json)
        with open(reverb_par_json, "r") as json_file:
            json_data = json.load(json_file)
            reverb_par = json_data["reverb_par"]
        for is_line in is_line_list:
            if is_line:
                liner_circular = "liner"
            else:
                liner_circular = "circular"
            for distance in distance_list:
                for channel in channel_list:
                    # out_dir = os.path.join("./", "IR",  f"{channel}ch_{distance}cm_{liner_circular}")
                    out_dir = os.path.join(const.MIX_DATA_DIR, "IR", f"{channel}ch_{distance}cm_{liner_circular}")
                    IR_speech(out_dir, reverb_sec, reverb_par, channel=channel, distance=distance, is_line=is_line)
                    # out_dir = os.path.join(const.SAMPLE_DATA_DIR, "IR",  f"{channel}ch_{distance}cm_{liner_circular}")
                    for dig in range(0, 0+1, 1):
                        angle = math.radians(dig)   # rad ← °
                        angle_name = f"{dig:03}dig"
                        IR_noise(out_dir, reverb_sec, reverb_par, channel=channel, distance=distance, angle=angle, angle_name=angle_name, is_line=is_line)
