import json
import os.path

import numpy as np
import pyroomacoustics as pa
from tqdm import tqdm

from mymodule import const, rec_config as rec_conf, rec_utility as rec_util


def rec_IR(out_dir, reverbe_sec, reverbe_par, channel=1, distance=0, angle=np.pi, angle_name: str = "None",
           is_line=False):
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
    # print("out_dir:", out_dir)
    num_sources = 2

    """ 音源のパラメータ """
    sample_rate = rec_conf.sampling_rate  # サンプリング周波数
    """ シミュレーションのパラメータ """
    room_dim = np.r_[3.0, 3.0, 3.0]  # 部屋の大きさ[x,y,z](m)
    mic_center = room_dim / 2  # アレイマイクの中心[x,y,z](m)
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
        [np.pi / 2., angle]
    ])  # 音源の方向[仰角, 方位角](ラジアン)
    distance = [0.5, 0.7]  # 音源とマイクの距離(m)

    """ 部屋の生成 """
    room_mix = pa.ShoeBox(room_dim, fs=sample_rate, max_order=reverbe_par[1], absorption=reverbe_par[0])  # 雑音 + 残響
    room_reverbe = pa.ShoeBox(room_dim, fs=sample_rate, max_order=reverbe_par[1], absorption=reverbe_par[0])  # 残響のみ
    room_noise = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0, absorption=1.0)  # 雑音のみ
    room_clean = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0, absorption=1.0)  # 教師信号

    """ 部屋にマイクを設置 """
    room_mix.add_microphone_array(pa.MicrophoneArray(mic_coordinate, fs=room_mix.fs))
    room_reverbe.add_microphone_array(pa.MicrophoneArray(mic_coordinate, fs=room_reverbe.fs))
    room_noise.add_microphone_array(pa.MicrophoneArray(mic_coordinate, fs=room_noise.fs))
    room_clean.add_microphone_array(pa.MicrophoneArray(mic_coordinate, fs=room_clean.fs))

    """ 各音源の座標 """
    source_coordinate = rec_util.set_souces_coordinate2(doas, distance, mic_center)

    """ 各音源を部屋に追加する """
    for idx in range(num_sources):
        room_mix.add_source(source_coordinate[:, idx])
        room_noise.add_source(source_coordinate[:, idx])
        if idx == 0:  # 目的信号のみ追加する
            room_reverbe.add_source(source_coordinate[:, idx])
            room_clean.add_source(source_coordinate[:, idx])

    """ インパルス応答を取得する [ チャンネル, マイク, サンプル ] """
    room_mix.compute_rir()
    room_reverbe.compute_rir()
    room_noise.compute_rir()
    room_clean.compute_rir()

    """ インパルス応答の波形データを保存 """
    ir_mix = room_mix.rir
    ir_reverbe = room_reverbe.rir
    ir_noise = room_noise.rir
    ir_clean = room_clean.rir

    """ 最長のモノに長さをそろえる """
    # 各次元の最大長を取得
    length_mix = max(len(item) for sublist in ir_mix for item in sublist)
    length_reverbe = max(len(item) for sublist in ir_reverbe for item in sublist)
    length_noise = max(len(item) for sublist in ir_noise for item in sublist)
    length_clean = max(len(item) for sublist in ir_clean for item in sublist)

    # 0でパディングした新しい3次元リストを作成
    ir_mix = [[np.pad(item, (0, length_mix - len(item)), constant_values=0).tolist() for item in sublist] for sublist in ir_mix]
    ir_reverbe = [[np.pad(item, (0, length_reverbe - len(item)), constant_values=0).tolist() for item in sublist] for sublist in ir_reverbe]
    ir_noise = [[np.pad(item, (0, length_noise - len(item)), constant_values=0).tolist() for item in sublist] for sublist in ir_noise]
    ir_clean = [[np.pad(item, (0, length_clean - len(item)), constant_values=0).tolist() for item in sublist] for sublist in ir_clean]

    """ 正規化の確認 """
    ir_mix /= np.max(np.abs(ir_mix))  # 可視化のため正規化
    ir_reverbe /= np.max(np.abs(ir_reverbe))  # 可視化のため正規化
    ir_noise /= np.max(np.abs(ir_noise))  # 可視化のため正規化
    ir_clean /= np.max(np.abs(ir_clean))  # 可視化のため正規化

    """ 畳み込んだ波形をファイルに書き込む """
    """ チャンネルをまとめて保存 """
    """ noise_reverberation """
    # print(f"result_mix.shape:{result_mix.shape}")
    mix_path = f"{out_dir}/noise_reverbe/target/{int(reverbe_sec * 10):02}sec.wav"
    rec_util.save_wave(ir_mix[0], mix_path)  # 保存
    mix_path = f"{out_dir}/noise_reverbe/noise/{int(reverbe_sec * 10):02}sec_{angle_name}.wav"
    rec_util.save_wave(ir_mix[1], mix_path)  # 保存
    """ reverberation_only """
    # print(f"ir_reverbe.shape:{ir_reverbe.shape}")               # 確認用
    reverbe_path = f"{out_dir}/reverbe_only/target/{int(reverbe_sec * 10):02}sec.wav"
    rec_util.save_wave(ir_reverbe[0], reverbe_path)  # 保存
    reverbe_path = f"{out_dir}/reverbe_only/noise/{int(reverbe_sec * 10):02}sec_{angle_name}.wav"
    rec_util.save_wave(ir_reverbe[1], reverbe_path)  # 保存
    """ nosie_only """
    # print(f"ir_nosie.shape:{ir_noise.shape}")               # 確認用
    reverbe_path = f"{out_dir}/reverbe_only/target/{int(reverbe_sec * 10):02}sec.wav"
    rec_util.save_wave(ir_reverbe[0], reverbe_path)  # 保存
    reverbe_path = f"{out_dir}/reverbe_only/noise/{int(reverbe_sec * 10):02}sec_{angle_name}.wav"
    rec_util.save_wave(ir_reverbe[1], reverbe_path)  # 保存
    """ clean """
    # print(f"ir_clean.shape:{ir_clean.shape}")               # 確認用
    reverbe_path = f"{out_dir}/reverbe_only/target/{int(reverbe_sec * 10):02}sec.wav"
    rec_util.save_wave(ir_reverbe[0], reverbe_path)  # 保存
    reverbe_path = f"{out_dir}/reverbe_only/noise/{int(reverbe_sec * 10):02}sec_{angle_name}.wav"
    rec_util.save_wave(ir_reverbe[1], reverbe_path)  # 保存


if __name__ == "__main__":
    print("generate_IR")
    channel = 4
    distance = 6  # cm
    is_line = False
    if is_line:
        liner_circular = "liner"
    else:
        liner_circular = "circular"

    for reverbe_sec in tqdm(range(10, 100+1)):
        out_dir = os.path.join(const.SAMPLE_DATA_DIR, "IR", f"{channel}ch_{distance}cm_")
        reverbe_par_json = f"C:/Users/kataoka-lab/Desktop/sound_data/mix_data/reverbe_condition/6cm/{reverbe_sec:03}sec_{channel}ch_{distance}cm.json"
        # print("json_path:", reverbe_par_json)
        with open(reverbe_par_json, "r") as json_file:
            json_data = json.load(json_file)
            reverbe_par = json_data["reverbe_par"]

        rec_IR(out_dir, reverbe_sec*0.01, reverbe_par, channel=channel, distance=distance, angle=np.pi, angle_name="None",
               is_line=is_line)
