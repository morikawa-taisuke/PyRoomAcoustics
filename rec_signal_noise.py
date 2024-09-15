import os.path
import wave as wave
import pyroomacoustics as pa
import numpy as np
from tqdm import tqdm
import scipy
import random
import torchaudio
import torchaudio.transforms as transforms
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

# my_module
import rec_config as rec_conf
import rec_utility as rec_util
from mymodule import my_func, const


def serch_reverbe_sec(reverbe_sec, channel=1, angle=np.pi):
    reverbe = reverbe_sec
    cnt = 0
    room_dim = np.r_[10.0, 10.0, 10.0]
    """ 音源の読み込み """
    target_data = rec_util.load_wave_data(f'./mymodule/JA01F049.wav')
    noise_data = target_data
    wave_data = []  # 1つの配列に格納
    wave_data.append(target_data)
    wave_data.append(noise_data)
    mic_center = room_dim / 2  # アレイマイクの中心[x,y,z](m)
    num_channels = channel  # マイクの個数(チャンネル数)
    distance = 0.1  # 各マイクの間隔(m)
    mic_codinate = rec_util.set_mic_coordinate(center=mic_center,
                                               num_channels=num_channels,
                                               distance=distance)  # 各マイクの座標
    doas = np.array([
        [np.pi/2., np.pi/2],
        [np.pi/2., angle]
    ])  # 音源の方向[仰角, 方位角](ラジアン)
    distance = [2., 3.]  # 音源とマイクの距離(m)

    while True:
        e_absorption, max_order = pa.inverse_sabine(reverbe, room_dim)  # Sabineの残響式から壁の吸収率と反射上限回数を決定
        room = pa.ShoeBox(room_dim, fs=rec_conf.sampling_rate, max_order=max_order, absorption=e_absorption)

        """ 部屋にマイクを設置 """
        room.add_microphone_array(pa.MicrophoneArray(mic_codinate, fs=room.fs))
        """ 各音源の座標 """
        source_codinate = rec_util.set_souces_coordinate2(doas, distance, mic_center)
        """ 各音源を部屋に追加する """
        for idx in range(2):
            wave_data[idx] /= np.std(wave_data[idx])
            room.add_source(source_codinate[:, idx], signal=wave_data[idx])

        room.simulate()
        rt60 = room.measure_rt60()
        round_rt60 = round(np.mean(rt60), 3)
        if round_rt60 >= reverbe_sec or cnt == 100:
            break
        cnt += 1
        reverbe += 0.01
    # print(f'[{cnt}]rt60:{np.mean(rt60)}')
    print(f'max_order:{max_order}\ne_absorption:{e_absorption}')
    print(f'rt60={np.mean(rt60)}')
    return e_absorption, max_order


def recoding(wave_files, out_dir, snr, reverbe_sec, channel=1, is_split=False):
    """ シミュレーションを用いた録音

    Args:
        wave_files: シミュレーションで使用する音声([目的音声,雑音]) [0]:目的音声　[1]:雑音
        out_dir: 出力先のディレクトリ
        snr: 雑音と原音のSNR
        reverbe_sec: 残響時間
        channel: チャンネル数
        is_split: Ture=チャンネルごとにファイルを分ける False=すべてのチャンネルを1つのファイルにまとめる

    Returns:
        None
    """

    """ ファイル名の取得 """
    signal_name = rec_util.get_file_name(wave_files[0])
    noise_name = rec_util.get_file_name(wave_files[1])
    print(f'signal_name:{signal_name}')
    print(f'noise_name:{noise_name}')

    """ 音源の読み込み """
    target_data = rec_util.load_wave_data(wave_files[0])
    noise_data = rec_util.load_wave_data(wave_files[1])
    # print(f'target_data.shape:{target_data.shape}')     # 確認用
    # print(f'noise_data.shape:{noise_data.shape}')       # 確認用

    """ 雑音データをランダムに切り出す """
    start = random.randint(0, len(noise_data) - len(target_data))  # スタート位置をランダムに決定
    noise_data = noise_data[start: start + len(target_data)]  # noise_dataを切り出す
    scale_nosie = rec_util.get_scale_noise(target_data, noise_data, snr)  # noise_dataの大きさを調節
    # print(f'len(target_data):{len(target_data)}')   # 確認用
    # print(f'len(noise_data):{len(noise_data)}') # 確認用
    # print(f'len(scale_nosie):{len(scale_nosie)}')   # 確認用

    wave_data = []  # 1つの配列に格納
    wave_data.append(target_data)
    wave_data.append(scale_nosie)

    """ 音源のパラメータ """
    sample_rate = rec_conf.sampling_rate  # サンプリング周波数
    """ シミュレーションのパラメータ """
    room_dim = np.r_[10.0, 10.0, 10.0]  # 部屋の大きさ[x,y,z](m)
    num_sources = len(wave_data)  # シミュレーションで用いる音源数
    reverberation = reverbe_sec  # 残響時間(sec)
    # e_absorption, max_order = pa.inverse_sabine(reverberation, room_dim)  # Sabineの残響式から壁の吸収率と反射上限回数を決定
    e_absorption, max_order = serch_reverbe_sec(reverbe_sec=reverberation)
    mic_center = room_dim / 2  # アレイマイクの中心[x,y,z](m)
    num_channels = channel  # マイクの個数(チャンネル数)
    distance = 0.1  # 各マイクの間隔(m)
    mic_codinate = rec_util.set_mic_coordinate(center=mic_center,
                                               num_channels=num_channels,
                                               distance=distance)  # 各マイクの座標

    doas = np.array([
        [np.pi / 2., np.pi / 2],
        [np.pi / 2., np.pi]
    ])  # 音源の方向[仰角, 方位角](ラジアン)
    distance = [2., 3.]  # 音源とマイクの距離(m)

    """ 部屋の生成 """
    room_mix = pa.ShoeBox(room_dim, fs=sample_rate, max_order=max_order, absorption=e_absorption)
    room_reverbe = pa.ShoeBox(room_dim, fs=sample_rate, max_order=max_order, absorption=e_absorption)
    room_noise = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0, absorption=1.0)
    room_clean = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0, absorption=1.0)

    """ 部屋にマイクを設置 """
    room_mix.add_microphone_array(pa.MicrophoneArray(mic_codinate, fs=room_mix.fs))
    room_reverbe.add_microphone_array(pa.MicrophoneArray(mic_codinate, fs=room_mix.fs))
    room_noise.add_microphone_array(pa.MicrophoneArray(mic_codinate, fs=room_mix.fs))
    room_clean.add_microphone_array(pa.MicrophoneArray(mic_codinate, fs=room_mix.fs))

    """ 各音源の座標 """
    source_codinate = rec_util.set_souces_coordinate2(doas, distance, mic_center)

    """ 各音源を部屋に追加する """
    for idx in range(num_sources):
        wave_data[idx] /= np.std(wave_data[idx])
        room_mix.add_source(source_codinate[:, idx], signal=wave_data[idx])
        room_noise.add_source(source_codinate[:, idx], signal=wave_data[idx])
        if idx == 0:
            room_reverbe.add_source(source_codinate[:, idx], signal=wave_data[idx])
            room_clean.add_source(source_codinate[:, idx], signal=wave_data[idx])

    """ シミュレーションを回す """
    room_mix.simulate()
    room_reverbe.simulate()
    room_noise.simulate()
    room_clean.simulate()

    """ 畳み込んだ波形を取得する(チャンネル、サンプル）"""
    result_mix = room_mix.mic_array.signals
    result_reverbe = room_reverbe.mic_array.signals
    result_noise = room_noise.mic_array.signals
    result_clean = room_clean.mic_array.signals

    """ 残響時間の確認 """
    # rt60 = room_mix.measure_rt60()
    # print(f'mix：{rt60}')
    # rt60 = room_reverbe.measure_rt60()
    # print(f'reverbe：{rt60}')
    # rt60 = room_noise.measure_rt60()
    # print(f'noise：{rt60}')
    # rt60 = room_clean.measure_rt60()
    # print(f'clean：{rt60}')

    """ 畳み込んだ波形をファイルに書き込む """
    if is_split:
        """チャンネルごとにファイルを分けて保存する"""
        for i in range(num_channels):
            """ noise_reverberation """
            mix_path = f'./{out_dir}/mix_split/{i + 1:02}ch/{i + 1:02}ch_{signal_name}_{noise_name}_{snr}db_{int(reverbe_sec*10):02}sec.wav'
            rec_util.save_wave(result_mix[i, :] * np.iinfo(np.int16).max / 20.,
                               mix_path, sample_rate)
            """ reverberation_only """
            reverbe_path = f'./{out_dir}/reverbe_split/{i + 1:02}ch/{i + 1:02}ch_{signal_name}_{int(reverbe_sec*10):02}sec.wav'
            rec_util.save_wave(result_mix[i, :] * np.iinfo(np.int16).max / 20.,
                               reverbe_path, sample_rate)
            """ noise_only """
            noise_path = f'./{out_dir}/noise_split/{i + 1:02}ch/{i + 1:02}ch_{signal_name}_{noise_name}_{snr}db.wav'
            rec_util.save_wave(result_noise[i, :] * np.iinfo(np.int16).max / 20.,
                               noise_path, sample_rate)
            """ clean """
            clean_path = f'./{out_dir}/clean_split/{i + 1:02}ch/{i + 1:02}ch_{signal_name}.wav'
            rec_util.save_wave(result_clean[i, :] * np.iinfo(np.int16).max / 20.,
                               clean_path, sample_rate)
    else:
        """ チャンネルをまとめて保存 """
        """ noise_reverberation """
        mix_path = f'./{out_dir}/noise_reverberation/{signal_name}_{noise_name}_{snr}db_{int(reverbe_sec*10):02}sec.wav'
        result_mix = result_mix * np.iinfo(np.int16).max / 15  # スケーリング
        # print(f'result_mix.shape:{result_mix.shape}')
        rec_util.save_wave(result_mix, mix_path)  # 保存
        """ reverberation_only """
        reverbe_path = f'./{out_dir}/reverberation_only/{signal_name}_{int(reverbe_sec*10):02}sec.wav'
        result_reverbe = result_reverbe * np.iinfo(np.int16).max / 15  # 全てのチャンネルを保存
        # print(f'result_reverbe.shape:{result_reverbe.shape}')               # 確認用
        rec_util.save_wave(result_reverbe, reverbe_path)  # 保存
        """ nosie_only """
        noise_path = f'./{out_dir}/noise_only/{signal_name}_{noise_name}_{snr}db.wav'
        result_noise = result_noise * np.iinfo(np.int16).max / 15  # 全てのチャンネルを保存
        # print(f'result_nosie.shape:{result_noise.shape}')               # 確認用
        rec_util.save_wave(result_noise, noise_path)  # 保存
        """ clean """
        clean_path = f'./{out_dir}/clean/{signal_name}.wav'
        result_clean = result_clean * np.iinfo(np.int16).max / 15  # 全てのチャンネルを保存
        # print(f'result_clean.shape:{result_clean.shape}')               # 確認用
        rec_util.save_wave(result_clean, clean_path)  # 保存


def recoding2(wave_files, out_dir, snr, reverbe_sec, reverbe_par, channel=1, is_split=False, angle=np.pi):
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
    print('out_dir:', out_dir)

    """ ファイル名の取得 """
    signal_name = rec_util.get_file_name(wave_files[0])
    noise_name = rec_util.get_file_name(wave_files[1])
    # print(f'signal_name:{signal_name}')
    # print(f'noise_name:{noise_name}')

    """ 音源の読み込み """
    target_data = rec_util.load_wave_data(wave_files[0])
    noise_data = rec_util.load_wave_data(wave_files[1])
    # print(f'target_data.shape:{target_data.shape}')     # 確認用
    # print(f'noise_data.shape:{noise_data.shape}')       # 確認用

    """ 雑音データをランダムに切り出す """
    start = random.randint(0, len(noise_data) - len(target_data))  # スタート位置をランダムに決定
    noise_data = noise_data[start: start + len(target_data)]  # noise_dataを切り出す
    scale_nosie = rec_util.get_scale_noise(target_data, noise_data, snr)  # noise_dataの大きさを調節
    # print(f'len(target_data):{len(target_data)}')   # 確認用
    # print(f'len(noise_data):{len(noise_data)}') # 確認用
    # print(f'len(scale_nosie):{len(scale_nosie)}')   # 確認用

    wave_data = [target_data, scale_nosie]  # 1つの配列に格納

    """ 音源のパラメータ """
    sample_rate = rec_conf.sampling_rate  # サンプリング周波数
    """ シミュレーションのパラメータ """
    room_dim = np.r_[10.0, 10.0, 10.0]  # 部屋の大きさ[x,y,z](m)
    num_sources = len(wave_data)  # シミュレーションで用いる音源数
    mic_center = room_dim / 2  # アレイマイクの中心[x,y,z](m)
    num_channels = channel  # マイクの個数(チャンネル数)
    distance = 0.06  # 各マイクの間隔(m)
    mic_coordinate = rec_util.set_mic_coordinate(center=mic_center, num_channels=num_channels, distance=distance)  # 各マイクの座標
    # mic_coordinate = rec_util.set_circular_mic_coordinate(center=mic_center, num_channels=num_channels, radius=distance)

    doas = np.array([
        [np.pi/2., np.pi/2],
        [np.pi/2., angle]
    ])  # 音源の方向[仰角, 方位角](ラジアン)
    distance = [2., 3.]  # 音源とマイクの距離(m)

    """ 部屋の生成 """
    room_mix = pa.ShoeBox(room_dim, fs=sample_rate, max_order=reverbe_par[1], absorption=reverbe_par[0])
    room_reverbe = pa.ShoeBox(room_dim, fs=sample_rate, max_order=reverbe_par[1], absorption=reverbe_par[0])
    room_noise = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0, absorption=1.0)
    room_clean = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0, absorption=1.0)

    """ 部屋にマイクを設置 """
    room_mix.add_microphone_array(pa.MicrophoneArray(mic_coordinate, fs=room_mix.fs))
    room_reverbe.add_microphone_array(pa.MicrophoneArray(mic_coordinate, fs=room_reverbe.fs))
    room_noise.add_microphone_array(pa.MicrophoneArray(mic_coordinate, fs=room_noise.fs))
    room_clean.add_microphone_array(pa.MicrophoneArray(mic_coordinate, fs=room_clean.fs))

    """ 各音源の座標 """
    source_coordinate = rec_util.set_souces_coordinate2(doas, distance, mic_center)

    """ 各音源を部屋に追加する """
    for idx in range(num_sources):
        wave_data[idx] /= np.std(wave_data[idx])
        room_mix.add_source(source_coordinate[:, idx], signal=wave_data[idx])
        room_noise.add_source(source_coordinate[:, idx], signal=wave_data[idx])
        if idx == 0:
            room_reverbe.add_source(source_coordinate[:, idx], signal=wave_data[idx])
            room_clean.add_source(source_coordinate[:, idx], signal=wave_data[idx])

    """ シミュレーションを回す """
    room_mix.simulate()
    room_reverbe.simulate()
    room_noise.simulate()
    room_clean.simulate()

    """ 畳み込んだ波形を取得する(チャンネル、サンプル）"""
    result_mix = room_mix.mic_array.signals
    result_reverbe = room_reverbe.mic_array.signals
    result_noise = room_noise.mic_array.signals
    result_clean = room_clean.mic_array.signals

    """ 残響時間の確認 """
    # rt60 = room_mix.measure_rt60()
    # print(f'mix：{rt60}')
    # rt60 = room_reverbe.measure_rt60()
    # print(f'reverbe：{rt60}')
    # rt60 = room_noise.measure_rt60()
    # print(f'noise：{rt60}')
    # rt60 = room_clean.measure_rt60()
    # print(f'clean：{rt60}')

    """ 畳み込んだ波形をファイルに書き込む """
    if is_split:
        """チャンネルごとにファイルを分けて保存する"""
        for i in range(num_channels):
            """ noise_reverberation """
            mix_path = f'{out_dir}/split/noise_reverbe_split/{i + 1:02}ch/{i + 1:02}ch_{signal_name}_{noise_name}_{snr}db_{int(reverbe_sec*10):02}sec.wav'
            rec_util.save_wave(result_mix[i, :] * np.iinfo(np.int16).max / 15, mix_path, sample_rate)
            """ reverberation_only """
            reverbe_path = f'{out_dir}/split/reverbe_only_split/{i + 1:02}ch/{i + 1:02}ch_{signal_name}_{int(reverbe_sec*10):02}sec.wav'
            rec_util.save_wave(result_reverbe[i, :] * np.iinfo(np.int16).max / 15, reverbe_path, sample_rate)
            """ noise_only """
            noise_path = f'{out_dir}/split/noise_only_split/{i + 1:02}ch/{i + 1:02}ch_{signal_name}_{noise_name}_{snr}db.wav'
            rec_util.save_wave(result_noise[i, :] * np.iinfo(np.int16).max / 15, noise_path, sample_rate)
            """ clean """
            clean_path = f'{out_dir}/split/clean_split/{i + 1:02}ch/{i + 1:02}ch_{signal_name}.wav'
            rec_util.save_wave(result_clean[i, :] * np.iinfo(np.int16).max / 15, clean_path, sample_rate)
    else:
        """ チャンネルをまとめて保存 """
        """ noise_reverberation """
        mix_path = f'{out_dir}/noise_reverbe/{signal_name}_{noise_name}_{snr}db_{int(reverbe_sec*10):02}sec.wav'
        result_mix = result_mix * np.iinfo(np.int16).max / 15  # スケーリング
        # print(f'result_mix.shape:{result_mix.shape}')
        rec_util.save_wave(result_mix, mix_path)  # 保存
        """ reverberation_only """
        reverbe_path = f'{out_dir}/reverbe_only/{signal_name}_{int(reverbe_sec*10):02}sec.wav'
        result_reverbe = result_reverbe * np.iinfo(np.int16).max / 15  # 全てのチャンネルを保存
        # print(f'result_reverbe.shape:{result_reverbe.shape}')               # 確認用
        rec_util.save_wave(result_reverbe, reverbe_path)  # 保存
        """ nosie_only """
        noise_path = f'{out_dir}/noise_only/{signal_name}_{noise_name}_{snr}db.wav'
        result_noise = result_noise * np.iinfo(np.int16).max / 15  # 全てのチャンネルを保存
        # print(f'result_nosie.shape:{result_noise.shape}')               # 確認用
        rec_util.save_wave(result_noise, noise_path)  # 保存
        """ clean """
        clean_path = f'{out_dir}/clean/{signal_name}_{noise_name}_{snr}db_{int(reverbe_sec*10):02}sec.wav'
        # clean_path = f'{out_dir}/clean/{signal_name}.wav'
        result_clean = result_clean * np.iinfo(np.int16).max / 15  # 全てのチャンネルを保存
        # print(f'result_clean.shape:{result_clean.shape}')               # 確認用
        rec_util.save_wave(result_clean, clean_path)  # 保存

def process_recoding_thread(angle, angle_name):
    speech_type = 'subset_DEMAND'
    noise_type = 'hoth'
    target_dir = f'{const.SAMPLE_DATA_DIR}\\speech\\{speech_type}\\'  # 目的信号のディレクトリ
    sub_dir_list = my_func.get_subdir_list(target_dir)
    noise_path = f'{const.SAMPLE_DATA_DIR}\\noise\\{noise_type}.wav'  # 雑音信号のディレクトリ
    snr = 10  # SNR
    reverbe_sec = 0.5  # 残響
    ch = 4  # マイク数
    is_split = False  # 信号の保存方法 True:各チャンネルごとにファイルを分ける False:1つのファイルにまとめる
    out_dir = f"{const.MIX_DATA_DIR}\\{speech_type}_{noise_type}_{snr:02}{snr:02}dB_{int(reverbe_sec * 10):02}sec_{ch}ch_6cm\\{angle_name}"


    reverbe_par = serch_reverbe_sec(reverbe_sec=reverbe_sec, channel=ch, angle=angle)  # 任意の残響になるようなパラメータを求める
    for sub_dir in sub_dir_list:
        """音声ファイルリストの作成"""
        target_list = my_func.get_wave_filelist(os.path.join(target_dir, sub_dir))
        print(f'len:{len(target_list)}')
        for target_file in tqdm(target_list):
            wave_file = []
            wave_file.append(target_file)
            wave_file.append(noise_path)

            """録音(シミュレーション)"""
            recoding2(wave_files=wave_file,
                      out_dir=os.path.join(out_dir, sub_dir),
                      snr=snr,
                      reverbe_sec=reverbe_sec,
                      reverbe_par=reverbe_par,
                      channel=ch,
                      is_split=is_split,
                      angle=angle)

if __name__ == '__main__':
    print('main')
    """畳み込みに用いる音声波形(ディレクトリ)"""
    # noise_list = ['hoth', 'white']
    # learning_list = ['noise_reverberation', 'reverberation_only']
    # speech_list = ['JA', 'CMU']
    # type_list = ['training', 'test']
    #
    # speech = 'CMU'
    # noise = 'white'
    #
    # channel = 1
    # reverberation_sec = 0.7
    # snr = 10
    #
    # for type in type_list:
    #     target_dir = f"./wave/sample_data/speech/{speech}/{type}"
    #     noise_dir=f"./wave/sample_data/noise/{noise}.wav"
    #     # out_dir = conf.MIX_DIR + "MC_JA01_00_4ch_1ch"
    #     out_dir = f"./rec_12_12/{speech}_{noise}_{snr}db_{reverberation_sec}sec/{type}"
    #
    #     """音声ファイルリストの作成"""
    #     target_list = my_func.get_wave_filelist(target_dir)
    #     noise_list = my_func.get_wave_filelist(noise_dir)
    #     # print(f'len(target_list):{len(target_list)}')
    #     # print(f'len(noise_list):{len(noise_list)}')
    #
    #     # for trget_file in target_list:
    #     #   """録音(シミュレーション)"""
    #     #   main(trget_file, out_dir)
    #
    #     for target_file in target_list:
    #         for noise_file in noise_list:
    #             wave_file = []
    #             wave_file.append(target_file)
    #             wave_file.append(noise_file)
    #             # print(f'wave_file:{wave_file}') # 確認用
    #
    #             """録音(シミュレーション)"""
    #             # main(wave_file, out_dir)
    #             signal_noise(wave_files=wave_file,
    #                          out_dir=out_dir,
    #                          snr=10,
    #                          reverberation_sec=reverberation_sec,
    #                          channel=channel)

    """ シミュレーションの設定"""
    # speech_type = 'subset_DEMAND'
    # noise_type = 'hoth'
    # target_dir = f'{const.SAMPLE_DATA_DIR}\\speech\\{speech_type}\\'  # 目的信号のディレクトリ
    # sub_dir_list = my_func.get_subdir_list(target_dir)
    # noise_path = f'{const.SAMPLE_DATA_DIR}\\noise\\{noise_type}.wav'  # 雑音信号のディレクトリ
    # snr = 10  # SNR
    # reverbe_sec = 0.5  # 残響
    # ch = 4  # マイク数
    # is_split = False  # 信号の保存方法 True:各チャンネルごとにファイルを分ける False:1つのファイルにまとめる
    angle_list = [np.pi*i/4. for i in range(5)]
    angle_name_list = ['Right', 'FrontRight', 'Front', 'FrontLeft', 'Left']
    print(angle_list)
    # # for channel in channel_list:
    # # for reverbe_sec in reverbe_list:
    # reverbe_sec = reverbe_list[0]
    # out_dir = f"{const.MIX_DATA_DIR}\\{speech_type}_{noise_type}_{snr:02}{snr:02}dB_{int(reverbe_sec * 10):02}sec_{channel_list}ch_circular_10cm\\{angle_name}"
    start = time.time()
    with ProcessPoolExecutor() as executor:
        executor.map(process_recoding_thread,
                     angle_list,
                     angle_name_list,
                     )

    # for angle, angle_name in zip(angle_list, angle_name_list):
    #     out_dir = f"{const.MIX_DATA_DIR}\\{speech_type}_{noise_type}_{snr:02}{snr:02}dB_{int(reverbe_sec * 10):02}sec_{ch}ch_circular_10cm\\{angle_name}"
    #     print(f'out_dir:{out_dir}')
    #     reverbe_par = serch_reverbe_sec(reverbe_sec=reverbe_sec, channel=ch, angle=angle)  # 任意の残響になるようなパラメータを求める
    #     for sub_dir in sub_dir_list:
    #         """音声ファイルリストの作成"""
    #         target_list = my_func.get_wave_filelist(os.path.join(target_dir, sub_dir))
    #         print(f'len:{len(target_list)}')
    #         for target_file in tqdm(target_list):
    #             wave_file = []
    #             wave_file.append(target_file)
    #             wave_file.append(noise_path)
    #
    #             """録音(シミュレーション)"""
    #             # recoding(wave_files=wave_file, out_dir=out_dir, snr=snr, reverbe_sec=reverbe_sec,channel=channel, is_split=is_split)
    #             recoding2(wave_files=wave_file,
    #                       out_dir=os.path.join(out_dir, sub_dir),
    #                       snr=snr,
    #                       reverbe_sec=reverbe_sec,
    #                       reverbe_par=reverbe_par,
    #                       channel=ch,
    #                       is_split=is_split,
    #                       angle=angle)

    end = time.time()
    print(f'time:{(end-start)/60:.2f}')
