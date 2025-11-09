# coding:utf-8

import os
import numpy as np
import torch
#import dataset
import const, my_func
import ConvTasNet

"""
def try_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e

#2バイトに変換してファイルに保存
#signal: time-domain 1d array (float)
#file_name: 出力先のファイル名
#sample_rate: サンプリングレート
def write_file_from_time_signal(signal,file_name,sample_rate):
    #2バイトのデータに変換
    signal=signal.astype(np.int16)
    #waveファイルに書き込む
    wave_out = wave.open(file_name, 'w')
    #モノラル:1、ステレオ:2
    wave_out.setnchannels(1)
    #サンプルサイズ2byte
    wave_out.setsampwidth(2)
    #サンプリング周波数
    wave_out.setframerate(sample_rate)
    #データを書き込み
    wave_out.writeframes(signal)
    #ファイルを閉じる
    wave_out.close()
"""
def psd(y_mixdown, name, fname, prm, FFT_SIZE=const.FFT_SIZE, H=const.H):
    #c = float(bias2)
    """
    # 評価するデータを取得するパスの準備
    input_mix = PATH_INPUT

    # 評価後のデータを保存するパスの準備
    output_mix = PATH_OUTPUT

    if os.path.isdir(input_mix):
        # 入力がディレクトリーの場合、ファイルリストをつくる
        filelist_mixdown = find_files(input_mix, ext="wav", case_sensitive=True)
    else:
        # 入力が単一ファイルの場合
        filelist_mixdown = [input_mix]

    print('number of mixdown file', len(filelist_mixdown))

    # 出力用のディレクトリーがない場合は　作成する。
    ext = my_func.exists_dir(PATH_OUTPUT)

    # ディレクトリを作成
    my_func.exists_dir(output_mix)
    """
    str_name = os.path.splitext(os.path.basename(str(name)))
    #print("str_name[0]", str_name[0])

    # モデルの読み込み
    TasNet_model = ConvTasNet.TasNet().to("cpu")
    TasNet_model.load_state_dict(torch.load('./pth/' + str(str_name[0]) + '.pth'))
    #TCN_model.load_state_dict(torch.load('reverb_03_snr20_reverb1020_snr20-clean_DNN-WPE_TCN_100.pth'))

    #for fmixdown in tqdm(filelist_mixdown):  # filelist_mixdownを全て確認して、それぞれをfmixdownに代入
    # y_mixdownは振幅、prmはパラメータ
    #y_mixdown, prm = my_func.load_wav(fmixdown)  # サンプリング周波数を合わせ、データをロードする
    # print("y_mixdown", y_mixdown.dtype)
    y_mixdown = y_mixdown.astype(np.float32)
    y_mixdown_max = np.max(y_mixdown)

    MIX = y_mixdown[np.newaxis, :]
    MIX = torch.from_numpy(MIX)
    #MIX = try_gpu(MIX)
    #print("MIX", MIX.shape)
    separate = TasNet_model(MIX)
    #print("separate", separate.shape)
    separate = separate.cpu()
    separate = separate.detach().numpy()
    tas_y_m = separate[0, 0, :]

    tas_y_m = tas_y_m * (y_mixdown_max / np.max(tas_y_m))
    """
    # 分離した speechを出力ファイルとして保存する。
    # 拡張子を変更したパス文字列を作成
    foutname, _ = os.path.splitext(os.path.basename(fmixdown))
    #ファイル名とフォルダ名を結合してパス文字列を作成
    fname = os.path.join(output_mix, (foutname + '.wav'))
    #print('saving... ', fname)
    #混合データを保存
    #mask = mask*y_mixdown
    """
    my_func.save_wav(fname, tas_y_m, prm)
    #my_func.save_wav(fname, tas_y_m, prm)
    return tas_y_m

if __name__ == '__main__':
    print("a")

    #separate('', '', '')
    #psd('../../data_sample/test/mix_05-k_2030/mix', '../../data_sample/test/tasnet_mix_05-k_2030-clean_04-mse',
    #       'train1020_mix_05-k_2030-clean_04_k-noise_2030_stft', '0.5')

