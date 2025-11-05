import os
import random

import pyroomacoustics as pa
from tqdm import tqdm

# mymodule内のモジュールをインポート
# パスが通っていない場合は、実行環境に合わせて調整してください
# import sys
# sys.path.append('C:/Users/kataoka-lab/Desktop/PyRoomAcoustics')
from mymodule import const, my_func
from rec_signal_noise import recoding2

""" パラメータ設定 """
speech_type = "DEMAND"
noise_type = "DEMAND"  # 雑音信号のディレクトリ
out_dir = f"{const.MIX_DATA_DIR}/DEMAND_DEMAND"  # 出力ディレクトリ

sub_dir_list = ["train", "val", "test"]  # サブディレクトリのリスト
ch = 1  # チャンネル数
distance = 0    # マイク間距離[m]

# SNR（Signal-to-Noise Ratio）の範囲
SNR_MIN = -5 # dB
SNR_MAX = 15 # dB

# 残響時間（Reverberation Time, Rt60）の範囲
Rt60_MIN = 0.3 # sec
Rt60_MAX = 1.0 # sec

noise_dir = os.path.join(const.SAMPLE_DATA_DIR, "noise", noise_type)  # 雑音信号のディレクトリ
noise_list = my_func.get_file_list(noise_dir)


for sub_dir in sub_dir_list:
	speech_dir = os.path.join(const.SAMPLE_DATA_DIR, "speech", speech_type, sub_dir)
	speech_list = my_func.get_file_list(speech_dir)
	print(f"speech:[{len(speech_list)}]:{speech_dir}")
	print(f"noise::[{len(noise_list)}]{noise_dir}")

	for speech_path in tqdm(speech_list):
		noise_path = random.choice(noise_list)  # ランダムに雑音ファイルを選択
		wave_file = [speech_path, noise_path]

		snr = random.randint(SNR_MIN, SNR_MAX)
		rt60 = round(random.uniform(Rt60_MIN, Rt60_MAX),2)
		reverbe_par = pa.inverse_sabine(rt60, [5., 5., 5.])  # Sabineの残響式から壁の吸収率と反射上限回数を決定

		recoding2(wave_files=wave_file,
		          out_dir=os.path.join(out_dir, sub_dir),
		          snr=snr,
		          reverbe_sec=rt60,
		          reverbe_par=reverbe_par)