import wave as wave
import numpy as np

# my_module
import rec_config as rec_conf
import rec_utility as rec_util
from mymodule import my_func


def search_sample(wave_list):
	num_sample = 0  # 初期化
	for wave_file in wave_list:
		with wave.open(wave_file) as wav:
			if num_sample < wav.getnframes():
				num_sample = wav.getnframes()
	return num_sample

def get_delay_time(wave_files):
	""" 各chのサンプル数のズレを取得
	
	Parameters
	----------
	wave_files: ズレを取得する音声ファイル, clean_splitの中身を指定する

	Returns
	-------
	delay_time:各chのズレ(最も遅いchを基準にどれだけ速いかを表している)
	"""
	
	""" サンプル数の調査 """
	num_sample = search_sample(wave_files)
	""" 音声ファイルの読み込み """
	wave_data_list = np.zeros([len(wave_files), num_sample], dtype=np.float64)  # 初期化
	for idx, wave_file in enumerate(wave_files):
		# wave_data = rec_util.load_wave_data(wave_file)  # 読み込み
		# print(wave_data.shape)
		# wave_data_list[idx, :num_sample] = wave_data    # 追加
		wave_data_list[idx, :num_sample] = rec_util.load_wave_data(wave_file)  # 読み込み
	# print(wave_data_list.shape)
	# print(wave_data_list)
	""" 各チャンネルの開始位置を取得 """
	start_list = []  # 初期化
	for wave_data in wave_data_list:
		start = np.nonzero(wave_data)[0][0]  # 音声の開始位置を取得
		# print(start)
		# print(type(start))
		start_list.append(start)
	# print(start_list)
	
	""" 各チャンネルの遅延時間を計算 """
	max_idx = max(start_list)  # 一番遅い開始位置を取得
	# print(max_idx)
	delay_time = max_idx - start_list  # 各チャンネルの遅延時間を計算(最も遅いchと比較して)
	# print(delay_time)
	return delay_time


def delay_signal(wave_files, out_dir, is_split=False):
	""" ファイル間の位相差を合わせる

	Parameters
	----------
	wave_files: 音声ファイルのリスト[ch数]
	out_dir: 出力先
	is_split: Ture:chごとにファイルを分ける, False:1つにまとめて出力

	Returns
	-------
	None
	"""
	""" ファイル名の取得 """
	file_name = rec_util.get_file_name(wave_files[0]).replace("01ch_", "")	# ファイル名の取得とch番号の削除
	print(file_name)
	sample_rate = rec_conf.sampling_rate  # サンプリング周波数
	""" サンプル数の調査 """
	num_sample = search_sample(wave_files)
	""" 音声ファイルの読み込み """
	wave_data_list = np.zeros([len(wave_files), num_sample], dtype=np.float64)    # 初期化
	for idx, wave_file in enumerate(wave_files):
		wave_data = rec_util.load_wave_data(wave_file)  # 読み込み
		# print(wave_data.shape)
		# wave_data_list[idx, :num_sample] = wave_data    # 追加
		wave_data_list[idx, :num_sample] = rec_util.load_wave_data(wave_file)  # 読み込み
	# print(wave_data_list.shape)
	# print(wave_data_list)
	""" 音声データがあっているかを調べる """
	# cnt = 0
	# miss_match = 0
	# for idx1, data1 in enumerate(wave_data_list):
	# 	for idx2, data2 in enumerate(wave_data_list):
	# 		if all(data1 == data2):
	# 			print(idx1+1, idx2+1)
	# 			cnt += 1
	# 		else:
	# 			miss_match += 1
	# print(len(wave_data_list)**2)
	# print(cnt)
	# print(miss_match)
	""" 各チャンネルの開始位置を取得 """
	start_list = []   # 初期化
	for wave_data in wave_data_list:
		start = np.nonzero(wave_data)[0][0]   # 音声の開始位置を取得
		# print(start)
		# print(type(start))
		start_list.append(start)
	# print(start_list)
	
	""" 各チャンネルの遅延時間を計算 """
	max_idx = max(start_list)   # 一番遅い開始位置を取得
	# print(max_idx)
	delay_time = max_idx - start_list   # 各チャンネルの遅延時間を計算(最も遅いchと比較して)
	# print(delay_time)
	
	# for i in wave_data_list:
	# 	print(i.shape)
	
	""" 各チャンネルを遅らせる """
	delay_wave_list = np.zeros(wave_data_list.shape, dtype=np.float64)
	# print(delay_wave_list.shape)
	for idx, start_time in enumerate(delay_time):
		if start_time > 0:
			delay_wave_list[idx, start_time:] = wave_data_list[idx, :-1*start_time]
		else:
			delay_wave_list[idx, :] = wave_data_list[idx, :]
		
	# print(delay_wave_list.shape)
		
	""" 保存 """
	if is_split:
		"""チャンネルごとにファイルを分けて保存する"""
		for i in range(len(wave_files)):
			out_path = f"./{out_dir}_split/{i + 1:02}ch/{i + 1:02}ch_{file_name}.wav"
			rec_util.save_wave(delay_wave_list[i, :] * np.iinfo(np.int16).max / 15.0, out_path, sample_rate)
	else:
		""" チャンネルをまとめて保存 """
		out_path = f"./{out_dir}/{file_name}.wav"
		delay_wave_list = delay_wave_list * np.iinfo(np.int16).max  # スケーリング
		# print(f"result_mix.shape:{result_mix.shape}")
		rec_util.save_wave(delay_wave_list, out_path)  # 保存


def new_delay_signal(wave_files, out_dir, delay_time, is_split=False):
	""" ファイル間の位相差を合わせる

	Parameters
	----------
	wave_files: 音声ファイルのリスト[ch数]
	out_dir: 出力先
	is_split: Ture:chごとにファイルを分ける, False:1つにまとめて出力

	Returns
	-------
	None
	"""
	""" ファイル名の取得 """
	file_name = rec_util.get_file_name(wave_files[0]).replace("01ch_", "")  # ファイル名の取得とch番号の削除
	print(file_name)
	sample_rate = rec_conf.sampling_rate  # サンプリング周波数
	""" 音声ファイルの読み込み """
	num_sample = search_sample(wave_files)  # サンプル数の取得
	wave_data_list = np.zeros([len(wave_files), num_sample], dtype=np.float64)  # 初期化
	for idx, wave_file in enumerate(wave_files):
		# wave_data = rec_util.load_wave_data(wave_file)  # 読み込み
		# print(wave_data.shape)
		wave_data_list[idx, :num_sample] = rec_util.load_wave_data(wave_file)  # 読み込み
	# print(wave_data_list.shape)
	# print(wave_data_list)
	""" 各チャンネルを遅らせる """
	delay_wave_list = np.zeros(wave_data_list.shape, dtype=np.float64)
	# print(delay_wave_list.shape)
	for idx, start_time in enumerate(delay_time):
		if start_time > 0:
			delay_wave_list[idx, start_time:] = wave_data_list[idx, :-1 * start_time]
		else:
			delay_wave_list[idx, :] = wave_data_list[idx, :]
	# print(delay_wave_list.shape)
	""" 保存 """
	if is_split:
		"""チャンネルごとにファイルを分けて保存する"""
		for i in range(len(wave_files)):
			out_path = f"./{out_dir}/split/{i + 1:02}ch/{i + 1:02}ch_{file_name}.wav"
			rec_util.save_wave(delay_wave_list[i, :] * np.iinfo(np.int16).max / 15.0, out_path, sample_rate)
	else:
		""" チャンネルをまとめて保存 """
		out_path = f"./{out_dir}/{file_name}.wav"
		delay_wave_list = delay_wave_list * np.iinfo(np.int16).max  # スケーリング
		# print(f"result_mix.shape:{result_mix.shape}")
		rec_util.save_wave(delay_wave_list, out_path)  # 保存


if __name__ == "__main__":
	# list = ["./sound_data/rec_data/JA_07sec_4ch/training/clean_split/01ch/01ch_JA01F049.wav",
	# 		"./sound_data/rec_data/JA_07sec_4ch/training/clean_split/02ch/02ch_JA01F049.wav",
	# 		"./sound_data/rec_data/JA_07sec_4ch/training/clean_split/03ch/03ch_JA01F049.wav",
	# 		"./sound_data/rec_data/JA_07sec_4ch/training/clean_split/04ch/04ch_JA01F049.wav"]
	
	dir_path = "./sound_data/rec_data/JA_hoth_10dB_05sec_4ch/test/"
	split_dir_path = f"{dir_path}/split/"
	print(split_dir_path)
	out_dir = f"{dir_path}/delay/"
	sub_dir_list = my_func.get_subdir_list(split_dir_path)
	# print(sub_dir_list)
	list = [f"{dir_path}/split/clean_split/01ch/01ch_JA04F085.wav",
	        f"{dir_path}/split/clean_split/02ch/02ch_JA04F085.wav",
	        f"{dir_path}/split/clean_split/03ch/03ch_JA04F085.wav",
	        f"{dir_path}/split/clean_split/04ch/04ch_JA04F085.wav"]
	delay_time = get_delay_time(list)
	for sub_dir in sub_dir_list:
		ch01 = f"{split_dir_path}/{sub_dir}/01ch/"
		ch02 = f"{split_dir_path}/{sub_dir}/02ch/"
		ch03 = f"{split_dir_path}/{sub_dir}/03ch/"
		ch04 = f"{split_dir_path}/{sub_dir}/04ch/"
		list1 = my_func.get_wave_filelist(ch01)
		list2 = my_func.get_wave_filelist(ch02)
		list3 = my_func.get_wave_filelist(ch03)
		list4 = my_func.get_wave_filelist(ch04)
		for file_list in zip(list1, list2, list3, list4):
			new_delay_signal(wave_files=file_list, out_dir=f"{out_dir}/{sub_dir}/", delay_time=delay_time, is_split=False)

	# ch01 = "./sound_data/rec_data/JA_07sec_4ch/training/reverbe_only_split/01ch/"
	# ch02 = "./sound_data/rec_data/JA_07sec_4ch/training/reverbe_only_split/02ch/"
	# ch03 = "./sound_data/rec_data/JA_07sec_4ch/training/reverbe_only_split/03ch/"
	# ch04 = "./sound_data/rec_data/JA_07sec_4ch/training/reverbe_only_split/04ch/"
	# list1 = my_func.get_wave_filelist(ch01)
	# list2 = my_func.get_wave_filelist(ch02)
	# list3 = my_func.get_wave_filelist(ch03)
	# list4 = my_func.get_wave_filelist(ch04)
	# for file_list in zip(list1, list2, list3, list4):
	# 	delay_signal(file_list, out_dir="./sound_data/rec_data/JA_07sec_4ch/training/reverbe_delay", is_split=True)
