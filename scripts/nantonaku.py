# scripts/new_signal_noise.py
import json
import random
import numpy as np
import pyroomacoustics as pa
from tqdm import tqdm
from pathlib import Path
import sys
import argparse
import decimal

# 親ディレクトリをsys.pathに追加
# sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
	from mymodule import rec_utility as rec_util
	from mymodule import const
	# (rec_utility.py に移動させた定数と関数をインポート)
	from mymodule.rec_utility import (
		SAMPLING_RATE,
		load_yaml_config,
		load_wav,
		save_wav,
		get_file_list,
		get_random_value,
		get_mic_array,
		get_source_positions,
		compute_rirs,
		convolve_and_mix,
		calculate_c50,
		calculate_d50,
	)
except ImportError:
	print("=" * 50)
	print("❌ エラー: 'mymodule' が見つかりません。")
	print("リポジトリのルートで 'pip install -e .' を実行しましたか？")
	print("=" * 50)
	sys.exit(1)


class NumpyDecimalEncoder(json.JSONEncoder):
	def default(self, o):
		if isinstance(o, np.integer):
			return int(o)
		if isinstance(o, np.floating):
			return float(o)
		if isinstance(o, np.ndarray):
			return o.tolist()
		if isinstance(o, decimal.Decimal):
			return str(o)
		return super(NumpyDecimalEncoder, self).default(o)


def load_and_filter_params(precomputed_path: Path, rt60_range: list):
	"""
	事前計算されたJSONファイル(単一またはディレクトリ)を読み込み、指定RT60範囲のパラメータを抽出する
	"""
	all_params = []
	
	# パスがディレクトリかファイルかを判定
	if precomputed_path.is_dir():
		json_files = get_file_list(precomputed_path, '.json')
		if not json_files:
			raise FileNotFoundError(f"指定されたディレクトリに事前計算ファイルが0件です: {precomputed_path}")
	elif precomputed_path.is_file():
		if precomputed_path.suffix != '.json':
			raise ValueError(f"指定されたファイルはJSONファイルではありません: {precomputed_path}")
		json_files = [precomputed_path]
	else:
		raise FileNotFoundError(f"指定されたパスが見つかりません: {precomputed_path}")

	print(f"    - {len(json_files)} 個の事前計算ファイルからパラメータを読み込みます...")

	for json_path in tqdm(json_files, desc="Loading params", leave=False):
		json_path = Path(json_path)
		with open(json_path, 'r') as f:
			room_params_data = json.load(f)

		dims_str = json_path.stem.replace('cm', '').split('_')
		room_dim = [float(dims_str[0])*0.01, float(dims_str[1])*0.01, float(dims_str[2])*0.01]

		for rt60_key, params in room_params_data.items():
			rt60_val = float(rt60_key.replace('s', ''))
			if rt60_range[0] <= rt60_val <= rt60_range[1]:
				all_params.append({
					"json_path": json_path.name,
					"room_dim": room_dim,
					"rt60": rt60_val,
					"absorption": params['absorption'],
					"max_order": params['max_order']
				})

	if not all_params:
		raise ValueError(f"指定されたRT60範囲 [{rt60_range[0]}, {rt60_range[1]}] に一致するパラメータが、"
						 f"{precomputed_path} 内に見つかりませんでした。")

	print(f"    - ✅ RT60範囲に一致するパラメータ候補を {len(all_params)} 件発見しました。")
	return all_params


def generate_dataset(config_path):
	"""
	YAML設定ファイルに基づき、データセット生成の全プロセスを実行する
	"""
	# 1. 設定ファイルの読み込み
	try:
		config = load_yaml_config(config_path)
	except FileNotFoundError:
		print(f"❌ エラー: 設定ファイルが見つかりません: {config_path}", file=sys.stderr)
		sys.exit(1)
	except Exception as e:
		print(f"❌ エラー: 設定ファイル {config_path} の読み込みに失敗: {e}", file=sys.stderr)
		sys.exit(1)

	print(f"✅ 設定ファイル {config_path} を読み込みました。")

	paths = config['path']
	# YAMLから読み取ったパスをPathオブジェクトに変換
	precomputed_path = const.PARMS_DATA_DIR / config['room']['precomputed_path']
	speech_root = const.SPEECH_DATA_DIR / paths['sound_dir_name']
	noise_root = const.NOISE_DATA_DIR / paths['noise_dir_name']
	output_root = const.MIX_DATA_DIR / paths['output_dir_name']

	# 2. 事前計算パラメータとノイズファイルのリスト取得
	try:
		# ディレクトリ/ファイル両対応の関数を呼び出す
		all_valid_room_params = load_and_filter_params(precomputed_path, config['room']['rt60'])
		all_noise_files = get_file_list(noise_root, '.wav')
		print(f"    - {len(all_noise_files)} 個の雑音ファイルを発見。")
	except (FileNotFoundError, ValueError) as e:
		print(f"❌ エラー: {e}", file=sys.stderr)
		sys.exit(1)

	if not all_noise_files:
		print(f"❌ エラー: 雑音ファイルが0件です: {noise_root}", file=sys.stderr)
		sys.exit(1)

	# 3. 各スプリットごとに処理
	for split in paths['wave_type_list']:
		print(f"\n--- \"{split}\" の処理を開始 ---")
		split_speech_dir = speech_root / split
		try:
			split_speech_files = get_file_list(split_speech_dir, '.wav')
			print(f"    - {split}: {len(split_speech_files)} 個の教師用音声ファイルを発見。")
			if not split_speech_files:
				print(f"警告: {split_speech_dir} に音声ファイルがありません。スキップします。")
				continue
		except FileNotFoundError:
			print(f"警告: {split_speech_dir} が見つかりません。スキップします。")
			continue

		process_split(
			config=config,
			split=split,
			split_speech_files=split_speech_files,
			all_noise_files=all_noise_files,
			all_valid_room_params=all_valid_room_params,
			output_root=output_root
		)


def process_split(config, split, split_speech_files, all_noise_files, all_valid_room_params, output_root):
	num_files = len(split_speech_files)
	print(f"    - {split}: {num_files} 個のファイルを生成します。")

	for i, speech_filepath in enumerate(tqdm(split_speech_files, desc=f"Generating {split} files")):
		selected_room_param = random.choice(all_valid_room_params)
		output_dir = output_root / split

		generate_single_file(
			config=config,
			file_id=i,
			output_dir=output_dir,
			speech_filepath=speech_filepath,
			all_noise_files=all_noise_files,
			selected_room_param=selected_room_param
		)


def generate_single_file(config, file_id, output_dir, speech_filepath, all_noise_files, selected_room_param):
	room_dim = selected_room_param['room_dim']
	absorption = selected_room_param['absorption']
	max_order = selected_room_param['max_order']
	actual_rt60 = selected_room_param['rt60']

	try:
		room = pa.ShoeBox(
			room_dim,
			fs=SAMPLING_RATE,
			max_order=max_order,
			materials=pa.Material(absorption),
			air_absorption=True
		)
	except ValueError as e:
		tqdm.write(f"警告: Room生成に失敗 {room_dim}, {absorption}. スキップ。 ({e})")
		return

	room_center = np.array(room_dim) / 2.0
	mic_coords = get_mic_array(config['mic'], room_center)
	speech_pos_list = get_source_positions(
		config['source']['speech'], mic_coords[:, 0]
	)
	noise_pos_list = get_source_positions(
		config['source']['noise'], mic_coords[:, 0]
	)

	rir_dict = compute_rirs(room, mic_coords, speech_pos_list, noise_pos_list)
	rir_speech = rir_dict['rir_speech'][0]
	rir_noise = rir_dict['rir_noise'][0]

	try:
		clean_signal, _ = load_wav(speech_filepath, sr=SAMPLING_RATE)
		noise_filepath = Path(random.choice(all_noise_files))
		noise_signal, _ = load_wav(noise_filepath, sr=SAMPLING_RATE)

		snr_db_range = config['room']['snr']
		snr_db = random.randrange(snr_db_range[0], snr_db_range[1] + 1, snr_db_range[2])
	except Exception as e:
		tqdm.write(f"❌ ファイル読み込みエラー: {e}", file=sys.stderr)
		return

	signal_dict = convolve_and_mix(
		clean_signal,
		noise_signal,
		rir_speech,
		rir_noise,
		snr_db
	)

	base_filename = f"{speech_filepath.stem}_{int(actual_rt60*1000):}msec_snr{int(snr_db)}dB"
	rec_util.save_wav(
		output_dir / "noise_reverb" / f"{base_filename}_mix.wav",
		signal_dict['noise_reverb']
	)
	rec_util.save_wav(
		output_dir / "reverb_only" / f"{base_filename}_reverb.wav",
		signal_dict['reverb_only']
	)
	rec_util.save_wav(
		output_dir / "noise_only" / f"{base_filename}_noise.wav",
		signal_dict['noise_only']
	)
	rec_util.save_wav(
		output_dir / "clean" / f"{base_filename}_clean.wav",
		signal_dict['clean_speech']
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="YAML設定に基づき、音響データセットを生成します。"
	)
	parser.add_argument(
		'--config',
		type=str,
		default="C:/Users/kataoka-lab/Desktop/PyRoomAcoustics/config/sample/sample.yml",
		help="データセット生成の仕様を定義したYAMLファイルのパス"
	)
	args = parser.parse_args()

	generate_dataset(args.config)
