# scripts/new_signal_noise.py
# (事前計算JSONとYAMLに基づきデータセットを高速生成するハイブリッド版)
# (train/val/test の入力ディレクトリ分離を尊重するロジックに修正済み)

import json
import random
import numpy as np
import pyroomacoustics as pa
from tqdm import tqdm
from pathlib import Path
import sys
import argparse
import decimal

# (リポジトリ構成案 に従い、mymoduleからインポート)
# setup.py でインストールされている前提
try:
	from mymodule import rec_utility as rec_util
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
		calculate_d50
	)
except ImportError:
	print("=" * 50)
	print("❌ エラー: 'mymodule' が見つかりません。")
	print("リポジトリのルートで 'pip install -e .' を実行しましたか？")
	print("=" * 50)
	sys.exit(1)


# JSONシリアライズ用
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


def generate_dataset(config_path):
	"""
	YAML設定ファイルに基づき、データセット生成の全プロセスを実行する
	(司令塔の役割)
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

	# 2. パスの設定
	base_paths = config['base_paths']
	precomputed_dir = Path(base_paths['precomputed_params_dir'])
	speech_root = Path(base_paths['speech_data_root'])  # (例: .../speech/subset_DEMAND)
	noise_root = Path(base_paths['noise_data_root'])
	output_root = Path(base_paths['output_data_root'])

	# 3. 事前計算パラメータとノイズファイルのリスト取得 (全スプリットで共通)
	try:
		precomputed_json_files = rec_util.get_file_list(precomputed_dir, '.json')
		print(f"    - {len(precomputed_json_files)} 個の事前計算された部屋パラメータを発見。")

		all_noise_files = rec_util.get_file_list(noise_root, '.wav')
		print(f"    - {len(all_noise_files)} 個の雑音ファイルを発見。")

	except FileNotFoundError as e:
		print(f"❌ エラー: {e}", file=sys.stderr)
		sys.exit(1)

	if not precomputed_json_files:
		print(f"❌ エラー: 事前計算ファイルが0件です: {precomputed_dir}", file=sys.stderr)
		sys.exit(1)
	if not all_noise_files:
		print(f"❌ エラー: 雑音ファイルが0件です: {noise_root}", file=sys.stderr)
		sys.exit(1)

	# ===================================================================
	# === ▼▼▼ 修正箇所: スプリットごとに分離して処理 ▼▼▼ ===
	# ===================================================================

	# 4. スプリットごとに処理 (入力ディレクトリの分離を維持)
	for split in config['splits']:  # (例: ["train", "val", "test"])
		print(f"\n--- \"{split}\" の処理を開始 ---")

		# 4a. (旧B案 と同様のロジック)
		# "train" なら .../subset_DEMAND/train/
		# "val"   なら .../subset_DEMAND/val/
		# "test"  なら .../subset_DEMAND/test/
		# のように、スプリット専用の入力ディレクトリパスを構築
		split_speech_dir = speech_root / split

		try:
			split_speech_files = rec_util.get_file_list(split_speech_dir, '.wav')
			print(f"    - {split}: {len(split_speech_files)} 個の教師用音声ファイルを発見。")
			if not split_speech_files:
				print(f"警告: {split_speech_dir} に音声ファイルがありません。スキップします。")
				continue

		except FileNotFoundError:
			print(f"警告: {split_speech_dir} が見つかりません。スキップします。")
			continue

		# 4b. 処理を process_split に委譲
		process_split(
			config=config,
			split=split,
			split_speech_files=split_speech_files,
			all_noise_files=all_noise_files,
			precomputed_json_files=precomputed_json_files,
			output_root=output_root
		)


# ===================================================================

def process_split(config, split, split_speech_files, all_noise_files,
                  precomputed_json_files, output_root):
	"""
	単一のスプリット（例: "train"）のデータセット生成を実行する
	"""
	settings = config['domain_generation_settings']

	# スプリット固有の部屋数を取得
	if split not in settings['num_rooms_per_split']:
		print(f"警告: {split} の部屋数 (num_rooms_per_split) が定義されていません。スキップ。")
		return
	num_rooms = settings['num_rooms_per_split'][split]

	num_files_setting = settings['num_files_per_room']
	total_files = len(split_speech_files)

	if num_files_setting == 'auto':
		# (教師データ数 // 部屋数) で自動計算
		# データ拡張なし（RIRの再利用なし）
		num_files_per_room = total_files // num_rooms
		if num_files_per_room == 0:
			print(f"警告: 部屋数({num_rooms}) がファイル数({total_files}) より多いため、")
			print(f"       部屋数を {total_files} に制限します。")
			num_rooms = total_files
			num_files_per_room = 1

		print(f"    - {split}: {total_files} ファイル / {num_rooms} 部屋 = {num_files_per_room} ファイル/部屋 (自動)")
		# 教師データを、各部屋に重複なく割り当てる
		speech_chunks = np.array_split(split_speech_files, num_rooms)

	else:
		# 固定値（例: 100）が指定された場合
		# データ拡張あり（RIRの再利用あり）
		num_files_per_room = int(num_files_setting)
		print(f"    - {split}: {num_rooms} 部屋 × {num_files_per_room} ファイル/部屋 (固定)")
		# 各部屋で全教師データからランダムサンプリングする
		speech_chunks = [
			# (注: np.random.choice はリスト全体をメモリコピーするため、
			#  ファイル数が多い場合は random.choices のが効率的)
			random.choices(split_speech_files, k=num_files_per_room)
			for _ in range(num_rooms)
		]

	# --- 部屋（ドメイン）ごとにループ ---
	for room_id in tqdm(range(num_rooms), desc=f"Simulating {split} rooms"):

		room_output_dir = output_root / settings['output_name'] / split / f"room_{room_id:04d}"
		room_speech_files = speech_chunks[room_id]

		# (YAMLで "room_selection" の分岐を追加)
		room_selection_strategy = settings.get('room_selection', {}).get('strategy', 'random')

		if room_selection_strategy == 'fixed':
			# "fixed_room_json" で指定されたJSONのみを使う
			json_name = settings['room_selection']['fixed_room_json']
			json_path = Path(config['base_paths']['precomputed_params_dir']) / json_name
			if not json_path.exists():
				tqdm.write(f"❌ エラー: 指定された固定部屋JSONが見つかりません: {json_name}", file=sys.stderr)
				continue
		else:  # 'random' (デフォルト)
			# 事前計算ディレクトリから「部屋サイズJSON」をランダムに選択
			json_path = Path(random.choice(precomputed_json_files))

		generate_room_data(
			config=config,
			room_id=room_id,
			room_output_dir=room_output_dir,
			room_speech_files=room_speech_files,
			all_noise_files=all_noise_files,
			selected_room_json_path=json_path
		)


def generate_room_data(config, room_id, room_output_dir, room_speech_files,
                       all_noise_files, selected_room_json_path):
	"""
	単一の部屋（RIR）のシミュレーションと、
	それに紐づく全ファイル（例: 100件）の生成を実行する
	(この関数の中身は、前回のコード案 [ステップ2.2] とほぼ同じ)
	"""

	settings = config['domain_generation_settings']
	rt60_range = settings['rt60_range']

	# 1. 部屋パラメータ（JSON）の読み込み
	json_path = selected_room_json_path
	with open(json_path, 'r') as f:
		room_params_data = json.load(f)

	# 2. RT60の選択
	available_keys = [
		k for k in room_params_data.keys()
		if rt60_range[0] <= float(k.replace('s', '')) <= rt60_range[1]
	]
	if not available_keys:
		tqdm.write(f"警告: {json_path.name} に {rt60_range} のRT60データがありません。スキップ。")
		return

	selected_rt60_key = random.choice(available_keys)
	selected_params = room_params_data[selected_rt60_key]

	absorption = selected_params['absorption']
	max_order = selected_params['max_order']
	actual_rt60 = float(selected_rt60_key.replace('s', ''))

	dims_str = json_path.stem.replace('m', '').split('_')
	room_dim = [float(dims_str[0]), float(dims_str[1]), float(dims_str[2])]

	# 3. Pyroomacoustics Roomオブジェクトの生成
	try:
		room = pa.ShoeBox(
			room_dim,
			fs=SAMPLING_RATE,
			max_order=max_order,
			materials=pa.Material(absorption),
			air_absorption=True  # (空気減衰は有効にしておくのが現実的)
		)
	except ValueError as e:
		tqdm.write(f"警告: Room生成に失敗 {room_dim}, {absorption}. スキップ。 ({e})")
		return

	# 4. マイクと音源の配置
	room_center = np.array(room_dim) / 2.0
	mic_coords = rec_util.get_mic_array(config['microphone'], room_center)

	speech_pos_list = rec_util.get_source_positions(
		config['speech_source'], mic_coords[:, 0]  # アレイ中心（マイク0）を基準
	)
	noise_pos_list = rec_util.get_source_positions(
		config['noise_source'], mic_coords[:, 0]
	)

	# 5. RIR計算
	rir_dict = rec_util.compute_rirs(room, mic_coords, speech_pos_list, noise_pos_list)
	# print(rir_dict)

	rir_speech = rir_dict['rir_speech'][0]  # (C, N_rir)
	rir_noise = rir_dict['rir_noise'][0]  # (C, N_rir)

	# 6. 音響特徴量の計算
	measured_rt60_all_channels = room.measure_rt60()
	measured_rt60 = np.mean(measured_rt60_all_channels[:, 0])  # 話者音源->全マイク の平均
	c50 = rec_util.calculate_c50(rir_speech[0])  # マイク0のC50
	d50 = rec_util.calculate_d50(rir_speech[0])  # マイク0のD50

	# 7. メタデータの保存 (中身は前回の案 [ステップ2.2] と同じ)
	metadata_path = room_output_dir / "metadata.json"
	room_metadata = {
		"room_id": room_id,
		"precomputed_json": json_path.name,
		"room_dim": room_dim,
		"target_rt60": actual_rt60,
		"measured_rt60": measured_rt60,
		"c50": c50,
		"d50": d50,
		"absorption": absorption,
		"max_order": max_order,
		"mic_config": config['microphone'],
		"speech_source_config": config['speech_source'],
		"noise_source_config": config['noise_source'],
		"files": []
	}

	# 8. ファイル（畳み込み）のループ
	output_flags = config['output_files']

	for speech_filepath in tqdm(room_speech_files, desc=f"Room {room_id:04d}", leave=False):
		try:
			# 8a. 音声・ノイズの選択
			clean_signal, _ = rec_util.load_wav(speech_filepath, sr=SAMPLING_RATE)

			noise_filepath = Path(random.choice(all_noise_files))
			noise_signal, _ = rec_util.load_wav(noise_filepath, sr=SAMPLING_RATE)

			# 8b. SNRの選択
			snr_db = rec_util.get_random_value(config['mixing_params']['snr_range_db'])

			# 8c. 畳み込みと混合
			signal_dict = rec_util.convolve_and_mix(
				clean_signal,
				noise_signal,
				rir_speech,  # (C, N_rir)
				rir_noise,  # (C, N_rir)
				snr_db
			)

			# 8d. ファイルの保存
			base_filename = f"{speech_filepath.stem}_room{room_id:04d}_snr{snr_db:.0f}"

			def shape_output(data):
				if data.shape[1] == 1:  # (N, 1) -> (N,)
					return data[:, 0]
				return data  # (N, C)

			if output_flags.get('save_noise_reverb', False):  # .get()でキーが存在しなくても安全に
				rec_util.save_wav(
					room_output_dir / "noise_reverb" / f"{base_filename}_mix.wav",
					shape_output(signal_dict['noise_reverb'])
				)
			if output_flags.get('save_reverb_only', False):
				rec_util.save_wav(
					room_output_dir / "reverb_only" / f"{base_filename}_reverb.wav",
					shape_output(signal_dict['reverb_only'])
				)
			if output_flags.get('save_noise_only', False):  # (YAMLのキーを 'save_noise_only' に変更)
				rec_util.save_wav(
					room_output_dir / "noise_only" / f"{base_filename}_noise.wav",
					shape_output(signal_dict['noise_only'])  # (signal_dict['reverberant_noise'] から変更)
				)
			if output_flags.get('save_clean', False):
				rec_util.save_wav(
					room_output_dir / "clean" / f"{base_filename}_clean.wav",
					shape_output(signal_dict['clean_speech'])
				)

			# 8e. ファイルメタデータの記録 (中身は前回の案 [ステップ2.2] と同じ)
			file_meta = {
				"filename_base": base_filename,
				"clean_source_file": str(speech_filepath.relative_to(config['base_paths']['speech_data_root'])),
				"noise_source_file": str(noise_filepath.relative_to(config['base_paths']['noise_data_root'])),
				"snr_db": snr_db
			}
			room_metadata["files"].append(file_meta)

		except Exception as e:
			# (ファイル個別のエラーは記録して続行)
			tqdm.write(f"❌ ファイル処理中にエラー: {speech_filepath.name} ({e})", file=sys.stderr)

	# 9. 部屋の全メタデータを保存
	try:
		room_output_dir.mkdir(parents=True, exist_ok=True)
		with open(metadata_path, "w") as f:
			json.dump(room_metadata, f, indent=4, cls=NumpyDecimalEncoder)
	except Exception as e:
		tqdm.write(f"❌ メタデータ保存中にエラー: {metadata_path} ({e})", file=sys.stderr)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="事前計算されたパラメータとYAML設定に基づき、音響データセットを高速に生成します。"
	)
	parser.add_argument(
		'--config',
		type=str,
		# required=True,
		default="C:/Users/kataoka-lab/Desktop/PyRoomAcoustics/config/dataset_v1.yml",
		help="データセット生成の仕様を定義したYAMLファイルのパス (例: configs/dataset_hybrid_v1.yml)"
	)
	args = parser.parse_args()

	generate_dataset(args.config)