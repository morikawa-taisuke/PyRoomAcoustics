import yaml

import mymodule.my_func
import pyroomacoustics as pa
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import os
from mymodule import const
import pprint

# 親ディレクトリをsys.pathに追加
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.mymodule.rec_utility import compute_rirs, SAMPLING_RATE, get_source_positions, load_yaml_config, load_json_config, get_mic_array


def generate_rirs_from_metadata(metadata_path: Path, output_dir: Path):
	"""
	metadata.jsonを読み込み、RIRを生成してWAVファイルとして保存する
	"""
	if not metadata_path.exists():
		print(f"エラー: メタデータファイルが見つかりません: {metadata_path}")
		return

	with open(metadata_path, 'r', encoding='utf-8') as f:
		metadata = json.load(f)

	print(f"メタデータからRIRを生成します: {len(metadata)} 件")
	output_dir.mkdir(parents=True, exist_ok=True)

	room_params = metadata['room_dim']
	mic_params = room_params//2
	source_params = metadata['source']

	# 1. Pyroomacousticsの部屋オブジェクトを作成
	room = pa.ShoeBox(
		p=room_params,
		fs=SAMPLING_RATE,
		materials=pa.Material(room_params['absorption']),
		max_order=room_params['max_order']
	)

	# 2. マイクと音源の座標を取得
	mic_coords = np.array(mic_params).T  # (N, 3) -> (3, N)

	# compute_rirsは話者とノイズのリストを期待するため、
	# ここでは音源を「話者」として扱う
	speech_pos_list = [np.array(source_params['position'])]
	noise_pos_list = []

	# 3. RIRを計算
	rir_dict = compute_rirs(room, mic_coords, speech_pos_list, noise_pos_list)

	# 4. RIRをWAVファイルとして保存
	# rir_speech はリストであり、今回は音源が1つなので最初の要素を取得
	if rir_dict['rir_speech']:
		rir_data = rir_dict['rir_speech'][0]  # (C, N)

		# 保存パスを生成
		room_name = room_params.get('name', 'unknown_room')
		mic_name = mic_params.get('name', 'unknown_mic')
		source_name = source_params.get('name', 'unknown_source')

		# ファイル名に使えない文字を置換
		room_name_safe = room_name.replace(' ', '_')
		mic_name_safe = mic_name.replace(' ', '_')
		source_name_safe = source_name.replace(' ', '_')

		save_path = output_dir / f"{room_name_safe}" / f"{mic_name_safe}_{source_name_safe}.wav"
		save_path.parent.mkdir(parents=True, exist_ok=True)

		# (C, N) -> (N, C) に転置して保存
		sf.write(save_path, rir_data.T, SAMPLING_RATE)

	print(f"🎉 RIRの生成が完了しました。保存先: {output_dir.absolute()}")


def generate_rirs_from_metadata2(room_parms: dict, output_dir: Path, metadata: dict):
	"""
	metadata.jsonを読み込み、RIRを生成してWAVファイルとして保存する
	"""

	print(f"RIRを生成します: {len(room_parms)} 件")
	# output_dir.mkdir(parents=True, exist_ok=True)

	# 1. 各オブジェクトの座標を取得
	room_dim = np.array([5, 5, 5])  # 部屋の座標
	mic_position = get_mic_array(metadata['mic'], room_dim//2)    # マイクの座標
	specker_position = get_source_positions(metadata['source']['speech'], mic_center=room_dim//2)	# 話者の座標
	noise_position = get_source_positions(metadata['source']['noise'], mic_center=room_dim//2)	# ノイズの座標

	mic_config = metadata["mic"]
	ch = mic_config["channel"]
	if ch > 1:
		array_type = mic_config["array_type"]
		D = mic_config["D"]
		mic_name = f"{ch}ch_{array_type}_{D}cm"
	else:
		mic_name = f"{ch}ch"
	output_dir = output_dir / mic_name

	for rt60, room_parm in tqdm(room_parms.items()):
		# 2. 部屋を作成
		room = pa.ShoeBox(
			p=room_dim,
			fs=SAMPLING_RATE,
			max_order=room_parm['max_order'],
			materials = pa.Material(room_parm['absorption'])
		)

		# 3. RIRを計算
		rir_dict = compute_rirs(room, mic_position, specker_position, noise_position)

		# 4. RIRをWAVファイルとして保存
		# 保存パスを生成
		rt60 = int(float(rt60) * 1000)

		signal_rir_path = output_dir / "speech" / f"{rt60}ms.wav"
		noise_rir_path = output_dir / "noise" / f"{rt60}ms.wav"
		signal_rir_path.parent.mkdir(parents=True, exist_ok=True)
		noise_rir_path.parent.mkdir(parents=True, exist_ok=True)
		# (C, N) -> (N, C) に転置して保存
		sf.write(signal_rir_path, rir_dict["rir_speech"].T, SAMPLING_RATE)
		sf.write(noise_rir_path, rir_dict["rir_noise"].T, SAMPLING_RATE)

	print(f"🎉 RIRの生成が完了しました。保存先: {output_dir.absolute()}")

if __name__ == "__main__":
	# parser = argparse.ArgumentParser(
	#     description="metadata.jsonに基づいてRIRを生成し、WAVファイルとして保存します。"
	# )
	# parser.add_argument(
	#     '--metadata',
	#     type=str,
	#     default="./output/metadata.json",
	#     help="入力となるmetadata.jsonファイルのパス"
	# )
	# parser.add_argument(
	#     '--output',
	#     type=str,
	#     default="./output/rirs",
	#     help="生成されたRIR (WAV) の保存先ディレクトリ"
	# )
	# args = parser.parse_args()

	base_dir = const.SOUND_DATA_DIR
	output_dir = base_dir / 'RIR' / '500cm_500cm_500cm'

	print(base_dir)
	json_path = "C:/Users/kataoka-lab/Desktop/sound_data/preconpute_params/precomputed_params_2/500cm_500cm_500cm.json"
	yaml_path = Path("C:/Users/kataoka-lab/PycharmProjects/pythonProject/PyRoomAcoustics/config/sample/sample.yml")
	room_parms = load_json_config(json_path)
	metadata = load_yaml_config(yaml_path)

	generate_rirs_from_metadata2(room_parms=room_parms, output_dir=output_dir, metadata=metadata)
