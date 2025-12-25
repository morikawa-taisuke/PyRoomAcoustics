import os
import random
import math
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# rec2.py から必要な関数をインポート
from rec2 import recoding2
from mymodule import const, my_func
import pyroomacoustics as pa


def load_room_params(json_path):
	"""JSONファイルから部屋パラメータを読み込む"""
	with open(json_path, 'r') as f:
		return json.load(f)


def get_closest_params(room_params, target_rt60):
	"""目標RT60に最も近いパラメータを取得する"""
	# キー（"0.30s"など）から数値を取得し、差分が最小のものを選ぶ
	closest_key = min(room_params.keys(), key=lambda k: abs(float(k.replace('s', '')) - target_rt60))
	return room_params[closest_key], float(closest_key.replace('s', ''))


def generate_random_dataset(num_samples=1000):
	"""
	ランダムな条件下でデータセットを生成するスクリプト
	(事前計算された部屋パラメータを使用 + マルチプロセス並列化)
	"""

	# --- 設定項目（ここを調整してください） ---
	speech_type = "VCTK"  # 目的音声のタイプ
	noise_type = "DEMAND"  # 雑音信号のタイプ（フォルダ名など）

	subdir_list = ["train", "val", "test"]

	# 事前計算ファイルのディレクトリ (プロジェクトルートからの相対パス等を想定)
	# 必要に応じて絶対パスに変更してください
	precomputed_dir = Path("C:/Users/kataoka-lab/Desktop/sound_data/precompute_params/precomputed_params_2")


	# 並列処理のワーカー数 (Noneの場合はCPUコア数)
	max_workers = None

	for subdir in subdir_list:
		# パラメータの範囲設定
		snr_range = (-5, 15)  # SNRの候補 [dB]
		reverb_range = (0.3, 1.0)  # 残響時間の範囲 [sec] (min, max)
		room_dim = np.array([5.0, 5.0, 5.0])  # 部屋のサイズ

		# マイク設定
		ch = 1  # マイク数
		mic_distance = 10  # マイク間隔 [cm]

		# パス設定
		speech_dir = f"{const.SAMPLE_DATA_DIR}/speech/{speech_type}/{subdir}"
		noise_dir = f"{const.SAMPLE_DATA_DIR}/noise/{noise_type}"  # 雑音が複数入っているフォルダ
		output_base_dir = Path(f"{const.MIX_DATA_DIR}/Random_Dataset_{speech_type}_{noise_type}_{ch}ch/{subdir}")

		# 1. 事前計算データの読み込み
		room_dim_int = [int(d)*100 for d in room_dim]
		json_filename = f"{room_dim_int[0]}cm_{room_dim_int[1]}cm_{room_dim_int[2]}cm.json"
		json_path = precomputed_dir / json_filename

		if not json_path.exists():
			print(f"Error: 事前計算ファイルが見つかりません: {json_path}")
			print("scripts/precompute_room_params.py を実行してパラメータを生成してください。")
			return

		print(f"Loading room params from: {json_path}")
		room_params_data = load_room_params(json_path)

		# 2. データのリストを取得
		speech_files = my_func.get_file_list(speech_dir)
		noise_files = my_func.get_file_list(noise_dir)  # フォルダ内の全wav取得を想定

		if not speech_files or not noise_files:
			print(f"Error: 音声ファイルまたは雑音ファイルが見つかりません。\nSpeech: {speech_dir}\nNoise: {noise_dir}")
			continue  # 次のsubdirへ

		print(f"Total samples to generate for {subdir}: {len(speech_files)}")

		# 3. タスクリストの作成 (パラメータ決定フェーズ)
		tasks = []
		print("Preparing tasks...")
		for target_file in tqdm(speech_files):
			# --- パラメータのランダム決定 ---
			noise_file = random.choice(noise_files)
			snr = random.randint(snr_range[0], snr_range[1])
			# 目標RT60をランダムに決定
			target_rt60 = random.uniform(reverb_range[0], reverb_range[1])

			# 雑音の到来方向（方位角）を0〜360度でランダム決定
			angle_deg = random.randint(0, 359)
			angle_rad = math.radians(angle_deg)
			angle_name = f"{angle_deg:03}deg"

			# --- 残響パラメータの取得 (JSONから) ---
			params = room_params_data[f"{target_rt60:.2f}"]

			e_absorption = params['absorption']
			max_order = params['max_order']
			reverb_par = (e_absorption, max_order)

			# 出力ディレクトリの設定
			current_out_dir = output_base_dir
			my_func.exists_dir(current_out_dir)

			# recoding2 に渡す引数を辞書にまとめる
			task_kwargs = {
				"wave_files": [target_file, noise_file],
				"out_dir": current_out_dir,
				"snr": snr,
				"reverb_sec": target_rt60,  # 実際に使用するRT60を使用
				"reverb_par": reverb_par,
				"channel": ch,
				"distance": mic_distance,
				"is_split": False,
				"angle": angle_rad,
				"angle_name": angle_name
			}
			tasks.append(task_kwargs)

		# 4. 並列実行フェーズ
		print(f"Executing {len(tasks)} tasks in parallel...")
		with ProcessPoolExecutor(max_workers=max_workers) as executor:
			# タスクを投入
			futures = [executor.submit(recoding2, **kwargs) for kwargs in tasks]

			# 進捗表示 (完了したものから順次)
			for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {subdir}"):
				try:
					future.result()  # エラーがあればここで例外が発生
				except Exception as e:
					print(f"Task failed with error: {e}")

if __name__ == "__main__":
	# シード値を固定したい場合は以下を有効化
	random.seed(100)
	np.random.seed(100)

	generate_random_dataset()  # まずは少ない数でテスト推奨
