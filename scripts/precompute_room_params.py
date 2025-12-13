import pyroomacoustics as pa
import numpy as np
import json
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import itertools
import decimal
import time  # (進捗確認用)
from src.mymodule import const


# Pythonのfloat -> JSONのシリアライズで発生する微小な誤差を防ぐ
class DecimalEncoder(json.JSONEncoder):
	def default(self, o):
		if isinstance(o, decimal.Decimal):
			return str(o)
		return super(DecimalEncoder, self).default(o)


def load_yaml_config(config_path):
	"""YAML設定ファイルを読み込む"""
	with open(config_path, 'r', encoding='utf-8') as f:
		config = yaml.safe_load(f)
	return config


def generate_range(range_config):
	"""
	[min, max, step] の設定から、Decimalのリストを生成する
	"""
	min_val = decimal.Decimal(str(range_config[0]))
	max_val = decimal.Decimal(str(range_config[1]))
	step_val = decimal.Decimal(str(range_config[2]))

	values = []
	current_val = min_val
	while current_val <= max_val:
		values.append(current_val)
		current_val += step_val
	return values


# ===================================================================
# === ▼▼▼ 修正箇所 1: RT60探索関数の追加 ▼▼▼ ===
# ===================================================================
def find_parameters_for_rt60(target_rt60: float,
		room_dim: list,
		fs: int,
		tolerance: float = 0.01,  # 許容誤差 (±10ms)
		max_trials: int = 100,  # 最大試行回数
		step: float = 0.005  # 探索ステップ
):
	"""
	目標のRT60（実測値）に一致する吸収率と反射回数を探索する
	(rec_utility.py の search_reverb_sec のロジックを改良)
	"""

	current_target_rt60 = target_rt60

	for _ in range(max_trials):
		try:
			# 1. 理論値の計算
			e_absorption, max_order = pa.inverse_sabine(rt60=current_target_rt60, room_dim=room_dim)

			# 2. シミュレーションで実測
			room = pa.ShoeBox(room_dim, fs=fs, max_order=max_order, materials=pa.Material(e_absorption))

			# (計測のため、ダミーの音源とマイクを配置)
			room.add_source([room_dim[0] / 2 - 0.1, room_dim[1] / 2, 1.5])
			room.add_microphone([room_dim[0] / 2 + 0.1, room_dim[1] / 2, 1.5])

			room.compute_rir()
			measured_rt60 = np.mean(room.measure_rt60())

			# 3. 誤差の確認
			error = measured_rt60 - target_rt60

			# 4. 許容誤差内であれば成功
			if abs(error) <= tolerance:
				return e_absorption, int(max_order)  # 成功

			# 5. パラメータの更新
			# (実測値が目標より小さい -> もっと響かせる(RT60を上げる)必要あり)
			if error < 0:
				current_target_rt60 += step
			# (実測値が目標より大きい -> もっと吸音する(RT60を下げる)必要あり)
			else:
				# (ただし、補正幅はエラーの大きさに応じて小さくする)
				correction = max(error * 0.1, step)  # 最小でもstep分は動かす
				current_target_rt60 -= correction

		except ValueError:
			# 物理的に計算不可 (例: 部屋が小さすぎ/大きすぎ)
			# このRT60は実現不可能として探索終了
			return None

	# 最大試行回数に達しても収束しなかった
	return None


# ===================================================================

def precompute_parameters(config_path):
	"""
	設定ファイルに基づき、部屋のパラメータを事前計算してJSONに保存する
	"""
	# 1. 設定の読み込み
	config = load_yaml_config(config_path)
	output_dir = const.PARMS_DATA_DIR / config['output_dir']
	fs = config['simulation_constants']['fs']

	# 2. パラメータ範囲の生成
	param_ranges = config['parameter_ranges']
	room_xs = generate_range(param_ranges['room_dimensions']['x_m'])
	room_ys = generate_range(param_ranges['room_dimensions']['y_m'])
	room_zs = generate_range(param_ranges['room_dimensions']['z_m'])
	rt60s_target = generate_range(param_ranges['rt60_sec']['value'])
	print(rt60s_target)

	print(f"計算対象の部屋の組み合わせ総数: {len(room_xs) * len(room_ys) * len(room_zs)}")
	print(f"計算対象のRT60の総数: {len(rt60s_target)}")
	print(f"保存先: {output_dir.absolute()}")

	output_dir.mkdir(parents=True, exist_ok=True)

	room_combinations = list(itertools.product(room_xs, room_ys, room_zs))
	print(room_combinations)

	start_time = time.time()

	# --- 部屋のループ ---
	for (x, y, z) in tqdm(room_combinations, desc="Room Dimensions"):
		print(x,y,z)
		room_dim = [float(x), float(y), float(z)]
		output_data = {}

		# --- RT60のループ ---
		# (tqdmをネストさせ、どのRT60を計算中か表示)
		for rt60_decimal in tqdm(rt60s_target, desc=f"Room {x}x{y}x{z}", leave=False):
			target_rt60_float = float(rt60_decimal)

			# 探索関数を呼び出す
			result = find_parameters_for_rt60(target_rt60=target_rt60_float, room_dim=room_dim, fs=fs)

			# resultが None (計算不可 or 収束失敗) でない場合のみ、
			# JSONにデータを書き込む
			if result is not None:
				e_absorption, max_order = result

				# (rt60_keyは "0.50s" のように小数点以下2桁で統一)
				rt60_key = f"{rt60_decimal:.2f}s"
				output_data[rt60_key] = {
					"absorption": e_absorption,
					"max_order": max_order
				}
			else:
				print("計算ができませんでした．")

		# 6. 部屋ごと（Xm_Ym_Zm.json）にJSONファイルとして保存
		# (ただし、中身が空でない場合のみ)
		if output_data:  # 1つでも有効なRT60があれば保存
			filename = f"{int(x*100)}cm_{int(y*100)}cm_{int(z*100)}cm.json"
			filepath = output_dir / filename

			with open(filepath, 'w', encoding='utf-8') as f:
				json.dump(output_data, f, indent=4, cls=DecimalEncoder)

	end_time = time.time()
	print(f"\n🎉 事前計算が完了しました。(合計時間: {end_time - start_time:.2f} 秒)")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="部屋の音響パラメータを事前計算し、JSONファイルとして保存します。")
	parser.add_argument(
		'--config',
		type=str,
		# required=True,
		default="./../config/sample/precompute_params.yml",
		help="事前計算の範囲を定義したYAMLファイルのパス (例: configs/precompute_params.yml)"
	)
	args = parser.parse_args()

	precompute_parameters(args.config)
