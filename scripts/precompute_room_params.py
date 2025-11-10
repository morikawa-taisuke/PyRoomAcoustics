import argparse
import decimal
import itertools
import json
from pathlib import Path

import pyroomacoustics as pa
import yaml  # PyYAMLライブラリ
from tqdm import tqdm


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
	[min, max, step] の設定から、np.arangeと同様のリストを生成する
	(Decimalを使って浮動小数点数の誤差を回避する)
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


def precompute_parameters(config_path):
	"""
	設定ファイルに基づき、部屋のパラメータを事前計算してJSONに保存する
	"""
	# 1. 設定の読み込み
	config = load_yaml_config(config_path)
	output_dir = Path(config['output_dir'])

	# 2. パラメータ範囲の生成
	param_ranges = config['parameter_ranges']
	room_xs = generate_range(param_ranges['room_dimensions']['x_m'])
	room_ys = generate_range(param_ranges['room_dimensions']['y_m'])
	room_zs = generate_range(param_ranges['room_dimensions']['z_m'])
	rt60s = generate_range(param_ranges['rt60_sec']['value'])

	print(f"計算対象の部屋の組み合わせ総数: {len(room_xs) * len(room_ys) * len(room_zs)}")
	print(f"計算対象のRT60の総数: {len(rt60s)}")
	print(f"保存先: {output_dir.absolute()}")

	# 3. 保存先ディレクトリの作成
	output_dir.mkdir(parents=True, exist_ok=True)

	# 4. 全組み合わせについてループ処理
	# tqdmを使って進捗を表示
	room_combinations = list(itertools.product(room_xs, room_ys, room_zs))

	for (x, y, z) in tqdm(room_combinations, desc="Room Dimensions"):
		room_dim = [float(x), float(y), float(z)]
		output_data = {}  # この部屋の全RT60の結果を格納する辞書

		# 5. RT60ごとにパラメータを計算
		for rt60_decimal in rt60s:
			rt60_float = float(rt60_decimal)

			try:
				# pyroomacoustics の中核関数を呼び出す
				e_absorption, max_order = pa.inverse_sabine(
					rt60=rt60_float,
					room_dim=room_dim
				)

				# JSONに保存するデータを整形
				# (rt60_keyは "0.50s" のように小数点以下2桁で統一)
				rt60_key = f"{rt60_decimal:.2f}s"
				output_data[rt60_key] = {
					"absorption": e_absorption,
					"max_order": int(max_order)  # max_orderは整数
				}

			except ValueError as e:
				# (例: 部屋が小さすぎてRT60が達成不可能な場合)
				tqdm.write(f"警告: {room_dim} @ {rt60_float}s で計算エラー: {e}")

		# 6. 部屋ごと（Xm_Ym_Zm.json）にJSONファイルとして保存
		filename = f"{x:.1f}m_{y:.1f}m_{z:.1f}m.json"
		filepath = output_dir / filename

		with open(filepath, 'w', encoding='utf-8') as f:
			# DecimalEncoderを使用してJSONを保存
			json.dump(output_data, f, indent=4, cls=DecimalEncoder)

	print("\n🎉 事前計算が完了しました。")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="部屋の音響パラメータを事前計算し、JSONファイルとして保存します。"
	)
	parser.add_argument(
		'--config',
		type=str,
		# required=True,
		default='./../config/sample/precompute_params.yml',
		help="事前計算の範囲を定義したYAMLファイルのパス (例: configs/sample/precompute_params.yml)"
	)
	args = parser.parse_args()

	precompute_parameters(args.config)
