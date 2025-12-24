import csv
from pathlib import Path


def create_sample_csv(dir_path, output_csv='sample.csv', use_absolute_path=True):
	"""
	指定されたディレクトリ内のサブディレクトリとファイルからsample.csvを作成します。

	Args:
		dir_path (str): 処理対象のディレクトリの絶対パス
		output_csv (str): 出力するCSVファイル名（デフォルト: 'sample.csv'）
		use_absolute_path (bool): Trueの場合、絶対パスを記録。Falseの場合、ファイル名のみ記録
	"""
	# Pathオブジェクトに変換
	base_dir = Path(dir_path)

	# サブディレクトリを取得
	sub_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
	sub_dirs.sort()  # サブディレクトリを名前順にソート

	if not sub_dirs:
		print("サブディレクトリが見つかりません。")
		return

	# 各サブディレクトリ内のファイルリストを取得
	files_in_dirs = {}
	max_files = 0

	for sub_dir in sub_dirs:
		# ファイルリストを取得し、ソート
		if use_absolute_path:
			files = [str(f.absolute()) for f in sub_dir.iterdir() if f.is_file()]
		else:
			files = [f.name for f in sub_dir.iterdir() if f.is_file()]
		files.sort()
		files_in_dirs[sub_dir.name] = files
		max_files = max(max_files, len(files))

	# CSVファイルを作成
	with open(output_csv, 'w', newline='', encoding='utf-8') as f:
		writer = csv.writer(f)

		# ヘッダー行（サブディレクトリ名）を書き込み
		writer.writerow([d.name for d in sub_dirs])

		# ファイルパスを書き込み
		for i in range(max_files):
			row = []
			for sub_dir in sub_dirs:
				files = files_in_dirs[sub_dir.name]
				row.append(files[i] if i < len(files) else '')
			writer.writerow(row)

	print(f"CSVファイルを作成しました: {output_csv}")
	print(f"処理したサブディレクトリ数: {len(sub_dirs)}")
	print(f"最大ファイル数: {max_files}")
	print(f"パス形式: {'絶対パス' if use_absolute_path else 'ファイル名のみ'}")


if __name__ == '__main__':
	from src.mymodule import const

	sub_dir = "val"
	dir_path = f"{const.MIX_DATA_DIR}/DEMAND_hoth_10dB_500msec/{sub_dir}"
	output = f"{const.MIX_DATA_DIR}/DEMAND_hoth_10dB_500msec/{sub_dir}.csv"
	create_sample_csv(dir_path, output)