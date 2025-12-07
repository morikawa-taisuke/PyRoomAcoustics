import argparse
import csv
import sys
from collections import defaultdict

try:
	from mymodule import const
	from mymodule.rec_utility import load_yaml_config
except ImportError:
	print("=" * 50)
	print("❌ エラー: 'mymodule' が見つかりません。")
	print("リポジトリのルートで 'pip install -e .' を実行しましたか？")
	print("=" * 50)
	sys.exit(1)


def create_audio_paths_csv(config_path):
	"""
	指定された設定ファイルに基づき、生成された音声ファイルのパス一覧CSVを作成する。
	このスクリプトは、サブディレクトリ内の実際のファイル構成に基づいてCSVを生成し、
	metadata.jsonファイルには依存しません。
	"""
	# 1. 設定ファイルの読み込み
	try:
		config = load_yaml_config(config_path)
		output_dir_name = config['path']['output_dir_name']
		wave_types = config['path']['wave_type_list']
	except Exception as e:
		print(f"❌ エラー: 設定ファイル {config_path} の読み込みに失敗: {e}", file=sys.stderr)
		sys.exit(1)

	output_root = const.MIX_DATA_DIR / output_dir_name
	print(f"✅ データセットのルートディレクトリ: {output_root}")

	# 2. 各スプリット（train, test, valなど）ごとに処理
	for split in wave_types:
		split_dir = output_root / split
		print(f"\n--- '{split}' の処理を開始 ---")

		# 3. 各サブディレクトリの音声ファイルを取得
		header = ['clean', 'noise_only', 'reverb_only', 'noise_reverb']
		path_map = defaultdict(dict)

		all_dirs_exist = True
		for dir_type in header:
			target_dir = split_dir / dir_type
			if not target_dir.exists():
				print(f"  - 警告: ディレクトリが見つかりません: {target_dir}。'{split}' をスキップします。")
				all_dirs_exist = False
				break

			# ファイル名（拡張子なし）をキー、フルパスを値とする辞書を作成
			for file_path in target_dir.rglob('*.wav'):
				path_map[file_path.stem][dir_type] = file_path

		if not all_dirs_exist:
			continue

		# 4. 4つのディレクトリすべてに存在するファイルのパスを収集
		rows = []
		for file_stem, path_dict in path_map.items():
			# headerのすべてのキーが辞書に存在するかチェック
			if all(h in path_dict for h in header):
				# headerの順番に従ってパスをリストに格納
				row = [path_dict[h] for h in header]
				rows.append(row)

		# 5. 結果のサマリーとCSVファイル保存
		found_count = len(rows)
		total_files_in_map = len(path_map)
		skipped_count = total_files_in_map - found_count

		print(f"  - {found_count} 件の有効な音声セットを発見。")
		if skipped_count > 0:
			print(f"  - ⚠️ {skipped_count} 件の音声セットをスキップしました (ファイル欠損のため)。")

		if found_count == 0:
			print(f"  - CSVファイルは生成されませんでした（有効なセットが0件）。")
			continue

		csv_output_path = output_root / f"{split}.csv"
		try:
			with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
				writer = csv.writer(f)
				writer.writerow(header)
				# ファイル名でソートして、毎回同じ順序で出力されるようにする
				rows.sort(key=lambda x: x[0].name)
				writer.writerows(rows)
			print(f"  - ✅ CSVファイルを保存しました: {csv_output_path}")
		except IOError as e:
			print(f"❌ エラー: CSVファイル '{csv_output_path}' の書き込みに失敗しました: {e}", file=sys.stderr)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="音声データセットのファイルパスをまとめたCSVを作成します。"
	)
	parser.add_argument(
		'--config',
		type=str,
		default="config/sample/sample.yml",
		help="データセット生成時に使用したYAMLファイルのパス"
	)
	args = parser.parse_args()

	# configパスをプロジェクトルートからの相対パスとして解決
	create_audio_paths_csv(args.config)
