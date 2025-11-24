import pandas as pd
import argparse
import json
from pathlib import Path
from tqdm import tqdm


def parse_metadata(metadata_path: Path, output_base_dir: Path) -> list[dict]:
	"""
	単一の metadata.json を読み込み、
	フラットな辞書のリスト（CSVの各行に対応）に変換する
	"""

	with open(metadata_path, 'r', encoding='utf-8') as f:
		metadata = json.load(f)

	# 1. 部屋（Room）レベルの共通情報を抽出
	room_id_str = metadata_path.parent.name  # (例: "room_0000")
	room_info = {
		"room_id": room_id_str,
		"rt60": metadata.get("measured_rt60", metadata.get("target_rt60")),
		# (注: 座標情報は複雑なので、ここでは省略し、必要なら room_id で元JSONを参照)
		# "mic_coords": json.dumps(metadata.get("mic_config", {}).get("position_coords")),
		# "speech_coords": json.dumps(metadata.get("speech_source_config", {}).get("params")),
	}

	# 2. ファイル（File）レベルの情報を展開
	csv_rows = []
	for file_info in metadata.get("files", []):
		base_name = file_info["filename_base"]

		# 出力パスを構築
		# (例: .../dataset_hybrid_v1/train/room_0000/mixture/p225_..._mix.wav)
		# ※ metadata.json から見た相対パスではなく、CSVの利用者が使いやすいよう
		#   dataset_hybrid_v1 からの相対パス、あるいは絶対パスを構築する

		# (metadata_path = .../room_0000/metadata.json)
		room_dir = metadata_path.parent

		# (output_base_dir = .../dataset_hybrid_v1/train)
		# (room_dir.relative_to(output_base_dir) = room_0000)

		# 3. 各ファイルの絶対パス（またはデータセットルートからの相対パス）を構築
		# (YAMLの output_files 設定に応じて、存在しないパスもある)
		row = {
			"mixture_path": room_dir / "mixture" / f"{base_name}_mix.wav",
			"reverb_speech_path": room_dir / "reverb_speech" / f"{base_name}_reverb.wav",
			"clean_speech_path": room_dir / "clean_speech" / f"{base_name}_clean.wav",
			"reverberant_noise_path": room_dir / "reverb_noise" / f"{base_name}_noise.wav",

			# (ご要望のカラム名に合わせて情報を追加)
			"clean_source_file": file_info["clean_source_file"],
			"snr": file_info["snr_db"],
		}

		# 部屋情報とファイル情報を結合
		row.update(room_info)
		csv_rows.append(row)

	return csv_rows


def create_dataset_csv(dataset_root_dir: str, splits: list[str]):
	"""
	データセットのルートディレクトリをスキャンし、
	metadata.json から train.csv, val.csv, test.csv を生成する
	"""
	root_path = Path(dataset_root_dir)
	print(f"スキャン対象: {root_path.absolute()}")

	for split in splits:
		print(f"\n--- \"{split}\" の処理を開始 ---")
		split_dir = root_path / split

		if not split_dir.exists():
			print(f"警告: {split_dir} が見つかりません。スキップします。")
			continue

		# 1. すべての metadata.json を検索
		metadata_files = list(tqdm(
			split_dir.rglob("metadata.json"),
			desc=f"Finding metadata in {split}"
		))

		if not metadata_files:
			print(f"警告: {split_dir} で metadata.json が見つかりません。")
			continue

		# 2. 全JSONをパースして、行データのリストにまとめる
		all_rows = []
		for meta_path in tqdm(metadata_files, desc=f"Parsing metadata for {split}"):
			# (output_base_dir = split_dir (例: .../dataset_hybrid_v1/train) )
			rows = parse_metadata(meta_path, split_dir)
			all_rows.extend(rows)

		# 3. pandas DataFrame に変換
		df = pd.DataFrame(all_rows)

		# 4. カラムの順序をご要望に合わせて調整
		# (存在しないカラムは無視される)
		columns_order = [
			"clean_source_file",  # 話者のファイル名 (元ファイル)
			"mixture_path",  # noise_reverb
			"reverberant_noise_path",  # noise_only (※B案拡張版)
			"reverb_speech_path",  # reverb_only
			"clean_speech_path",  # clean (※B案拡張版)
			"room_id",  # room番号
			"snr",
			"rt60",
			# (座標情報は複雑なので除外)
		]

		# 存在するカラムのみで順序を再定義
		final_columns = [col for col in columns_order if col in df.columns]
		df = df[final_columns]

		# 5. CSVファイルとして保存
		output_csv_path = root_path / f"{split}.csv"
		df.to_csv(output_csv_path, index=False, encoding='utf-8')
		print(f"✅ {output_csv_path} に {len(df)} 行のCSVを保存しました。")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="データセットディレクトリ内の全 metadata.json をスキャンし、"
		            "機械学習用のフラットなCSV (train.csv, val.csv, test.csv) を生成します。"
	)
	parser.add_argument(
		'--dataset_dir',
		type=str,
		required=True,
		help="`new_signal_noise.py` で生成されたデータセットのルートディレクトリ"
		     "(例: ./sound_data/mix_data/dataset_hybrid_v1)"
	)
	parser.add_argument(
		'--splits',
		nargs='+',  # 1つ以上の引数をリストとして受け取る
		default=["train", "val", "test"],
		help="処理対象のスプリット (デフォルト: train val test)"
	)
	args = parser.parse_args()

	create_dataset_csv(args.dataset_dir, args.splits)