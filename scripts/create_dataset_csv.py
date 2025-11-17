# scripts/create_dataset_csv.py

import argparse
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import sys

def create_dataset_csvs(dataset_root: Path):
    """
    データセットディレクトリ内のすべてのmetadata.jsonを解析し、
    'train.csv', 'test.csv', 'val.csv' のようにスプリットごとのCSVファイルを作成する。

    Args:
        dataset_root (Path): new_signal_noise.pyで生成されたデータセットのルートディレクトリ。
        output_filename (str): 出力するCSVファイル名。
    """
    if not dataset_root.is_dir():
        print(f"❌ エラー: データセットディレクトリが見つかりません: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    # 1. すべての metadata.json を検索
    json_files = sorted(list(dataset_root.rglob("metadata.json")))
    if not json_files:
        print(f"❌ エラー: {dataset_root} 内に 'metadata.json' が見つかりませんでした。", file=sys.stderr)
        print("    'new_signal_noise.py' を実行して、先にデータセットを生成してください。", file=sys.stderr)
        sys.exit(1)

    print(f"✅ {len(json_files)} 個の 'metadata.json' ファイルを検出しました。CSVファイルを作成します...")

    # 2. 各JSONを解析してデータをリストに格納
    all_records = []
    for json_path in tqdm(json_files, desc="Processing metadata"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                room_meta = json.load(f)

            room_output_dir = json_path.parent
            split = room_output_dir.parent.name # train, test, val

            # 部屋レベルの情報を抽出
            room_info = {
                "split": split,
                "room_id": room_meta.get("room_id"),
                "measured_rt60": room_meta.get("measured_rt60"),
                "c50": room_meta.get("c50"),
                "d50": room_meta.get("d50"),
                "x": room_meta.get("room_dim", [None,None,None])[0],
                "y": room_meta.get("room_dim", [None,None,None])[1],
                "z": room_meta.get("room_dim", [None,None,None])[2],
                "ch": room_meta.get("mic_config", {}).get("array", {}).get("channels"),
                "mic_shape": room_meta.get("mic_config", {}).get("array", {}).get("shape"),
            }

            # ファイルレベルの情報をループ処理
            for file_info in room_meta.get("files", []):
                record = room_info.copy()
                record.update({
                    "snr_db": file_info.get("snr_db"),
                    "clean_source_file": file_info.get("clean_source_file"),
                    "noise_source_file": file_info.get("noise_source_file"),
                })

                # 各音声ファイルの相対パスを構築
                base_name = file_info["filename_base"]
                record["mixture_path"] = str((room_output_dir / "mixture" / f"{base_name}_mix.wav").relative_to(dataset_root))
                record["clean_speech_path"] = str((room_output_dir / "clean_speech" / f"{base_name}_clean.wav").relative_to(dataset_root))
                record["reverb_speech_path"] = str((room_output_dir / "reverb_speech" / f"{base_name}_reverb.wav").relative_to(dataset_root))
                record["noise_only_path"] = str((room_output_dir / "noise_only" / f"{base_name}_noise.wav").relative_to(dataset_root))

                all_records.append(record)

        except Exception as e:
            tqdm.write(f"⚠️ 警告: {json_path} の処理中にエラーが発生しました: {e}", file=sys.stderr)

    if not all_records:
        print("❌ エラー: 有効なレコードが1件も見つかりませんでした。", file=sys.stderr)
        sys.exit(1)

    # 3. pandas DataFrame に変換
    df = pd.DataFrame(all_records)

    # 列の順序を整える
    ordered_columns = [
        "split", "room_id",
        "mixture_path", "clean_speech_path", "reverb_speech_path", "noise_only_path",
        "snr_db", "measured_rt60", "c50", "d50",
        "x", "y", "z",
        "ch", "mic_shape",
        "clean_source_file", "noise_source_file"
    ]
    # 存在しない列があった場合のエラーを防ぐ
    final_columns = [col for col in ordered_columns if col in df.columns]
    df = df[final_columns]

    # 4. スプリットごとにグループ化して、個別のCSVファイルとして保存
    if "split" not in df.columns:
        print("❌ エラー: DataFrameに 'split' 列が見つかりません。CSVを分割できません。", file=sys.stderr)
        sys.exit(1)

    print("\nCSVファイルに分割して保存します...")
    num_splits = 0
    for split_name, split_df in df.groupby("split"):
        output_path = dataset_root / f"{split_name}.csv"
        # 'split' 列はファイル名でわかるので、CSVからは削除する
        split_df.drop(columns="split").to_csv(output_path, index=False, encoding='utf-8')
        print(f"  ✅ {split_name:<5}: {len(split_df):>5} 件 -> {output_path.resolve()}")
        num_splits += 1

    print(f"\n🎉 完了！ {num_splits} 個のCSVファイルを作成しました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="データセットディレクトリを解析し、全ファイルの情報をまとめたCSVを作成します。"
    )
    parser.add_argument(
        "dataset_root",
        type=str,
        help="`new_signal_noise.py`で生成されたデータセットのルートディレクトリパス。"
    )
    args = parser.parse_args()

    create_dataset_csvs(Path(args.dataset_root))