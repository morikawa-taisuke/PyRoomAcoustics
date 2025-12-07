import csv
import json
from pathlib import Path
import argparse
import sys

try:
    from mymodule import const
    from mymodule.rec_utility import get_file_list, load_yaml_config
except ImportError:
    print("=" * 50)
    print("❌ エラー: 'mymodule' が見つかりません。")
    print("リポジトリのルートで 'pip install -e .' を実行しましたか？")
    print("=" * 50)
    sys.exit(1)

def create_audio_paths_csv(config_path):
    """
    指定された設定ファイルに基づき、生成された音声ファイルのパス一覧CSVを作成する。
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
        metadata_dir = split_dir / "metadata"

        if not metadata_dir.exists():
            print(f"️ 警告: メタデータディレクトリが見つかりません: {metadata_dir}。'{split}' をスキップします。")
            continue

        # 3. メタデータファイル（.json）のリストを取得
        try:
            metadata_files = get_file_list(metadata_dir, '.json')
            if not metadata_files:
                print(f"️ 警告: '{split}' にメタデータファイルがありません。スキップします。")
                continue
            print(f"  - '{split}' で {len(metadata_files)} 件のメタデータを発見。")
        except FileNotFoundError:
            print(f"️ 警告: '{split}' のメタデータ検索中にエラーが発生しました。スキップします。")
            continue

        # 4. CSVファイルへの書き込み
        csv_output_path = output_root / f"{split}.csv"
        
        # カラムの順番を指定
        header = ['clean', 'noise_only', 'reverb_only', 'noise_reverb']
        
        rows = []
        for meta_file_path in metadata_files:
            with open(meta_file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 絶対パスを構築
            row_data = {
                key: split_dir / path_str
                for key, path_str in metadata.get('output_files', {}).items()
                if key in header
            }
            
            # headerの順番に従ってパスをリストに格納
            row = [row_data.get(col) for col in header]
            rows.append(row)

        # 5. CSVファイルに保存
        try:
            with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
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
    config_full_path = const.ROOT / args.config
    create_audio_paths_csv(config_full_path)
