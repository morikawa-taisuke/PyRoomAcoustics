import csv
from pathlib import Path
import argparse
import sys

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
    'clean'ディレクトリを基準とし、「話者番号_発話番号」をプレフィックスとして
    対応するファイルを他のディレクトリから検索する。
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

        # 3. 基準となるcleanディレクトリのファイルリストを取得
        clean_dir = split_dir / 'clean'
        if not clean_dir.exists():
            print(f"  - 警告: 基準ディレクトリが見つかりません: {clean_dir}。'{split}' をスキップします。")
            continue

        clean_files = list(clean_dir.rglob('*.wav'))
        if not clean_files:
            print(f"  - 警告: '{clean_dir}' に基準となる音声ファイルがありません。")
            continue
        
        print(f"  - '{clean_dir.name}' で {len(clean_files)} 件の基準ファイルを発見。")

        # 4. 対応するファイルの存在を確認し、CSV用の行を作成
        header = ['clean', 'noise_only', 'reverb_only', 'noise_reverb']
        rows = []
        
        # 他のディレクトリのパスをあらかじめ定義
        other_dirs = {
            'noise_only': split_dir / 'noise_only',
            'reverb_only': split_dir / 'reverb_only',
            'noise_reverb': split_dir / 'noise_reverb'
        }

        for clean_path in clean_files:
            try:
                # ファイル名から「話者番号_発話番号」を抽出 (例: p225_001)
                parts = clean_path.stem.split('_')
                base_name = f"{parts[0]}_{parts[1]}"
            except IndexError:
                print(f"    - 警告: ファイル名が命名規則に合いません。スキップします: {clean_path.name}")
                continue

            found_all = True
            row_dict = {'clean': clean_path}

            for dir_type, target_dir in other_dirs.items():
                if not target_dir.exists():
                    found_all = False
                    break
                
                # base_nameで始まるファイルを検索
                matches = list(target_dir.glob(f'{base_name}*.wav'))
                
                if len(matches) == 1:
                    row_dict[dir_type] = matches[0]
                else:
                    # 該当ファイルが0個または2個以上見つかった場合はセット不成立
                    found_all = False
                    if len(matches) > 1:
                        print(f"    - 警告: {target_dir} に {base_name} で始まるファイルが複数あります。スキップします。")
                    break
            
            if found_all:
                # headerの順番に従ってパスをリストに格納
                row = [row_dict[h] for h in header]
                rows.append(row)

        # 5. 結果のサマリーとCSVファイル保存
        found_count = len(rows)
        total_clean_files = len(clean_files)
        skipped_count = total_clean_files - found_count
        
        print(f"  - {found_count} 件の有効な音声セットを発見。")
        if skipped_count > 0:
            print(f"  - ⚠️ {skipped_count} 件の音声セットをスキップしました (対応ファイル欠損または曖昧さのため)。")

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
    # config_full_path = const.ROOT / args.config
    create_audio_paths_csv(args.config)
