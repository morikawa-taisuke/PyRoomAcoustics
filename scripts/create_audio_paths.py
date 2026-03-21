import csv
from pathlib import Path
import argparse
import sys
import concurrent.futures
from tqdm import tqdm

try:
    from core import const
    from core.rec_utility import load_yaml_config
except ImportError:
    print("=" * 50)
    print("❌ エラー: 'mymodule' が見つかりません。")
    print("リポジトリのルートで 'pip install -e .' を実行しましたか？")
    print("=" * 50)
    sys.exit(1)

def get_file_prefix(file_path):
    """
    ファイルパスからプレフィックス（話者番号_発話番号）を抽出する。
    例: p225_001_mic1.wav -> p225_001
    """
    try:
        parts = file_path.stem.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
    except Exception:
        pass
    return None

def scan_directory(directory):
    """
    ディレクトリ内の.wavファイルをスキャンし、プレフィックスをキー、パスを値とする辞書を作成する。
    """
    if not directory.exists():
        return None, directory

    file_map = {}
    # rglobで再帰的に探す
    files = list(directory.rglob('*.wav'))
    
    duplicate_prefixes = set()

    for f in files:
        prefix = get_file_prefix(f)
        if prefix:
            if prefix in file_map:
                # 重複がある場合、このプレフィックスは無効とするために記録しておく
                duplicate_prefixes.add(prefix)
            else:
                file_map[prefix] = f
    
    # 重複があったプレフィックスは削除する（曖昧さ回避のため）
    for p in duplicate_prefixes:
        if p in file_map:
            del file_map[p]
            
    return file_map, directory

def process_split(split, output_root):
    """
    1つのsplit（train, test, valなど）に対する処理を行う。
    """
    split_dir = output_root / split
    print(f"\n--- '{split}' の処理を開始 ---")

    # 必要なディレクトリ
    target_dirs = {
        'clean': split_dir / 'clean',
        'noise_only': split_dir / 'noise_only',
        'reverb_only': split_dir / 'reverb_only',
        'noise_reverb': split_dir / 'noise_reverb'
    }

    # 並列でディレクトリをスキャンして辞書を作成
    scanned_data = {}
    
    # ディレクトリごとのスキャンを並列化
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_name = {
            executor.submit(scan_directory, path): name 
            for name, path in target_dirs.items()
        }
        
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result_map, dir_path = future.result()
                if result_map is None:
                    print(f"  - 警告: ディレクトリが見つかりません: {dir_path}")
                    scanned_data[name] = {}
                else:
                    scanned_data[name] = result_map
                    print(f"  - '{name}' スキャン完了: {len(result_map)} ファイル")
            except Exception as exc:
                print(f"  - エラー: {name} のスキャン中に例外が発生: {exc}")
                scanned_data[name] = {}

    # clean を基準にマッチング
    clean_map = scanned_data.get('clean', {})
    if not clean_map:
        print(f"  - 警告: clean データがありません。スキップします。")
        return

    header = ['clean', 'noise_only', 'reverb_only', 'noise_reverb']
    rows = []
    
    # マッチング処理 (メモリ上の辞書引きなので高速)
    # tqdmで進捗表示
    print("  - マッチング処理中...")
    for prefix, clean_path in tqdm(clean_map.items()):
        row_dict = {'clean': clean_path}
        found_all = True
        
        for other_type in ['noise_only', 'reverb_only', 'noise_reverb']:
            other_map = scanned_data.get(other_type, {})
            if prefix in other_map:
                row_dict[other_type] = other_map[prefix]
            else:
                found_all = False
                break
        
        if found_all:
            rows.append([row_dict[h] for h in header])

    # 結果出力
    found_count = len(rows)
    print(f"  - {found_count} 件の有効な音声セットを発見。")
    
    if found_count > 0:
        csv_output_path = output_root / f"{split}.csv"
        try:
            with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                # ファイル名でソート
                rows.sort(key=lambda x: x[0].name)
                writer.writerows(rows)
            print(f"  - ✅ CSVファイルを保存しました: {csv_output_path}")
        except IOError as e:
            print(f"❌ エラー: CSVファイル書き込み失敗: {e}", file=sys.stderr)
    else:
        print("  - 有効なセットが見つからなかったため、CSVは生成されませんでした。")

def create_audio_paths_csv_fast(config_path):
    """
    並列処理を用いて高速にCSVを作成するメイン関数
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
    print("🚀 並列処理による高速化モードで実行中...")

    # 2. 各スプリットごとに処理
    for split in wave_types:
        process_split(split, output_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="音声データセットのファイルパスをまとめたCSVを高速に作成します（並列処理版）。"
    )
    parser.add_argument(
        '--config',
        type=str,
        default="config/sample/sample.yml",
        help="データセット生成時に使用したYAMLファイルのパス"
    )
    args = parser.parse_args()

    create_audio_paths_csv_fast(args.config)
