"""
【役割】
プロジェクト全体で参照されるパスや環境設定などの定数を定義するモジュール
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# .env ファイルを読み込む
load_dotenv()

# .env から大元のデータディレクトリのパスを取得
raw_path = os.getenv("SOUND_DATA_DIR")
if raw_path is None:
    raise ValueError(
        "【エラー】パスが設定されていません！\n"
        "リポジトリ直下に '.env' ファイルを作成し、SOUND_DATA_DIR を指定してください。\n"
        "例: SOUND_DATA_DIR=D:/sound_data/"
    )

SOUND_DATA_DIR = Path(raw_path)

SAMPLE_DATA_DIR = SOUND_DATA_DIR / 'sample_data'
SPEECH_DATA_DIR = SAMPLE_DATA_DIR / 'speaker'
NOISE_DATA_DIR = SAMPLE_DATA_DIR / 'noise'

MIX_DATA_DIR = SOUND_DATA_DIR / 'mix_data'
PARMS_DATA_DIR = SOUND_DATA_DIR / 'preconpute_params'
RIR_DIR = SOUND_DATA_DIR / 'RIR'
