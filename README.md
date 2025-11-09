# PyRoomAcoustics

## 概要

`pyroomacoustics` を用いた音響シミュレーションのためのライブラリです。
主に、音源強調用の音声データセットを生成するために使用します。

## 使い方 (データセット生成)

このリポジトリは、音響シミュレーションによる音声データセットを生成するために、2つの主要なワークフローを提供します。

  * **A案: ステップ分離ワークフロー** (マルチチャンネル対応)
      * RIRの生成と、音声との畳み込みを別々のステップで実行します。
      * マルチチャンネルのRIRや、RIRを固定して音声だけ差し替える場合に適しています。
  * **B案: 一括実行ワークフロー** (メタデータ管理)
      * 部屋（ドメイン）のシミュレーションから、RIR計算、音声との畳み込み、メタデータ（RT60, C50など）の保存までを一括で実行します。
      * シングルチャンネル用で、RIRの物理特性を含めたメタデータ管理を行いたい場合に適しています。

-----

### A案: ステップ分離ワークフロー (マルチチャンネル対応)

#### ステップ1: RIRの生成

`Generate_IR.py` を使用して、マルチチャンネルRIR（WAVファイル）を生成します。

**実行例:**

`Generate_IR.py` の `if __name__ == "__main__":` 内のパラメータ（`channel_list`, `distance_list` など）を直接編集して実行します。

```bash
python Generate_IR.py
```

  * `const.MIX_DATA_DIR` 内の `IR` ディレクトリなどに、`(サンプル数, チャンネル数)` 形式のマルチチャンネルWAVファイルとしてRIRが保存されます。

#### ステップ2: 音声との畳み込み

`process_audio2.py` と設定ファイル（JSON）を使用して、音声とRIRを畳み込みます。

**1. 設定ファイルの準備 (例: `config_A.json`)**

  * `base_paths` で各データ（音声、雑音、RIR、出力先）のルートパスを指定します。
  * `splits` で処理対象のサブディレクトリ（例: "test", "train"）を指定します。
  * `tasks` で生成したいデータ（`reverbe_only` など）の `enabled` を `true` にし、使用するRIRのパスを指定します。

<!-- end list -->

```json
{
  "base_paths": {
    "speech_data_root": "./sound_data/sample_data/speech/DEMAND/clean",
    "noise_data_root": "./sound_data/sample_data/noise",
    "ir_data_root": "./sound_data/sample_data/IR/4ch_10cm_liner",
    "output_data_root": "./sound_data/mix_data/DEMAND_hoth_4ch_0dB_500msec"
  },
  "splits": ["test", "train"],
  "tasks": {
    "clean": {
      "enabled": false,
      "speech_ir_path": "clean/speech/050sec.wav"
    },
    "noise_only": {
      "enabled": true,
      "noise_type": "hoth",
      "speech_ir_path": "clean/speech/050sec.wav",
      "noise_ir_path": "clean/noise/050sec_000dig.wav"
    },
    "reverbe_only": {
      "enabled": true,
      "speech_ir_path": "reverbe_only/speech/050sec.wav"
    },
    "noise_reverbe": {
      "enabled": true,
      "noise_type": "hoth",
      "speech_ir_path": "reverbe_only/speech/050sec.wav",
      "noise_ir_path": "reverbe_only/noise/050sec_000dig.wav"
    }
  }
}
```

**2. 実行**

```bash
python process_audio2.py --config config_A.json
```

-----

### B案: 一括実行ワークフロー (メタデータ管理)

`new_signal_noise.py` と設定ファイル（JSON）を使用して、シミュレーションから畳み込み、メタデータ（RT60, C50 など）の保存までを一括で実行します。

**1. 設定ファイルの準備 (例: `config_B_domain.json`)**

  * `domain_generation_settings` 内に、シミュレーションのパラメータ（部屋数、SNR、チャンネル数など）を指定します。

<!-- end list -->

```json
{
  "base_paths": {
    "speech_data_root": "./sound_data/sample_data/speech",
    "noise_data_root": "./sound_data/sample_data/noise",
    "output_data_root": "./sound_data/mix_data"
  },
  "splits": ["test", "train"],
  "domain_generation_settings": {
    "output_name": "reverb_encoder_dataset_v2",
    "speech_type": "subset_DEMAND",
    "noise_type": "hoth.wav",
    "num_rooms": 10,
    "num_files_per_room": 20,
    "snr": 10,
    "channel": 1
  }
}
```

**2. 実行**

```bash
python new_signal_noise.py --config config_B_domain.json
```

  * 実行後、`output_data_root` に指定されたディレクトリ（例: `reverb_encoder_dataset_v2/test`）内に、`room_000`, `room_001`... といったサブディレクトリが作成され、音声ファイルと共に `metadata.json` が保存されます。

-----

## 従来の機能

### サンプルIRの可視化

`sample_IR.py` を実行することで、サンプルのインパルス応答を生成し、可視化（スペクトログラム など）することができます。

```bash
python sample_IR.py
```

### ビームフォーミング

`All_BF.py` は、DSBF, MVDR, MWFなど複数のビームフォーミング手法を実行するためのスクリプトです。

## セットアップ

### Pythonのバージョン

Python 3.x を推奨します。

### インストール手順

このリポジトリのスクリプトを実行するには、`mymodule` ライブラリをインストールする必要があります。

1.  **仮想環境の作成と有効化 (推奨)**
    ```bash
    # (Windows)
    python -m venv venv
    .\venv\Scripts\activate

    # (Mac/Linux)
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **必要なライブラリのインストール**
    ```bash
    pip install -r requirements.txt
    ```

3.  **`mymodule` のインストール**
    リポジトリのルート（`setup.py` がある場所）で、以下のコマンドを実行します。
    ```bash
    pip install -e .
    ```
    * (`-e` は「編集可能（editable）モード」を意味します。これにより、`src/mymodule/` 内のコードを変更しても、すぐに実行スクリプトに反映されます。)

4.  **動作確認**
    インストールが成功したか確認します。
    ```bash
    python scripts/sample_IR.py
    ```
    * (matplotlibのウィンドウが開き、インパルス応答のスペクトログラム が表示されれば成功です。)

## ディレクトリ構成

  * `new_signal_noise.py`: （B案）メタデータ付きデータセットを一括生成するスクリプト
  * `process_audio2.py`: （A案）RIRと音声を畳み込むスクリプト
  * `Generate_IR.py`: （A案）RIRを生成・保存するスクリプト
  * `sample_IR.py`: インパルス応答の生成と可視化を行うサンプルコード
  * `All_BF.py`: ビームフォーミング（MVDR, MWF等）を実行するスクリプト
  * `mymodule/`: 音響処理に関する自作モジュール
      * `const.py`: 定数を定義 (主にファイルパス)
      * `rec_config.py`: 録音に関する設定（サンプリングレート など）を定義
      * `rec_utility.py`: 録音に関するユーティリティ関数（マイク座標設定、SNR調整、シミュレーション関数など）を定義
      * `my_func.py`: 汎用ヘルパー関数（ファイル検索、WAV操作 など）
      * `reverbe_feater.py`: C50/D50など音響特徴量を計算する関数
  * `requirements.txt`: 必要なライブラリの一覧