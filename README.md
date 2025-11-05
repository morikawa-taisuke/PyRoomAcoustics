# PyRoomAcoustics

## 概要

`pyroomacoustics` を用いた音響シミュレーションのためのライブラリです。

## 主な機能

* 部屋のインパルス応答の生成
* インパルス応答の可視化（波形、周波数特性、スペクトログラム）

## セットアップ

### 必要なライブラリ

```bash
pip install -r requirements.txt
```

### Pythonのバージョン

Python 3.x を推奨します。

## 使い方

`sample_IR.py` を実行することで、サンプルのインパルス応答を生成し、可視化することができます。

```bash
python sample_IR.py
```

## ディレクトリ構成

* `sample_IR.py`: インパルス応答の生成と可視化を行うサンプルコード
* `mymodule/`: 音響処理に関する自作モジュール
    * `const.py`: 定数を定義
    * `rec_config.py`: 録音に関する設定を定義
    * `rec_utility.py`: 録音に関するユーティリティ関数を定義
* `requirements.txt`: 必要なライブラリの一覧
