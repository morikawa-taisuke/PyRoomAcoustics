FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 基本ツールのインストール
RUN apt-get update && apt-get install -y \
    python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /app

# ソースコードをコピー
COPY . /app

# Pythonパッケージのインストール
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# エントリーポイント（必要に応じて）
#CMD ["python3", "train.py"]
