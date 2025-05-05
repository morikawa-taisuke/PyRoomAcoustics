# ベースイメージを指定
FROM python:3.8

# 作業ディレクトリを作成
WORKDIR /app

# ローカルのファイルをコンテナにコピー
COPY . /app

# 必要なライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションを実行
# CMD ["python", "app.py"]
