from setuptools import setup, find_packages

setup(
    name="PyRoomAcousticsTools",  # プロジェクト名 (任意)
    version="1.0.0",
    packages=find_packages(where='src'),  # 'src' ディレクトリ配下をパッケージとして検索
    package_dir={'': 'src'},             # パッケージのルートを 'src' に指定
    # 必要なライブラリは requirements.txt に記載されているため、
    # install_requires=[] は省略しても良い
)