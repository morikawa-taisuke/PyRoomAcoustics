import os

def get_file_name(file_path: str) -> str:
    """ ファイル名を取得する
 
    Parameters
    ----------
    file_path:ファイル名を取得するパス
 
    Returns
    -------
    file_name: ファイル名
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    return file_name


def get_dir_name(dir_path):
    """ ファイルパスからディレクトリ名を取得する """
    dir_name = os.path.dirname(dir_path)
    return dir_name


def exists_dir(dir_path):
    """ 
    パスで指定されたディレクトリが存在するか確認し、なければ作成する。
    ファイルパスが渡された場合は、そのファイルが含まれるディレクトリを確認する。
    """
    _, ext = os.path.splitext(dir_path)
    
    # 拡張子がない場合はディレクトリパスとみなし、そのパスを作成
    if not ext:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    # 拡張子がある場合はファイルパスとみなし、親ディレクトリを作成
    else:
        dir_name = os.path.dirname(dir_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
