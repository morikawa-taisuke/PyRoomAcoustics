import os
import numpy as np
import wave
import array

from librosa.util import find_files

from mymodule import const


def get_fname(path):
    """ 目的のファイル名と拡張子を取得

    Args:
        path: 目的ファイルのパス

    Returns:
        fname   : ファイル名
        ext     : 拡張子
    """
    fname, ext = os.path.splitext(os.path.basename(path))
    return fname, ext

def get_dirname(path):
    """ 目的ファイルのディレクトリ名を取得

    Args:
        path: 目的のファイル名

    Returns:
        dir_name : ディレクトリ名
    """
    dir_name = os.path.dirname(path)
    return dir_name


def remove_file(path):
    """ファイルが存在していれば削除

    Args:
        path: 目的ファイルのパス

    Returns:
        None
    """
    if os.path.exists(path):
        os.remove(path)

def exists_dir(dir_name):
    """ 目的のディレクトリが存在するか確認.ない場合は作る

    Args:
        dir_name: ディレクトリ名

    Returns:
        ext: 拡張子
    """
    _, ext = os.path.splitext(dir_name)
    # print("util : exists_dir", _, ext)
    if len(ext) == 0 and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    elif not len(ext) == 0 and not os.path.exists(get_dirname(dir_name)):
        os.makedirs(get_dirname(dir_name))
    return ext

def get_dir_list(path):
    """指定したディレクトリ内に存在するディレクトリをリストアップ

    :param path: 探索するディレクトリのパス
    :return dir_list: path内に存在するディレクトリのパスのリスト
    """
    dir_list = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return dir_list

def get_subdir_list(path:str)->list:
    """
    指定したディレクトリの子ディレクトリのディレクトリ名のみをリストアップ

    dir
    |
    |----dir1
    |
    -----dir2

    get_dir_list("./dir")->["dir1", "dir2"]
    Parameters
    ----------
    path(str):  親ディレクトリ(dir)

    Returns
    -------

    """
    subdir_list = [subdir for subdir in os.listdir(path) if os.path.isdir(os.path.join(path,subdir))]
    return subdir_list

def get_file_list(dir_path:str, ext:str='.wav') -> list:
    """
    指定したディレクトリ内の任意の拡張子のファイルをリストアップ

    Parameters
    ----------
    dir_path(str):ディレクトリのパス
    ext(str):拡張子

    Returns
    -------
    list[str]
    """
    if os.path.isdir(dir_path):
        return [f'{dir_path}/{file_path}' for file_path in os.listdir(dir_path) if os.path.splitext(file_path)[1] == ext]
    else:
        return [dir_path]

def load_wav(path):
    """ 保存のSRが異なれば変換する

    Args:
        path: wavファイルのパス

    Returns:
        amplitude   : 振幅
        prm         : パラメータ
    """
    wav = wave.open(path, "r")
    #print("wav", wav.shape)
    prm = wav.getparams()
    #print("prm", prm.shape)
    buffer = wav.readframes(wav.getnframes())
    #print("buffer", buffer.shape)
    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    #print("amptitude", amptitude.shape)
    #sr = prm.framerate
    #if not sr == SR:
        # サンプリングレートをあわせる
        #amptitude = resample(amptitude.astype(np.float64), sr, SR)

    return amptitude, prm

def save_wav(path, wav, prm):
    """ wavファイルの保存

    Args:
        path    : wavファイルのパス
        wav     : 保存する波形
        prm     : パラメータ

    Returns:
        None
    """
    f = wave.Wave_write(path)
    f.setparams(prm)
    #f.setframerate(SR)
    f.writeframes(array.array("h", wav.astype(np.int16)).tobytes())
    f.close()

def make_filename(file_A,file_B):
    name_A, _ = get_fname(file_A)
    name_B, _ = get_fname(file_B)
    print("name_A:", name_A)
    print("name_B:", name_B)
    file_name = name_A+name_B
    return file_name
