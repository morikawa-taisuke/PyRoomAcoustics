# -*- coding: utf-8 -*-
"""
【役割】
ファイルパスの操作やディレクトリ作成などのファイルシステム系ユーティリティ
"""
import os
import numpy as np

def get_file_name(file_path: str) -> str:
    """ ファイル名を取得する """
    return os.path.splitext(os.path.basename(file_path))[0]

def get_fname(path):
    """ 目的のファイル名と拡張子を取得 """
    return os.path.splitext(os.path.basename(path))

def get_dirname(path):
    """ 目的ファイルのディレクトリ名を取得 """
    return os.path.dirname(path)

get_dir_name = get_dirname # alias for compatibility

def remove_file(path):
    """ファイルが存在していれば削除"""
    if os.path.exists(path):
        os.remove(path)

def exists_dir(dir_path):
    """パスで指定されたディレクトリが存在するか確認し、なければ作成する。"""
    _, ext = os.path.splitext(dir_path)
    if not ext:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    else:
        dir_name = os.path.dirname(dir_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
    return ext

def get_dir_list(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def get_subdir_list(path:str)->list:
    return [subdir for subdir in os.listdir(path) if os.path.isdir(os.path.join(path,subdir))]

def get_file_list(dir_path:str, ext:str='.wav') -> list:
    if os.path.isdir(dir_path):
        return [f'{dir_path}/{file_path}' for file_path in os.listdir(dir_path) if os.path.splitext(file_path)[1] == ext]
    else:
        return [dir_path]

def make_filename(file_A,file_B):
    name_A, _ = get_fname(file_A)
    name_B, _ = get_fname(file_B)
    return name_A + name_B
