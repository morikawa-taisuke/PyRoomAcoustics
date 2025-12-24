import os
import random
import math
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# rec2.py から必要な関数をインポート
from rec2 import recoding2
from mymodule import const, my_func
import pyroomacoustics as pa


def generate_random_dataset(num_samples=1000):
    """
    ランダムな条件下でデータセットを生成するスクリプト
    """

    # --- 設定項目（ここを調整してください） ---
    speech_type = "VCTK"  # 目的音声のタイプ
    noise_type = "DEMAND"  # 雑音信号のタイプ（フォルダ名など）

    # パス設定
    speech_dir = f"{const.SAMPLE_DATA_DIR}/speech/{speech_type}/test"
    noise_dir = f"{const.SAMPLE_DATA_DIR}/noise/{noise_type}"  # 雑音が複数入っているフォルダ

    # パラメータの範囲設定
    snr_range = (-5, 15)  # SNRの候補 [dB]
    reverbe_range = (0.3, 1.0)  # 残響時間の範囲 [sec] (min, max)
    room_dim = np.array([5.0, 5.0, 5.0])  # 部屋のサイズ

    # マイク設定
    ch = 1  # マイク数
    mic_distance = 10  # マイク間隔 [cm]

    output_base_dir = (f"{const.MIX_DATA_DIR}/Random_Dataset_{speech_type}_{noise_type}_{ch}ch"
                       f"")
    # 1. データのリストを取得
    speech_files = my_func.get_file_list(speech_dir)
    noise_files = my_func.get_file_list(noise_dir)  # フォルダ内の全wav取得を想定

    if not speech_files or not noise_files:
        print("Error: 音声ファイルまたは雑音ファイルが見つかりません。")
        return

    print(f"Total samples to generate: {num_samples}")

    # 2. 生成ループ
    for i in tqdm(range(len(speech_files))):
        # --- パラメータのランダム決定 ---
        target_file = random.choice(speech_files)
        noise_file = random.choice(noise_files)

        snr = random.randint(snr_range[0], snr_range[1])
        rt60 = random.uniform(reverbe_range[0], reverbe_range[1])  # 範囲内から一様分布で

        # 雑音の到来方向（方位角）を0〜360度でランダム決定
        angle_deg = random.randint(0, 359)
        angle_rad = math.radians(angle_deg)
        angle_name = f"{angle_deg:03}deg"

        # --- 残響パラメータの計算 ---
        # 毎回 serch_reverbe_sec を回すと遅いため、簡易的に Sabine の式で計算
        # もし精度を優先する場合は serch_reverbe_sec を使用してください
        e_absorption, max_order = pa.inverse_sabine(rt60, room_dim)
        reverbe_par = (e_absorption, max_order)

        # 出力ディレクトリの設定（サブディレクトリを分けると管理しやすい）
        # 例: out_dir/snr_10/rt60_0450/
        # current_out_dir = os.path.join(output_base_dir, f"sample_{i:05}")
        current_out_dir = output_base_dir
        my_func.exists_dir(current_out_dir)

        # 3. 録音実行
        # try:
        recoding2(
            wave_files=[target_file, noise_file],
            out_dir=current_out_dir,
            snr=snr,
            reverbe_sec=rt60,  # ファイル名用
            reverbe_par=reverbe_par,
            channel=ch,
            distance=mic_distance,
            is_split=False,
            angle=angle_rad,
            angle_name=angle_name
        )
        # except Exception as e:
        #     print(f"Error at sample {i}: {e}")
        #     continue


if __name__ == "__main__":
    # シード値を固定したい場合は以下を有効化
    random.seed(100)
    np.random.seed(100)

    generate_random_dataset()  # まずは少ない数でテスト推奨