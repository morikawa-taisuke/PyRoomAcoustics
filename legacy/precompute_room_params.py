import argparse
import json
import numpy as np
import pyroomacoustics as pa
from pathlib import Path
from tqdm import tqdm

from src.mymodule import const, rec_utility as rec_util

def search_room_params(target_rt60, room_dim, fs=16000, max_iter=20):
    """
    指定されたRT60になるような部屋のパラメータ（吸音率、最大反射回数）を探索する。
    """
    
    # 初期推定: 目標値をそのまま入力
    current_rt60_input = target_rt60
    best_params = None
    min_diff = float('inf')
    
    for i in range(max_iter):
        try:
            # Sabineの式から吸音率と最大反射回数を計算
            e_absorption, max_order = pa.inverse_sabine(current_rt60_input, room_dim)
        except ValueError:
             # 計算不能な場合（吸音率が範囲外など）
             break

        # 部屋を作成
        # air_absorption=True にすることでより現実に近いシミュレーションを行う
        try:
            room = pa.ShoeBox(room_dim, fs=fs, max_order=max_order, materials=pa.Material(e_absorption), air_absorption=True)
        except Exception:
            break
        
        # RT60計測のためにマイクと音源を配置
        # 部屋の中心付近に配置
        room.add_microphone_array(pa.MicrophoneArray(np.c_[room_dim/2], fs=fs))
        source_position = rec_util.get_source_positions()
        room.add_source(room_dim/3)
        
        # RIR計算
        room.compute_rir()
        
        # RT60計測 (平均値を使用)
        try:
            measured_rt60 = room.measure_rt60()[0][0]
        except Exception:
            measured_rt60 = 0
        
        diff = abs(measured_rt60 - target_rt60)
        
        # 最良の結果を更新
        if diff < min_diff:
            min_diff = diff
            best_params = {
                "rt60": measured_rt60, # 実測値
                "absorption": e_absorption,
                "max_order": max_order
            }
        
        # 許容誤差 (例: 0.01秒) 以内なら終了
        if diff < 0.01:
            break
            
        # フィードバック制御: 実測値と目標値の比率で入力値を補正
        if measured_rt60 == 0:
            break
            
        ratio = target_rt60 / measured_rt60
        # 急激な変化を抑えるために平方根をとるなどの工夫
        current_rt60_input *= (ratio ** 0.5)
            
    return best_params

def main():
    parser = argparse.ArgumentParser(description="各残響時間の時のパラメータを計算しJSON形式で出力する")
    parser.add_argument("--out_dir", type=str, default=f"{const.SAMPLE_DATA_DIR}/precompute_params", help="出力ディレクトリ")
    parser.add_argument("--room_dim", type=float, nargs=3, default=[5.0, 5.0, 5.0], help="部屋の寸法 (x y z) [m]")
    parser.add_argument("--rt60_min", type=float, default=0.3, help="最小RT60 [s]")
    parser.add_argument("--rt60_max", type=float, default=1.0, help="最大RT60 [s]")
    parser.add_argument("--rt60_step", type=float, default=0.01, help="RT60のステップ [s]")
    parser.add_argument("--fs", type=int, default=16000, help="サンプリング周波数 [Hz]")
    
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    room_dim = np.array(args.room_dim)
    # 浮動小数点の誤差を考慮して少し大きめに範囲を取る
    rt60_list = np.arange(args.rt60_min, args.rt60_max + 0.001, args.rt60_step)
    
    results = {}
    
    print(f"Room Dim: {room_dim}")
    print(f"Calculating params for RT60: {rt60_list}")
    
    for target_rt60 in tqdm(rt60_list):
        params = search_room_params(target_rt60, room_dim, fs=args.fs)
        if params:
            # キーは "0.20s" のような形式にする
            key = f"{target_rt60:.2f}s"
            results[key] = params
            tqdm.write(f"\nTarget: {target_rt60:.2f}s -> Measured: {params['rt60']:.3f}s, Abs: {params['absorption']:.4f}, Order: {params['max_order']}")
        else:
            tqdm.write(f"Failed to find params for {target_rt60:.2f}s")

    # ファイル名に部屋の寸法を含める
    filename = f"room_params_{int(room_dim[0]*100)}cm_{int(room_dim[1]*100)}cm_{int(room_dim[2]*100)}cm.json"
    out_path = out_dir / filename
    
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
