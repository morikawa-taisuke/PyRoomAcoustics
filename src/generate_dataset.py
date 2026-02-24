import os
import sys
import yaml
import numpy as np
import soundfile as sf
import pyroomacoustics as pa
import random
import argparse
import pandas as pd
from pathlib import Path
from scipy.signal import fftconvolve
import concurrent.futures
from tqdm import tqdm
from mymodule import const

def load_wav(filepath):
    data, sr = sf.read(filepath, dtype='float32')
    return data, sr

def save_wav(filepath, data, sr):
    sf.write(filepath, data, sr)

def random_crop(noise, target_length):
    if len(noise) <= target_length:
        repeat_times = int(np.ceil(target_length / len(noise)))
        noise = np.tile(noise, repeat_times)
    start = np.random.randint(0, len(noise) - target_length + 1)
    return noise[start:start + target_length]

def mix_snr(speech, noise, snr_db):
    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power < 1e-10:
        return speech

    target_noise_power = speech_power / (10 ** (snr_db / 10))
    noise_scale = np.sqrt(target_noise_power / noise_power)
    
    return speech + noise * noise_scale

def normalize_audio(audio, max_amplitude=0.9):
    """最大振幅を max_amplitude に収める"""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio * (max_amplitude / max_val)
    return audio

def process_single_file(speech_path, noise_paths, config, output_dir_dict, split):
    """
    1つの音声ファイルに対してシミュレーションと重畳を行い、4種類の出力とCSV用のパス情報を返す。
    """
    try:
        # ランダムな雑音とパラメータを選択
        noise_path = Path(random.choice(noise_paths))
        
        # 音声・雑音データの読み込み
        speech, sr = load_wav(speech_path)
        noise, _ = load_wav(noise_path)
        
        # サンプリング周波数のチェック
        if sr != config['system']['sample_rate']:
            raise ValueError(f"Sample rate mismatch: {speech_path} is {sr}Hz, expected {config['system']['sample_rate']}Hz")
            
        noise = random_crop(noise, len(speech))

        # Config からのパラメータ取得
        snr_min = config['acoustic_params']['snr']['min']
        snr_max = config['acoustic_params']['snr']['max']
        snr_db = random.uniform(snr_min, snr_max) if snr_min != snr_max else snr_min

        rt60_min = config['acoustic_params']['rt60']['min']
        rt60_max = config['acoustic_params']['rt60']['max']
        rt60 = random.uniform(rt60_min, rt60_max) if rt60_min != rt60_max else rt60_min

        room_dim = config['room_params']['dimensions']
        mic_pos = config['room_params']['mic_position']
        speech_pos = config['room_params']['speech_position']
        noise_pos = config['room_params']['noise_position']

        # Sabineの式から吸音率と最大反射回数を計算
        e_absorption, max_order = pa.inverse_sabine(rt60, room_dim)
        
        # ====== 1. 教師データ (Clean) ======
        clean_audio = normalize_audio(speech)
        
        # ====== 2. 雑音のみ追加 (Noise Only) ======
        mixed_noise_only = mix_snr(speech, noise, snr_db)
        mixed_noise_only = normalize_audio(mixed_noise_only)
        
        # ====== シミュレーション環境の構築 ======
        room = pa.ShoeBox(
            room_dim, 
            fs=sr, 
            max_order=max_order, 
            materials=pa.Material(e_absorption), 
            air_absorption=True
        )
        
        room.add_microphone(mic_pos)
        room.add_source(speech_pos, signal=speech)
        room.add_source(noise_pos, signal=noise)
        
        # RIRの計算（ここは各音声ごとに実行）
        room.compute_rir()
        rir_speech = room.rir[0][0] # mic 0, source 0 (speech)
        rir_noise = room.rir[0][1] # mic 0, source 1 (noise)
        
        # 畳み込みの実装 (FFT convolution)
        reverb_speech = fftconvolve(speech, rir_speech, mode='full')[:len(speech)]
        reverb_noise = fftconvolve(noise, rir_noise, mode='full')[:len(noise)]
        
        # ====== 3. 残響のみ追加 (Reverb Only) ======
        reverb_only = normalize_audio(reverb_speech)
        
        # ====== 4. 雑音と残響を追加 (Noise + Reverb) ======
        mixed_reverb_noise = mix_snr(reverb_speech, reverb_noise, snr_db)
        mixed_reverb_noise = normalize_audio(mixed_reverb_noise)
        
        # ====== 保存 ======
        speech_stem = speech_path.stem
        noise_stem = noise_path.stem
        
        snr_str = f"{int(snr_db)}dB"
        rt60_str = f"{int(rt60 * 1000)}msec"
        
        # ファイル名の生成
        file_clean = f"{speech_stem}.wav"
        file_noise_only = f"{speech_stem}_{noise_stem}_{snr_str}.wav"
        file_reverb_only = f"{speech_stem}_{rt60_str}.wav"
        file_noise_reverb = f"{speech_stem}_{noise_stem}_{snr_str}_{rt60_str}.wav"
        
        # パスの生成
        path_clean = output_dir_dict['clean'] / file_clean
        path_noise_only = output_dir_dict['noise_only'] / file_noise_only
        path_reverb_only = output_dir_dict['reverb_only'] / file_reverb_only
        path_noise_reverb = output_dir_dict['noise_reverb'] / file_noise_reverb
        
        # WAV書き出し
        save_wav(path_clean, clean_audio, sr)
        save_wav(path_noise_only, mixed_noise_only, sr)
        save_wav(path_reverb_only, reverb_only, sr)
        save_wav(path_noise_reverb, mixed_reverb_noise, sr)
        
        # CSV用レコードの返却
        return {
            "split": split,
            "speech_source_file": str(speech_path.resolve()),
            "noise_source_file": str(noise_path.resolve()),
            "snr_db": snr_db,
            "rt60": rt60,
            "clean": str(path_clean.resolve()),
            "noise_only": str(path_noise_only.resolve()),
            "reverb_only": str(path_reverb_only.resolve()),
            "noise_reverb": str(path_noise_reverb.resolve())
        }
        
    except Exception as e:
        return f"Error processing {speech_path.name}: {str(e)}"

def generate_dataset(config_path):
    # 1. コンフィグ読み込み
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # root_dir = Path(__file__).parent.parent
    root_dir = const.SAMPLE_DATA_DIR

    speech_root = root_dir / config['paths']['speech_dir']
    noise_dir = root_dir / config['paths']['noise_dir']
    output_root = root_dir / config['paths']['output_dir']

    # print(f"root_dir: {root_dir}")
    # print(f"speech_root: {speech_root}")
    # print(f"noise_dir: {noise_dir}")
    # print(f"output_root: {output_root}")
    # exit(2)    
    # 雑音ファイルの取得
    noise_paths = list(noise_dir.rglob("*.wav"))
    if not noise_paths:
        raise FileNotFoundError(f"No noise files found in {noise_dir}")
        
    print(f"Loaded config: {config_path}")
    print(f"Found {len(noise_paths)} noise files.")
    
    all_csv_records = []
    
    # 2. スプリットごとに処理
    for split in config['paths']['splits']:
        split_speech_dir = speech_root / split
        if not split_speech_dir.exists():
            print(f"Warning: Speech directory missing for split '{split}' -> {split_speech_dir}")
            continue
            
        speech_paths = list(split_speech_dir.rglob("*.wav"))
        if not speech_paths:
            print(f"Warning: No speech files found for split '{split}' in {split_speech_dir}")
            continue
            
        print(f"\n--- Processing split '{split}' ({len(speech_paths)} files) ---")
        
        # 出力ディレクトリの作成
        split_out_dir = output_root / split
        output_dir_dict = {
            'clean': split_out_dir / 'clean',
            'noise_only': split_out_dir / 'noise_only',
            'reverb_only': split_out_dir / 'reverb_only',
            'noise_reverb': split_out_dir / 'noise_reverb'
        }
        
        for d in output_dir_dict.values():
            d.mkdir(parents=True, exist_ok=True)
            
        # 3. マルチプロセスによる並列処理
        num_workers = config['system'].get('num_workers', 4)
        print(f"Using {num_workers} parallel workers...")
        
        # ProcessPoolExecutor で並行処理
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # タスクの登録
            futures = {
                executor.submit(process_single_file, sp, noise_paths, config, output_dir_dict, split): sp 
                for sp in speech_paths
            }
            
            # 結果の取得とプログレスバー表示
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Generating {split}"):
                result = future.result()
                if isinstance(result, dict):
                    all_csv_records.append(result)
                else:
                    tqdm.write(f"⚠️ {result}")
                    
    # 4. CSV の生成
    if all_csv_records:
        df = pd.DataFrame(all_csv_records)
        print("\n--- Generating CSV Files ---")
        
        # 全体 CSV
        all_csv_path = output_root / "dataset.csv"
        df.to_csv(all_csv_path, index=False)
        print(f"Saved global CSV: {all_csv_path}")
        
        # スプリット別 CSV
        for split_name, group in df.groupby("split"):
            split_csv = output_root / f"{split_name}.csv"
            # split列は不要なので落とす
            group.drop(columns="split").to_csv(split_csv, index=False)
            print(f"Saved split CSV: {split_csv}")
    else:
        print("\n⚠️ No records to save to CSV.")
        
    print("\n✅ Generation completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Audio Dataset")
    parser.add_argument("--config", type=str, default="config/dataset_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # スクリプトをルートディレクトリから実行している前提の相対パス補正
    # 実行場所に合わせてパスを調整する
    if not Path(args.config).exists():
        # src/ 内から実行した場合の補正
        fallback_config = Path(__file__).parent.parent / "config/dataset_config.yaml"
        if fallback_config.exists():
            args.config = str(fallback_config)
            
    generate_dataset(args.config)
