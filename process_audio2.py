import os
import argparse
import pandas as pd
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from tqdm import tqdm
from mymodule import my_func, const


def load_wav(filepath):
    data, sr = sf.read(filepath)
    return data, sr

def save_wav(filepath, data, sr):
    sf.write(filepath, data, sr)

def random_crop(noise, target_length):
    if len(noise) <= target_length:
        # 長さが足りない場合はループして埋める
        repeat_times = int(np.ceil(target_length / len(noise)))
        noise = np.tile(noise, repeat_times)
    start = np.random.randint(0, len(noise) - target_length + 1)
    return noise[start:start + target_length]

def apply_ir(signal, ir):
    return fftconvolve(signal, ir, mode='full')[:len(signal)]

def mix_snr(speech, noise, snr_db):
    # SNR調整
    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    target_noise_power = speech_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / (noise_power + 1e-10))
    return speech + noise


def clean(speech_dir, ir_path, output_dir):
    print("-"*32)
    print(f"speech_dir: {speech_dir}")
    print(f"ir_dir: {ir_path}")
    print(f"output_dir: {output_dir}")
    print("-"*32)

    os.makedirs(output_dir, exist_ok=True)
    my_func.exists_dir(output_dir)

    speech_files = [f for f in os.listdir(speech_dir) if f.endswith('.wav')]

    for speech_file in tqdm(speech_files):
        snr = 5.0  # SNRは0に設定

        # 読み込み
        speech_file = os.path.join(speech_dir, speech_file)
        speech, sr = load_wav(speech_file)
        speech_ir, _ = load_wav(ir_path[0])

        # IR畳み込み
        speech_reverb = apply_ir(speech, speech_ir)

        # 保存
        out_name = f"{my_func.get_fname(speech_file)[0]}.wav"
        out_path = os.path.join(output_dir, out_name)
        save_wav(out_path, speech_reverb, sr)
        # print(f"Saved: {out_path}")

def noise_reverbe(speech_dir, noise_path, ir_path, output_dir):
    print("--------------------------------")
    print(f"speech_dir: {speech_dir}")
    print(f"noise_dir: {noise_path}")
    print(f"ir_dir: {ir_path}")
    print(f"output_dir: {output_dir}")
    print("--------------------------------")

    os.makedirs(output_dir, exist_ok=True)
    my_func.exists_dir(output_dir)

    speech_files = [f for f in os.listdir(speech_dir) if f.endswith('.wav')]

    for speech_file in tqdm(speech_files, total=len(speech_files)):
        snr = 5.0  # SNRは0に設定

        # ファイルパス生成
        speech_file = os.path.join(speech_dir, speech_file)
        speech_ir_path = ir_path[0]
        noise_ir_path = ir_path[1]

        # 読み込み
        speech, sr = load_wav(speech_file)
        noise, _ = load_wav(noise_path)
        speech_ir, _ = load_wav(speech_ir_path)
        noise_ir, _ = load_wav(noise_ir_path)

        # noise切り出し
        noise = random_crop(noise, len(speech))

        # IR畳み込み
        speech_reverb = apply_ir(speech, speech_ir)
        noise_reverb = apply_ir(noise, noise_ir)

        # SNR調整
        mixed = mix_snr(speech_reverb, noise_reverb, snr)

        # 保存
        out_name = f"{my_func.get_fname(speech_file)[0]}_{my_func.get_fname(noise_path)[0]}_{int(snr * 10):03}dB.wav"
        out_path = os.path.join(output_dir, out_name)
        save_wav(out_path, mixed, sr)
        # print(f"Saved: {out_path}")

def reverbe_only(speech_dir, ir_path, output_dir):
    print("--------------------------------")
    print(f"speech_dir: {speech_dir}")
    print(f"ir_dir: {ir_path}")
    print(f"output_dir: {output_dir}")
    print("--------------------------------")

    speech_files = [f for f in os.listdir(speech_dir) if f.endswith('.wav')]
    os.makedirs(output_dir, exist_ok=True)
    my_func.exists_dir(output_dir)

    for speech_file in tqdm(speech_files):
        # 読み込み
        speech_file = os.path.join(speech_dir, speech_file)
        speech, sr = load_wav(speech_file)
        speech_ir, _ = load_wav(ir_path[0])

        # IR畳み込み
        speech_reverb = apply_ir(speech, speech_ir)

        # 保存
        out_name = f"{my_func.get_fname(speech_file)[0]}.wav"
        out_path = os.path.join(output_dir, out_name)
        save_wav(out_path, speech_reverb, sr)
        # print(f"Saved: {out_path}")

def noise_only(speech_dir, noise_dir, ir_path, output_dir):
    print("--------------------------------")
    print(f"speech_dir: {speech_dir}")
    print(f"noise_dir: {noise_dir}")
    print(f"ir_dir: {ir_path}")
    print(f"output_dir: {output_dir}")
    print("--------------------------------")

    os.makedirs(output_dir, exist_ok=True)
    my_func.exists_dir(output_dir)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # ファイル名抽出
        speech_file = os.path.basename(row['wav_path'])
        noise_file = os.path.basename(row['noise_path'])
        speech_ir_file = os.path.basename(row['speech_IR'])
        noise_ir_file = os.path.basename(row['noise_IR'])
        snr = float(row['snr'])

        # ファイルパス生成
        speech_path = os.path.join(speech_dir, speech_file)
        noise_path = os.path.join(noise_dir, noise_file)
        speech_ir_path = os.path.join(ir_dir, "speech", speech_ir_file + ".wav")
        noise_ir_path = os.path.join(ir_dir, "noise", noise_ir_file + ".wav")

        # 読み込み
        speech, sr = load_wav(speech_path)
        noise, _ = load_wav(noise_path)
        speech_ir, _ = load_wav(speech_ir_path)
        noise_ir, _ = load_wav(noise_ir_path)

        # noise切り出し
        noise = random_crop(noise, len(speech))

        # IR畳み込み
        speech_reverb = apply_ir(speech, speech_ir)
        noise_reverb = apply_ir(noise, noise_ir)

        # SNR調整
        mixed = mix_snr(speech_reverb, noise_reverb, snr)

        # 保存
        out_name = f"{my_func.get_fname(speech_file)[0]}_{my_func.get_fname(noise_file)[0]}_{int(snr * 10):03}dB.wav"
        out_path = os.path.join(output_dir, out_name)
        save_wav(out_path, mixed, sr)
        # print(f"Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSVに従い音声データを処理')
    parser.add_argument('--csv_path', type=str, help='csvファイルのパス')
    parser.add_argument('--speech_dir', type=str, help='speechデータのディレクトリ')
    parser.add_argument('--noise_dir', type=str, help='noiseデータのディレクトリ')
    parser.add_argument('--ir_dir', type=str, help='IRのディレクトリ')
    parser.add_argument('--output_dir', type=str, help='出力先ディレクトリ')
    args = parser.parse_args()

    test_train = "test"
    for i in range(1, 2):
        # "C:\Users\kataoka-lab\Desktop\sound_data\sample_data\speech\GNN\subset_DEMAND\condition\train\condition_1.csv"
        speech_dir = f"{const.SAMPLE_DATA_DIR}/speech/subset_DEMAND/{test_train}"

        ir_dir = [f"C:/Users/kataoka-lab/Desktop/sound_data/sample_data/IR/1ch_0cm_liner/clean/speech/050sec.wav",
                  f"C:/Users/kataoka-lab/Desktop/sound_data/sample_data/IR/1ch_0cm_liner/clean/noise/050sec_000dig.wav"]
        output_dir =  f"{const.MIX_DATA_DIR}/subset_DEMAND_1ch/{test_train}/clean"
        clean(speech_dir, ir_dir, output_dir)

        # noise_dir = "C:/Users/kataoka-lab/Desktop/sound_data/sample_data/noise/DEMAND/"
        ir_dir = [f"C:/Users/kataoka-lab/Desktop/sound_data/sample_data/IR/1ch_0cm_liner/reverbe_only/speech/050sec.wav",
                  f"C:/Users/kataoka-lab/Desktop/sound_data/sample_data/IR/1ch_0cm_liner/reverbe_only/noise/050sec_000dig.wav"]
        output_dir =  f"{const.MIX_DATA_DIR}/subset_DEMAND_1ch/{test_train}/reverbe_only"
        reverbe_only(speech_dir, ir_dir, output_dir)

        noise_dir = f"{const.SAMPLE_DATA_DIR}/noise/hoth.wav"
        output_dir =  f"{const.MIX_DATA_DIR}/subset_DEMAND_1ch/{test_train}/noise_reverbe"
        noise_reverbe(speech_dir, noise_dir, ir_dir, output_dir)
