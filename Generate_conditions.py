import csv
import os.path
import random

from mymodule import my_func, const


def main(wav_dir, noise_dir, csv_path):
    wav_list = my_func.get_file_list(wav_dir)
    noise_list = my_func.get_file_list(noise_dir)
    # print("wav_dir:", wav_dir)
    # print("wav_list:", len(wav_list))
    # print("noise_list:", len(noise_list))
    snr = [random.uniform(0.0, 10.0) for _ in range(len(wav_list))]
    reverbe = [random.uniform(10.0, 100.0) for _ in range(len(wav_list))]
    angle = [random.uniform(0, 90) for _ in range(len(wav_list))]
    noise = [random.choice(noise_list) for _ in range(len(wav_list))]
    # mic = [random.shuffle([0, 1, 2, 3]) for _ in range(len(wav_list))]

    my_func.exists_dir(csv_path)
    # CSV ファイルに保存
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # ヘッダーの書き込み
        writer.writerow(["wav_path", "noise_path", "snr", "speech_IR", "noise_IR", "reverbe_sec", "angle"])

        # 各行のデータを書き込み
        for i in range(len(wav_list)):
            writer.writerow([my_func.get_fname(wav_list[i])[0], my_func.get_fname(noise[i])[0], round(snr[i], 1), f"{int(round(reverbe[i], 0)):03}sec", f"{int(round(reverbe[i], 0)):03}sec_{int(round(angle[i], 0)):03}dig", int(round(reverbe[i], 0)), round(angle[i], 0)])

        print(f"CSV file saved to {csv_path}")

if __name__ == "__main__":
    test_train = "train"
    for i in range(1, 5+1):
        wav_dir = os.path.join(const.SAMPLE_DATA_DIR, "speech", "subset_DEMAND", test_train)
        noise_dir = os.path.join(const.SAMPLE_DATA_DIR, "noise", "DEMAND")
        csv_path = os.path.join(const.SAMPLE_DATA_DIR, "speech", "GNN", "subset_DEMAND", "condition", test_train, f"condition_{i}.csv")
        main(wav_dir, noise_dir, csv_path)
