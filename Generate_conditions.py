import csv
import os.path
import random

from mymodule import my_func, const

wav_path = os.path.join(const.SOUND_DATA_DIR, "sample_data", "speech", "DEMAND", "clean", "train")

wav_list = my_func.get_file_list(wav_path)
print(wav_path)
print(len(wav_list))

snr = [random.uniform(0.0, 10.0) for _ in range(len(wav_list))]
reverbe = [random.uniform(0.1, 1.0) for _ in range(len(wav_list))]
angle = [random.uniform(0.0, 90.0) for _ in range(len(wav_list))]
# mic = [random.shuffle([0, 1, 2, 3]) for _ in range(len(wav_list))]

# print(len(snr))
# print(len(reverbe))
# print(len(angle))
# print(len(mic))

# CSV ファイルに保存
csv_file = os.path.join(const.SOUND_DATA_DIR, "sample_data", "speech", "DEMAND", "clean", "condition", "train",
                        "condition_5.csv")
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # ヘッダーの書き込み
    writer.writerow(["wav_path", "snr", "reverbe", "angle"])

    # 各行のデータを書き込み
    for i in range(len(wav_list)):
        writer.writerow([wav_list[i], round(snr[i], 1), round(reverbe[i], 1), round(angle[i], 0)])

print(f"CSV file saved to {csv_file}")
