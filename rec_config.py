import numpy as np

sampling_rate = 16000   # サンプリング周波数
frame = 1024    # フレーム数
Nk = int(frame/2+1) # 周波数の分解能
freqs = np.arange(0, Nk, 1)* sampling_rate / frame  # 各ビン数

fft = 512
overlap = 512//16*15
