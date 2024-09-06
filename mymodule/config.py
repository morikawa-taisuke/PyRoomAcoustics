import numpy as np

MIX_DIR = "./rec_1ch/mix_data/"

# シミュレーションのパラメータ
SAMPLE_RATE=16000 #サンプリング周波数
N=1024 #フレームサイズ
Nk=N/2+1 #周波数の数
FREQS=np.arange(0,Nk,1)*SAMPLE_RATE/N #各ビンの周波数

NFFT = 512
NOVERLAP = 512//16*15
