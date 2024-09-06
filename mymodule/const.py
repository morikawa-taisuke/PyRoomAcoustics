"""
construction
設定

"""
KEY_SAVE_STFT_TARGET = 'target'
KEY_SAVE_STFT_MIXDOWN = 'mix'

KEY_FILE_WAVE_TARGET = 'target'
KEY_FILE_WAVE_NOISE = 'noise'
KEY_FILE_WAVE_MIX = 'mix'

SR = 16000 # サンプリングレート?
FFT_SIZE = 1024 # fftのサイズ
H = 256 # ?

BATCHSIZE = 32 # バッチサイズ(一度の学習に読み込むファイルサイズ)
PATCHLEN = 16 # ?
EPOCH = 5 # 学習回数
