import numpy as np
import pyroomacoustics as pra
import soundfile as sf  # soundfileライブラリを使用

# 任意のインパルス応答を生成
room_dim = [10, 7, 3]  # 部屋の寸法 (幅, 奥行き, 高さ)
room = pra.ShoeBox(room_dim, fs=16000, max_order=15)

# 音源とマイクアレイの位置を設定
source_position = [2, 3, 1.5]  # 音源の位置
mic_array_positions = [
    [6, 4, 1.5],
    [6, 4.05, 1.5],
    [6, 4.1, 1.5],
    [6, 4.15, 1.5],
]  # 4chマイクアレイ

room.add_source(source_position)
for mic in mic_array_positions:
    room.add_microphone(mic)

# 部屋のシミュレーションを実行
room.compute_rir()

# インパルス応答を取得
rir = [room.rir[i] for i in range(len(mic_array_positions))]

# 任意の音声信号を読み込み
input_audio, fs = sf.read("C:\\Users\\kataoka-lab\\Desktop\\sound_data\\sample_data\\speech\\subset_DEMAND\\test\\p232_068_16kHz.wav")  # 元の音声ファイル
if input_audio.ndim > 1:
    input_audio = input_audio[:, 0]  # モノラル化

# 各マイクで残響付加後の信号を作成
output_signals = []
for i in range(4):  # 4つのマイクに対して処理
    convolved_signal = np.convolve(input_audio, rir[i][0])  # インパルス応答と畳み込み
    output_signals.append(convolved_signal)

# 各信号の長さを揃える
max_length = max([len(signal) for signal in output_signals])
padded_signals = np.array([np.pad(signal, (0, max_length - len(signal))) for signal in output_signals])

# 信号を4チャンネルとして保存
output_file = "output_4ch_audio.wav"
sf.write(output_file, padded_signals.T, fs)  # `.T`でチャンネルを正しい形に
print(f"4ch音声ファイルを保存しました: {output_file}")
