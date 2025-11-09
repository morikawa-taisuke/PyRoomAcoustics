""" 参考サイト
 https://www.wizard-notes.com/entry/python/pyroomacoustics-compute-rir
"""
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from mymodule import rec_utility as rec_util
from scipy import signal


def plot_rir_frequency_response(ir_signal, fs, nfft=4096, use_db=True, title="インパルス応答の周波数特性"):
	"""
	インパルス応答の周波数特性をプロットする関数

	Args:
		ir_signal: インパルス応答（1次元配列）
		fs: サンプリング周波数
		nfft: FFTサイズ
		use_db: True なら dB スケール（20*log10）
		title: 図のタイトル
	"""
	# FFTサイズに合わせてゼロパディング
	ir_padded = np.zeros(nfft)
	ir_length = min(len(ir_signal), nfft)
	ir_padded[:ir_length] = ir_signal[:ir_length]

	# 窓関数を適用（ハニング窓）
	window = np.hanning(ir_length)
	ir_padded[:ir_length] *= window

	# FFTを計算
	H = np.fft.rfft(ir_padded, n=nfft)
	freq = np.fft.rfftfreq(nfft, d=1.0 / fs)

	# 振幅特性を計算
	magnitude = np.abs(H)

	if use_db:
		# dBスケールに変換（ゼロ割り防止）
		magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-12))
		y_data = magnitude_db
		y_label = "振幅 [dB]"
	else:
		y_data = magnitude
		y_label = "振幅"

	# プロット
	plt.figure(figsize=(10, 6))
	plt.subplot(2, 1, 1)
	plt.semilogx(freq, y_data)
	plt.xlabel("周波数 [Hz]")
	plt.ylabel(y_label)
	plt.title(title)
	plt.grid(True, alpha=0.3)
	plt.xlim([20, fs / 2])  # 可聴周波数範囲

	# 位相特性も表示
	plt.subplot(2, 1, 2)
	phase = np.unwrap(np.angle(H))
	plt.semilogx(freq, phase)
	plt.xlabel("周波数 [Hz]")
	plt.ylabel("位相 [rad]")
	plt.title("位相特性")
	plt.grid(True, alpha=0.3)
	plt.xlim([20, fs / 2])

	plt.tight_layout()
	plt.show()


def plot_rir_spectrogram(ir_signal, fs, nperseg=512, noverlap=None, title="インパルス応答のスペクトログラム"):
    """
    インパルス応答のスペクトログラムをプロットする関数（添付画像のような表示）
    
    Args:
        ir_signal: インパルス応答（1次元配列）
        fs: サンプリング周波数
        nperseg: STFTの窓長
        noverlap: オーバーラップサンプル数
        title: 図のタイトル
    """
    if noverlap is None:
        noverlap = nperseg // 2
    
    # STFTを計算
    frequencies, times, Sxx = signal.spectrogram(
        ir_signal, fs=fs, nperseg=nperseg, noverlap=noverlap, 
        window='hann', scaling='density'
    )
    
    # dBスケールに変換
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-12))
    
    # スペクトログラムをプロット
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, frequencies, Sxx_db, shading='gouraud', cmap='jet')
    plt.colorbar(label='パワー [dB]')
    plt.ylabel('周波数 [Hz]')
    plt.xlabel('時間 [s]')
    plt.title(title)
    plt.ylim([0, fs/2])
    plt.tight_layout()
    plt.show()


# パラメタ
fs = 16000       # サンプリング周波数
reverbe = 5
reverbe_par = rec_util.search_reverb_sec(reverbe_sec=reverbe * 0.1, channel=1)  # 任意の残響になるようなパラメータを求める

absorption = reverbe_par[0] # 反射率
max_order = reverbe_par[1]    # 次数

# 3次元の部屋形状生成
room_dim = np.r_[5.0, 5.0, 5.0]  # 部屋の大きさ[x,y,z](m)
room = pra.ShoeBox(room_dim, fs=fs, max_order=max_order, absorption=absorption)  # 残響のみ

# 音源位置を設定
doas = np.array([
    [np.pi / 2., np.pi / 2],
    ])  # 音源の方向[仰角, 方位角](ラジアン)
distance = [0.5, 0.7]  # 音源とマイクの距離(m)
mic_coordinate = rec_util.set_mic_coordinate(center=room_dim / 2, num_channels=1,
                                                     distance=0)  # 線形アレイの場合
source_coordinate = rec_util.set_souces_coordinate2(doas, distance, room_dim / 2)
room.add_source(source_coordinate)
# マイク位置を設定
room.add_microphone_array(pra.MicrophoneArray(mic_coordinate, fs=room.fs))

# 部屋形状、音源・マイク位置プロット
fig, ax = room.plot()
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_zlim([0, 5])
plt.show()


# インパルス応答を計算
room.compute_rir()
# インパルス応答の波形データを保存
ir_signal = room.rir[0][0]
ir_signal /= np.max(np.abs(ir_signal)) # 可視化のため正規化
sf.write(f"ir{fs}.wav", ir_signal, fs)

# 時間領域のIRプロット
room.plot_rir()
plt.show()

# 周波数領域のIRプロット（従来の方法）
# plot_rir_frequency_response(ir_signal, fs, nfft=4096, use_db=True,
#                            title=f"インパルス応答の周波数特性 (残響時間: {reverbe*0.1:.1f}秒)")

# スペクトログラムプロット（新機能 - 添付画像のような表示）
plot_rir_spectrogram(ir_signal, fs, nperseg=512, 
                    title=f"インパルス応答のスペクトログラム (残響時間: {reverbe*0.1:.1f}秒)")

