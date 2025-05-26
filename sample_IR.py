""" 参考サイト
 https://www.wizard-notes.com/entry/python/pyroomacoustics-compute-rir
"""
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from mymodule import const, rec_config as rec_conf, rec_utility as rec_util


# パラメタ
fs = 16000       # サンプリング周波数
absorption = 0.2 # 反射率
max_order = 10    # 次数

# 3次元の部屋形状生成
room_dim = np.r_[3.0, 3.0, 3.0]  # 部屋の大きさ[x,y,z](m)
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
ax.set_xlim([0, 3])
ax.set_ylim([0, 3])
ax.set_zlim([0, 3])
plt.show()


# インパルス応答を計算
room.compute_rir()
# インパルス応答の波形データを保存
ir_signal = room.rir[0][0]
ir_signal /= np.max(np.abs(ir_signal)) # 可視化のため正規化
sf.write(f"ir{fs}.wav", ir_signal, fs)

# IRプロット
room.plot_rir()
plt.show()
