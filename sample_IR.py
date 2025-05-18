""" 参考サイト
 https://www.wizard-notes.com/entry/python/pyroomacoustics-compute-rir
"""
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pyroomacoustics as pra

# パラメタ
fs = 44100       # サンプリング周波数
absorption = 0.2 # 反射率
max_order = 10    # 次数

# 3次元の部屋形状生成
corners = np.array([[0,0], [0,3], [5,3], [5,1], [3,1], [3,0]]).T
room = pra.Room.from_corners(
        corners,
        max_order=max_order,
        fs=fs,
        absorption=absorption
    )
room.extrude(2.) # 高さを設定

# 音源位置を設定
room.add_source([1., 1., 1.])
# マイク位置を設定
room.add_microphone([1., 0.7, 1.2])

# 部屋形状、音源・マイク位置プロット
fig, ax = room.plot()
ax.set_xlim([0, 5])
ax.set_ylim([0, 3])
ax.set_zlim([0, 2])
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
