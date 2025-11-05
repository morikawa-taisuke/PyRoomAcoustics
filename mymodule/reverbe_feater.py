import numpy as np
from mymodule import rec_config as rec_conf

# C50とD50の計算例
def calculate_c50(rir, fs=rec_conf.sampling_rate):
	t_50ms = int(0.050 * fs)

	# RIRのエネルギー (二乗)
	energy = rir ** 2

	# 50msまでのエネルギー
	e_early = np.sum(energy[:t_50ms])

	# 50ms以降のエネルギー
	e_late = np.sum(energy[t_50ms:])

	# C50の計算
	if e_late > 0:
		c50 = 10 * np.log10(e_early / e_late)
	else:
		c50 = np.inf
	return c50

def calculate_d50(rir, fs=rec_conf.sampling_rate):
	t_50ms = int(0.050 * fs)

	# RIRのエネルギー (二乗)
	energy = rir ** 2

	# 50msまでのエネルギー
	e_early = np.sum(energy[:t_50ms])

	# 全体のエネルギー
	e_total = np.sum(energy)

	# D50の計算
	d50 = (e_early / e_total) * 100

	return d50


if __name__ == "__main__":
	print("reverve_feater.pyの単体実行テスト")