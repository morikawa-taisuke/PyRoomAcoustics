# -*- coding: utf-8 -*-
"""
pyroomacousticsを用いてインパルス応答（RIR）を生成し、
そこから残響特性を評価する特徴量を抽出・保存するスクリプト。

特徴量として、ケプストラム係数とRT60を抽出します。
生成されたRIRと特徴量は.npzファイルとして保存されます。

使用例:
python generate_reverb_features.py --output_dir "reverb_features" --num_samples 50
"""

import argparse
import os
import pyroomacoustics as pra
import numpy as np
import soundfile as sf
import scipy.signal as sp
from scipy.signal import get_window
import random

# 既存のmymodule/my_func.pyをインポート
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'mymodule'))
from mymodule import my_func


def lpcoeff(speech_frame, model_order):
	"""
    音声フレームから線形予測係数（LPC）を計算します。
    (evaluation/cepsdist.pyから抜粋)
    """
	eps = np.finfo(np.float64).eps
	winlength = len(speech_frame)
	R = np.zeros((model_order + 1,))
	for k in range(model_order + 1):
		if k == 0:
			R[k] = np.sum(speech_frame * speech_frame)
		else:
			R[k] = np.sum(speech_frame[0:-k] * speech_frame[k:])

	a = np.ones((model_order,))
	a_past = np.ones((model_order,))
	rcoeff = np.zeros((model_order,))
	E = np.zeros((model_order + 1,))
	E[0] = R[0]

	for i in range(model_order):
		a_past[0:i] = a[0:i]
		sum_term = np.sum(a_past[0:i] * R[i:0:-1])
		if E[i] == 0.0:
			rcoeff[i] = np.inf
		else:
			rcoeff[i] = (R[i + 1] - sum_term) / (E[i])
		a[i] = rcoeff[i]
		if i > 0:
			a[0:i] = a_past[0:i] - rcoeff[i] * a_past[i - 1::-1]
		E[i + 1] = (1 - rcoeff[i] * rcoeff[i]) * E[i]

	lpparams = np.ones((model_order + 1,))
	lpparams[1:] = -a
	return lpparams


def lpc2cep(a):
	"""
    線形予測係数（LPC）からケプストラム係数に変換します。
    (evaluation/cepsdist.pyから抜粋)
    """
	M = len(a)
	cep = np.zeros((M - 1,))
	cep[0] = -a[1]

	for k in range(2, M):
		ix = np.arange(1, k)
		vec1 = cep[ix - 1] * a[k - 1:0:-1] * ix
		cep[k - 1] = -(a[k] + np.sum(vec1) / k)

	return cep


def extract_reverb_features(ir_signal, fs, model_order=16):
	"""
    RIR信号から残響特徴量を抽出します。

    Args:
        ir_signal (np.ndarray): インパルス応答信号。
        fs (int): サンプリング周波数。
        model_order (int): LPC分析のモデル次数。

    Returns:
        dict: 抽出された特徴量を含む辞書。
    """
	# RT60（残響時間）の計算
	rt60 = pra.experimental.rt60.measure(ir_signal, fs=fs)

	# RIRのLPC分析
	lpc_coeffs = lpcoeff(ir_signal, model_order)

	# ケプストラム係数の計算
	cepstrum_coeffs = lpc2cep(lpc_coeffs)

	# C50やD50も計算可能だが、ここではシンプルにケプストラム係数とRT60を返す
	return {
		'rt60': rt60,
		'cepstrum_coeffs': cepstrum_coeffs
	}


def main(output_dir, num_samples):
	"""
    複数の部屋パラメータでRIRを生成し、特徴量を抽出して保存します。
    """
	fs = 16000

	# 出力ディレクトリの作成
	my_func.make_dir(output_dir)

	print(f"Generating and extracting features for {num_samples} RIRs...")

	# 多様な部屋パラメータの範囲を定義
	room_dims_range = [(3, 4, 2.5), (5, 6, 3), (8, 10, 4)]
	rt60_range = np.arange(0.2, 1.0, 0.1)

	for i in range(num_samples):
		# パラメータをランダムに選択
		room_dims = random.choice(room_dims_range)
		rt60 = np.round(random.choice(rt60_range), 2)

		# 音源とマイクの位置を部屋の寸法内にランダムに設定
		source_pos = [np.random.uniform(1, dim - 1) for dim in room_dims]
		mic_pos = [np.random.uniform(1, dim - 1) for dim in room_dims]

		# 部屋の吸音率をRT60から計算
		e_absorption, max_order = pra.inverse_sabine(rt60, room_dims)

		# 部屋オブジェクトの作成
		room = pra.ShoeBox(
			room_dims,
			fs=fs,
			materials=pra.Material(e_absorption),
			max_order=max_order
		)

		# 音源とマイクを部屋に追加
		room.add_source(source_pos)
		room.add_microphone(mic_pos)

		# RIRを計算
		room.compute_rir()
		rir_signal = room.rir[0][0]

		# 特徴量を抽出
		features = extract_reverb_features(rir_signal, fs)

		# ファイル名を生成
		filename = f"rir_{i:03d}.npz"
		output_path = os.path.join(output_dir, filename)

		# RIRと特徴量を.npzファイルに保存
		np.savez(output_path,
		         rir=rir_signal,
		         rt60=features['rt60'],
		         cepstrum_coeffs=features['cepstrum_coeffs'])

		if (i + 1) % 10 == 0:
			print(f"✅ {i + 1}/{num_samples}件のデータ生成と特徴量抽出が完了しました。")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='RIRを生成し、特徴量を抽出して保存します。')
	parser.add_argument('--output_dir', type=str, default='reverb_features',
	                    help='生成した特徴量を保存するディレクトリ')
	parser.add_argument('--num_samples', type=int, default=100,
	                    help='生成するRIRの数')

	args = parser.parse_args()

	main(
		output_dir=args.output_dir,
		num_samples=args.num_samples
	)