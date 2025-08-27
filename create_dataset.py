import pyroomacoustics as pra
import numpy as np
import soundfile as sf
import os
import random


class DatasetGenerator:
	"""
	音声強調用データセットを生成するクラス
	"""

	def __init__(self, config):
		self.config = config
		self.room_dim_range = self.config.get('room_dim_range', ([3, 8], [3, 8], [2.5, 4]))
		self.absorption_range = self.config.get('absorption_range', (0.1, 0.6))
		self.fs = self.config.get('sample_rate', 16000)
		self.output_dir = self.config.get('output_dir', 'dataset_speech_enhancement')

		# 出力ディレクトリを作成
		for sub_dir in ['clean', 'reverbe_only', 'noise_only', 'noise_reverb']:
			os.makedirs(os.path.join(self.output_dir, sub_dir), exist_ok=True)

	def generate_single_sample(self, clean_speech_path, noise_path, sample_id):
		"""
		1つのサンプルを生成し、保存する
		"""
		clean_speech, sr_clean = sf.read(clean_speech_path)
		noise, sr_noise = sf.read(noise_path)

		# 部屋と音源・マイクのパラメータをランダムに決定
		room_dim = [
			random.uniform(self.room_dim_range[0][0], self.room_dim_range[0][1]),
			random.uniform(self.room_dim_range[1][0], self.room_dim_range[1][1]),
			random.uniform(self.room_dim_range[2][0], self.room_dim_range[2][1]),
		]
		absorption = random.uniform(self.absorption_range[0], self.absorption_range[1])

		# 共通の部屋オブジェクトを作成
		room = pra.ShoeBox(room_dim, fs=self.fs, absorption=absorption)

		# 音源とマイクを配置
		mic_pos = np.array([[3.5], [3.5], [1.5]])
		source_pos_speech = np.array([2, 3, 1.5])
		source_pos_noise = np.array([5, 4, 1.5])

		room.add_microphone(mic_pos)
		room.add_source(source_pos_speech, signal=clean_speech)
		room.add_source(source_pos_noise, signal=noise)

		# クリーンな音声（教師信号）を保存
		sf.write(os.path.join(self.output_dir, 'clean', f'sample_{sample_id:05d}.wav'), clean_speech, self.fs)

		# reverberant_only を作成
		room.sources[1].set_gain(0)  # ノイズ源をミュート
		room.simulate()
		reverbe_only = room.mic_array.signals[0]
		sf.write(os.path.join(self.output_dir, 'reverbe_only', f'sample_{sample_id:05d}.wav'), reverbe_only, self.fs)

		# noise_only を作成
		room.sources[0].set_gain(0)  # 音源をミュート
		room.sources[1].set_gain(1.0)  # ノイズ源のゲインを元に戻す
		room.simulate()
		noise_only = room.mic_array.signals[0]
		sf.write(os.path.join(self.output_dir, 'noise_only', f'sample_{sample_id:05d}.wav'), noise_only, self.fs)

		# noise_reverb を作成
		room.sources[0].set_gain(1.0)  # 音源のゲインを元に戻す
		room.simulate()
		noise_reverb = room.mic_array.signals[0]
		sf.write(os.path.join(self.output_dir, 'noise_reverb', f'sample_{sample_id:05d}.wav'), noise_reverb, self.fs)

	def run(self):
		"""
		データセット生成のメイン処理
		"""
		# クリーンな音声ファイルとノイズファイルのリストを準備
		clean_files = [os.path.join(self.config['clean_dir'], f) for f in os.listdir(self.config['clean_dir']) if
					   f.endswith('.wav')]
		noise_files = [os.path.join(self.config['noise_dir'], f) for f in os.listdir(self.config['noise_dir']) if
					   f.endswith('.wav')]

		for i in range(self.config.get('num_samples', 100)):
			clean_file = random.choice(clean_files)
			noise_file = random.choice(noise_files)
			print(f"Generating sample {i + 1}...")
			self.generate_single_sample(clean_file, noise_file, i)

		print("Dataset generation complete.")


if __name__ == '__main__':
	# 設定を辞書で定義
	generator_config = {
		'clean_dir': 'path/to/clean_speech_data',  # クリーンな音声データのパス
		'noise_dir': 'path/to/noise_data',  # ノイズデータのパス
		'num_samples': 100,
	}

# クラスをインスタンス化して実行
# `path/to/clean_speech_data` と `path/to/noise_data` は実際のパスに置き換えてください
# generator = DatasetGenerator(generator_config)
# generator.run()