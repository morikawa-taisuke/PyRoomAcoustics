"""
音源クラス群

pyroomacousticsを用いた音響シミュレーションで使用する音源の
管理と処理を行うクラス群です。音声読み込み、SNR調整、座標計算に対応しています。
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import wave
import os
import math
import random
import logging
from dataclasses import dataclass
from pathlib import Path

# 設定クラスのインポート（相対インポートを想定）
try:
	from config.config_classes import SourceConfig
except ImportError:
	# 開発時の代替
	SourceConfig = None


# カスタム例外クラス
class AudioFileError(IOError):
	"""音声ファイルエラー"""
	pass


class AudioFormatError(ValueError):
	"""音声フォーマットエラー"""
	pass


class SoundSourceConfigError(ValueError):
	"""音源設定エラー"""
	pass


class PositionCalculationError(ValueError):
	"""位置計算エラー"""
	pass


@dataclass
class AudioInfo:
	"""音声情報を格納するデータクラス"""
	file_path: str
	duration: float
	sample_rate: int
	channels: int
	bit_depth: int
	samples: int


@dataclass
class SourceInfo:
	"""音源情報を格納するデータクラス"""
	source_id: str
	source_type: str  # "target" or "noise"
	position: np.ndarray
	audio_info: AudioInfo
	snr_db: Optional[float] = None


class SoundSource:
	"""
	音源クラス

	音声ファイルの読み込み、位置設定、SNR調整などの
	音源に関する処理を統合して管理します。
	"""

	def __init__(self, source_id: str = "default", source_type: str = "target"):
		"""
		音源の初期化

		Args:
			source_id: 音源識別子
			source_type: 音源タイプ ("target" または "noise")
		"""
		self._source_id = source_id
		self._source_type = source_type
		self._file_path: Optional[str] = None
		self._audio_data: Optional[np.ndarray] = None
		self._original_audio_data: Optional[np.ndarray] = None  # 元データの保持
		self._position: Optional[np.ndarray] = None
		self._snr_db: Optional[float] = None
		self._audio_info: Optional[AudioInfo] = None
		self._logger = logging.getLogger(f"{self.__class__.__name__}_{source_id}")

		# 音声処理パラメータ
		self._normalization_factor = 1.0
		self._is_normalized = False

	def load_audio(self, file_path: str, validate_format: bool = True) -> np.ndarray:
		"""
		音声ファイルを読み込み

		既存のload_wave_data関数の機能を移植

		Args:
			file_path: 音声ファイルのパス
			validate_format: フォーマット検証を行うか

		Returns:
			読み込まれた音声データ

		Raises:
			AudioFileError: ファイル読み込みエラー
			AudioFormatError: フォーマットエラー
		"""
		try:
			self._validate_file_path(file_path)

			# WAVファイルを読み込み
			with wave.open(file_path, 'r') as wav_file:
				# 音声情報を取得
				frames = wav_file.getnframes()
				sample_rate = wav_file.getframerate()
				channels = wav_file.getnchannels()
				sample_width = wav_file.getsampwidth()

				# 音声データを読み込み
				raw_data = wav_file.readframes(frames)

				# バイトデータを数値配列に変換
				if sample_width == 2:  # 16bit
					audio_data = np.frombuffer(raw_data, dtype=np.int16)
				elif sample_width == 4:  # 32bit
					audio_data = np.frombuffer(raw_data, dtype=np.int32)
				else:
					raise AudioFormatError(f"未対応のビット深度: {sample_width * 8}bit")

				# モノラルに変換（ステレオの場合）
				if channels > 1:
					audio_data = audio_data.reshape(-1, channels)
					audio_data = np.mean(audio_data, axis=1)
					self._logger.info(f"ステレオからモノラルに変換しました: {file_path}")

				# 浮動小数点に正規化
				if sample_width == 2:
					audio_data = audio_data.astype(np.float64) / np.iinfo(np.int16).max
				elif sample_width == 4:
					audio_data = audio_data.astype(np.float64) / np.iinfo(np.int32).max

			# 音声情報を保存
			self._audio_info = AudioInfo(
				file_path=file_path,
				duration=frames / sample_rate,
				sample_rate=sample_rate,
				channels=1,  # モノラルに変換後
				bit_depth=sample_width * 8,
				samples=len(audio_data)
			)

			# データを保存
			self._file_path = file_path
			self._audio_data = audio_data.copy()
			self._original_audio_data = audio_data.copy()
			self._is_normalized = False

			if validate_format:
				self._validate_audio_format()

			self._logger.info(f"音声ファイルを読み込みました: {file_path}")
			self._logger.info(f"長さ: {len(audio_data)}サンプル ({self._audio_info.duration:.2f}秒)")

			return self._audio_data.copy()

		except wave.Error as e:
			raise AudioFileError(f"WAVファイルの読み込みに失敗: {e}")
		except Exception as e:
			raise AudioFileError(f"音声ファイルの読み込み中にエラーが発生: {e}")

	def normalize_audio(self, method: str = "peak") -> np.ndarray:
		"""
		音声データの正規化

		Args:
			method: 正規化方法 ("peak", "rms", "std")

		Returns:
			正規化された音声データ
		"""
		if self._audio_data is None:
			raise SoundSourceConfigError("音声データが読み込まれていません")

		if method == "peak":
			# ピーク正規化
			max_val = np.max(np.abs(self._audio_data))
			if max_val > 0:
				self._normalization_factor = 1.0 / max_val
				self._audio_data = self._audio_data * self._normalization_factor
		elif method == "rms":
			# RMS正規化
			rms_val = np.sqrt(np.mean(self._audio_data ** 2))
			if rms_val > 0:
				target_rms = 0.1  # ターゲットRMS値
				self._normalization_factor = target_rms / rms_val
				self._audio_data = self._audio_data * self._normalization_factor
		elif method == "std":
			# 標準偏差による正規化（既存実装と互換）
			std_val = np.std(self._audio_data)
			if std_val > 0:
				self._normalization_factor = 1.0 / std_val
				self._audio_data = self._audio_data / std_val
		else:
			raise ValueError(f"未対応の正規化方法: {method}")

		self._is_normalized = True
		self._logger.info(f"音声を正規化しました (方法: {method})")

		return self._audio_data.copy()

	def trim_audio(self, target_length: int, start_position: Optional[int] = None) -> np.ndarray:
		"""
		音声データを指定長に調整

		Args:
			target_length: ターゲット長（サンプル数）
			start_position: 開始位置（Noneの場合はランダム）

		Returns:
			調整後の音声データ
		"""
		if self._audio_data is None:
			raise SoundSourceConfigError("音声データが読み込まれていません")

		current_length = len(self._audio_data)

		if current_length == target_length:
			return self._audio_data.copy()
		elif current_length > target_length:
			# 切り出し処理
			if start_position is None:
				# ランダムな開始位置を選択
				max_start = current_length - target_length
				start_position = random.randint(0, max_start)
			else:
				# 範囲チェック
				if start_position + target_length > current_length:
					raise ValueError("指定された開始位置と長さが音声データの範囲を超えています")

			self._audio_data = self._audio_data[start_position:start_position + target_length]
			self._logger.info(f"音声を切り出しました: {start_position} - {start_position + target_length}")
		else:
			# ゼロパディング
			padding = target_length - current_length
			self._audio_data = np.pad(self._audio_data, (0, padding), mode='constant', constant_values=0)
			self._logger.warning(f"音声が短いためゼロパディングしました: +{padding}サンプル")

		return self._audio_data.copy()

	def apply_random_offset(self, reference_length: int) -> np.ndarray:
		"""
		ランダムなオフセット位置から音声を切り出し

		既存のrec_signal_noise.pyの雑音切り出しロジックを移植

		Args:
			reference_length: 参照となる音声の長さ

		Returns:
			切り出された音声データ
		"""
		if self._audio_data is None:
			raise SoundSourceConfigError("音声データが読み込まれていません")

		current_length = len(self._audio_data)

		if current_length <= reference_length:
			self._logger.warning("音声データが参照長以下です")
			return self.trim_audio(reference_length)

		# ランダムな開始位置を決定
		max_start = current_length - reference_length
		start_position = random.randint(0, max_start)

		return self.trim_audio(reference_length, start_position)

	def set_position_from_coordinates(self, coordinates: Tuple[float, float, float]) -> None:
		"""
		座標から直接位置を設定

		Args:
			coordinates: 3D座標 (x, y, z)
		"""
		self._position = np.array(coordinates, dtype=float)
		self._logger.info(f"音源位置を設定: {coordinates}")

	def set_position_from_angles(self, elevation: float, azimuth: float, distance: float, mic_center: Tuple[float, float, float]) -> None:
		"""
		角度と距離から音源位置を計算して設定

		既存のset_sources_coordinate2関数の機能を移植

		Args:
			elevation: 仰角（ラジアン）
			azimuth: 方位角（ラジアン）
			distance: 距離
			mic_center: マイクアレイ中心座標
		"""
		try:
			# 球面座標から直交座標への変換
			x = np.cos(azimuth) * np.sin(elevation)
			y = np.sin(azimuth) * np.sin(elevation)
			z = np.cos(elevation)

			# 距離をかけて実際の座標を計算
			relative_position = np.array([x, y, z]) * distance

			# マイク中心からの絶対位置を計算
			mic_center_array = np.array(mic_center)
			self._position = relative_position + mic_center_array

			self._logger.info(f"音源位置を角度から計算: 仰角={np.degrees(elevation):.1f}°, "
			                  f"方位角={np.degrees(azimuth):.1f}°, 距離={distance}m")
			self._logger.info(f"計算された位置: {self._position}")

		except Exception as e:
			raise PositionCalculationError(f"位置計算に失敗: {e}")

	def apply_snr(self, target_audio: np.ndarray, snr_db: float) -> np.ndarray:
		"""
		指定したSNRになるように音声レベルを調整

		既存のget_scale_noise関数の機能を移植

		Args:
			target_audio: 目的音声
			snr_db: 目標SNR（dB）

		Returns:
			SNR調整後の音声データ
		"""
		if self._audio_data is None:
			raise SoundSourceConfigError("音声データが読み込まれていません")

		try:
			# 音声パワーの計算
			target_power = self._calculate_power(target_audio)
			noise_power = self._calculate_power(self._audio_data)

			# SNR計算式: SNR = 10 * log10(P_signal / P_noise)
			# 逆算: P_noise_new = P_signal / (10^(SNR/10))
			snr_linear = 10 ** (snr_db / 10)
			target_noise_power = target_power / snr_linear

			# スケーリング係数の計算
			if noise_power > 0:
				scaling_factor = np.sqrt(target_noise_power / noise_power)
				self._audio_data = self._audio_data * scaling_factor
			else:
				raise SoundSourceConfigError("雑音のパワーが0です")

			# 結果の確認
			actual_snr = self._calculate_snr(target_audio, self._audio_data)
			self._snr_db = actual_snr

			self._logger.info(f"SNR調整完了: 目標={snr_db}dB, 実際={actual_snr:.1f}dB")

			if abs(actual_snr - snr_db) > 0.1:  # 0.1dB以上の差
				self._logger.warning(f"SNR調整に誤差があります: {abs(actual_snr - snr_db):.1f}dB")

			return self._audio_data.copy()

		except Exception as e:
			raise SoundSourceConfigError(f"SNR調整に失敗: {e}")

	def get_scaling_factor(self, target_power: float, snr_db: float) -> float:
		"""
		SNRに基づくスケーリング係数を計算

		Args:
			target_power: 目的音声のパワー
			snr_db: 目標SNR

		Returns:
			スケーリング係数
		"""
		if self._audio_data is None:
			raise SoundSourceConfigError("音声データが読み込まれていません")

		noise_power = self._calculate_power(self._audio_data)
		snr_linear = 10 ** (snr_db / 10)
		target_noise_power = target_power / snr_linear

		if noise_power > 0:
			return np.sqrt(target_noise_power / noise_power)
		else:
			raise SoundSourceConfigError("雑音のパワーが0です")

	def _calculate_power(self, audio_data: np.ndarray) -> float:
		"""
		音声データのパワーを計算

		既存のget_wave_power関数の機能を移植
		"""
		return np.sum(audio_data ** 2)

	def _calculate_snr(self, target_audio: np.ndarray, noise_audio: np.ndarray) -> float:
		"""
		SNRを計算

		既存のget_snr関数の機能を移植
		"""
		target_power = self._calculate_power(target_audio)
		noise_power = self._calculate_power(noise_audio)

		if noise_power > 0:
			return 10 * math.log10(target_power / noise_power)
		else:
			return float('inf')

	def get_audio_data(self) -> Optional[np.ndarray]:
		"""処理済み音声データを取得"""
		return self._audio_data.copy() if self._audio_data is not None else None

	def get_original_audio_data(self) -> Optional[np.ndarray]:
		"""元の音声データを取得"""
		return self._original_audio_data.copy() if self._original_audio_data is not None else None

	def get_position(self) -> Optional[np.ndarray]:
		"""音源位置を取得"""
		return self._position.copy() if self._position is not None else None

	def get_source_info(self) -> SourceInfo:
		"""音源情報を取得"""
		if self._audio_info is None:
			raise SoundSourceConfigError("音声データが読み込まれていません")

		return SourceInfo(
			source_id=self._source_id,
			source_type=self._source_type,
			position=self._position.copy() if self._position is not None else np.array([0, 0, 0]),
			audio_info=self._audio_info,
			snr_db=self._snr_db
		)

	def get_audio_statistics(self) -> Dict[str, Any]:
		"""音声統計情報を取得"""
		if self._audio_data is None:
			return {}

		return {
			'length': len(self._audio_data),
			'duration': len(self._audio_data) / (self._audio_info.sample_rate if self._audio_info else 16000),
			'min_value': float(np.min(self._audio_data)),
			'max_value': float(np.max(self._audio_data)),
			'mean_value': float(np.mean(self._audio_data)),
			'std_value': float(np.std(self._audio_data)),
			'rms_value': float(np.sqrt(np.mean(self._audio_data ** 2))),
			'power': float(self._calculate_power(self._audio_data)),
			'is_normalized': self._is_normalized,
			'normalization_factor': self._normalization_factor
		}

	def validate_audio_format(self) -> bool:
		"""音声フォーマットの検証"""
		if self._audio_info is None:
			return False

		try:
			self._validate_audio_format()
			return True
		except AudioFormatError:
			return False

	def reset_to_original(self) -> None:
		"""元の音声データにリセット"""
		if self._original_audio_data is not None:
			self._audio_data = self._original_audio_data.copy()
			self._is_normalized = False
			self._normalization_factor = 1.0
			self._snr_db = None
			self._logger.info("音声データを元の状態にリセットしました")

	def _validate_file_path(self, file_path: str) -> None:
		"""ファイルパスの検証"""
		if not os.path.exists(file_path):
			raise AudioFileError(f"ファイルが存在しません: {file_path}")

		if not file_path.lower().endswith('.wav'):
			raise AudioFormatError(f"WAVファイル以外は対応していません: {file_path}")

	def _validate_audio_format(self) -> None:
		"""音声フォーマットの検証"""
		if self._audio_info is None:
			raise AudioFormatError("音声情報が設定されていません")

		# サンプリングレートの確認
		if self._audio_info.sample_rate < 8000:
			raise AudioFormatError(f"サンプリングレートが低すぎます: {self._audio_info.sample_rate}Hz")

		# 音声長の確認
		if self._audio_info.samples == 0:
			raise AudioFormatError("音声データが空です")

	# プロパティ
	@property
	def source_id(self) -> str:
		"""音源識別子"""
		return self._source_id

	@property
	def source_type(self) -> str:
		"""音源タイプ"""
		return self._source_type

	@property
	def file_path(self) -> Optional[str]:
		"""ファイルパス"""
		return self._file_path

	@property
	def is_loaded(self) -> bool:
		"""音声データが読み込まれているか"""
		return self._audio_data is not None

	@property
	def has_position(self) -> bool:
		"""位置が設定されているか"""
		return self._position is not None


class MultiSoundSourceManager:
	"""
	複数音源の管理クラス
	"""

	def __init__(self):
		self._sources: Dict[str, SoundSource] = {}
		self._logger = logging.getLogger(self.__class__.__name__)

	def add_source(self, source: SoundSource) -> None:
		"""音源を追加"""
		self._sources[source.source_id] = source
		self._logger.info(f"音源を追加: {source.source_id} ({source.source_type})")

	def remove_source(self, source_id: str) -> bool:
		"""音源を削除"""
		if source_id in self._sources:
			del self._sources[source_id]
			self._logger.info(f"音源を削除: {source_id}")
			return True
		return False

	def get_source(self, source_id: str) -> Optional[SoundSource]:
		"""音源を取得"""
		return self._sources.get(source_id)

	def get_all_sources(self) -> List[SoundSource]:
		"""全音源を取得"""
		return list(self._sources.values())

	def get_sources_by_type(self, source_type: str) -> List[SoundSource]:
		"""タイプ別音源を取得"""
		return [source for source in self._sources.values() if source.source_type == source_type]

	def get_all_positions(self) -> List[np.ndarray]:
		"""全音源の位置を取得"""
		positions = []
		for source in self._sources.values():
			if source.has_position:
				positions.append(source.get_position())
		return positions

	def apply_snr_to_noise_sources(self, target_source: SoundSource, snr_db: float) -> None:
		"""雑音源にSNRを適用"""
		target_audio = target_source.get_audio_data()
		if target_audio is None:
			raise SoundSourceConfigError("目的音源のデータがありません")

		noise_sources = self.get_sources_by_type("noise")
		for noise_source in noise_sources:
			if noise_source.is_loaded:
				noise_source.apply_snr(target_audio, snr_db)

	def validate_all_sources(self) -> bool:
		"""全音源の検証"""
		all_valid = True
		for source in self._sources.values():
			if source.is_loaded and not source.validate_audio_format():
				all_valid = False
				self._logger.error(f"音源の検証に失敗: {source.source_id}")
		return all_valid

	def get_manager_statistics(self) -> Dict[str, Any]:
		"""管理統計情報を取得"""
		total_sources = len(self._sources)
		loaded_sources = sum(1 for source in self._sources.values() if source.is_loaded)
		positioned_sources = sum(1 for source in self._sources.values() if source.has_position)

		by_type = {}
		for source in self._sources.values():
			source_type = source.source_type
			if source_type not in by_type:
				by_type[source_type] = 0
			by_type[source_type] += 1

		return {
			'total_sources': total_sources,
			'loaded_sources': loaded_sources,
			'positioned_sources': positioned_sources,
			'sources_by_type': by_type,
			'source_ids': list(self._sources.keys())
		}


# ファクトリー関数群
class SoundSourceFactory:
	"""音源生成のためのファクトリークラス"""

	@staticmethod
	def create_target_source(file_path: str, position: Optional[Tuple[float, float, float]] = None,
	                         source_id: str = "target") -> SoundSource:
		"""目的音源を生成"""
		source = SoundSource(source_id=source_id, source_type="target")
		source.load_audio(file_path)
		source.normalize_audio(method="std")  # 既存実装と同様の正規化

		if position:
			source.set_position_from_coordinates(position)

		return source

	@staticmethod
	def create_noise_source(file_path: str, position: Optional[Tuple[float, float, float]] = None,
	                        snr_db: Optional[float] = None, target_audio: Optional[np.ndarray] = None,
	                        source_id: str = "noise") -> SoundSource:
		"""雑音源を生成"""
		source = SoundSource(source_id=source_id, source_type="noise")
		source.load_audio(file_path)
		source.normalize_audio(method="std")  # 既存実装と同様の正規化

		if position:
			source.set_position_from_coordinates(position)

		if snr_db is not None and target_audio is not None:
			source.apply_snr(target_audio, snr_db)

		return source

	@staticmethod
	def create_from_config(source_config: 'SourceConfig') -> Tuple[SoundSource, List[SoundSource]]:
		"""
		SourceConfigから音源を生成

		Returns:
			(目的音源, 雑音源リスト)
		"""
		if source_config is None:
			raise SoundSourceConfigError("設定が指定されていません")

		# 目的音源の作成（ディレクトリから最初のファイルを使用）
		target_files = []
		if os.path.isdir(source_config.target_directory):
			target_files = [f for f in os.listdir(source_config.target_directory) if f.endswith('.wav')]

		if not target_files:
			raise SoundSourceConfigError(f"目的音源が見つかりません: {source_config.target_directory}")

		target_file = os.path.join(source_config.target_directory, target_files[0])
		target_source = SoundSourceFactory.create_target_source(target_file)

		# 雑音源の作成
		noise_sources = []
		if os.path.isfile(source_config.noise_file):
			for snr in source_config.snr_values:
				noise_source = SoundSourceFactory.create_noise_source(
					source_config.noise_file,
					snr_db=snr,
					target_audio=target_source.get_audio_data(),
					source_id=f"noise_snr{snr}"
				)
				noise_sources.append(noise_source)

		return target_source, noise_sources


# レガシー互換性のための関数
def load_wave_data_compatible(wave_path: str) -> np.ndarray:
	"""
	既存のload_wave_data関数との互換性を保つ関数
	"""
	source = SoundSource()
	return source.load_audio(wave_path)


def get_scale_noise_compatible(signal_data: np.ndarray, noise_data: np.ndarray, snr_db: float) -> np.ndarray:
	"""
	既存のget_scale_noise関数との互換性を保つ関数
	"""
	# 一時的な音源オブジェクトを作成
	noise_source = SoundSource()
	noise_source._audio_data = noise_data.copy()

	# SNRを適用
	return noise_source.apply_snr(signal_data, snr_db)


def set_sources_coordinate2_compatible(doas: np.ndarray, distances: List[float],
                                       mic_center: np.ndarray) -> np.ndarray:
	"""
	既存のset_sources_coordinate2関数との互換性を保つ関数
	"""
	num_sources = doas.shape[0]
	coordinates = np.zeros((3, num_sources))

	for i in range(num_sources):
		elevation = doas[i, 0]
		azimuth = doas[i, 1]
		distance = distances[i]

		# 一時的な音源オブジェクトを作成
		temp_source = SoundSource()
		temp_source.set_position_from_angles(elevation, azimuth, distance, tuple(mic_center))
		coordinates[:, i] = temp_source.get_position()

	return coordinates


# テスト支援関数
def generate_test_sine_wave(frequency: float = 440.0, duration: float = 1.0,
                            sample_rate: int = 16000, amplitude: float = 0.5) -> np.ndarray:
	"""テスト用正弦波を生成"""
	t = np.linspace(0, duration, int(sample_rate * duration), False)
	return amplitude * np.sin(2 * np.pi * frequency * t)


def generate_white_noise(duration: float = 1.0, sample_rate: int = 16000,
                         amplitude: float = 0.1) -> np.ndarray:
	"""ホワイトノイズを生成"""
	samples = int(sample_rate * duration)
	return amplitude * np.random.normal(0, 1, samples)


def create_test_audio_file(file_path: str, audio_data: np.ndarray, sample_rate: int = 16000) -> None:
	"""テスト音声ファイルを作成"""
	# 16bit整数に変換
	audio_int16 = (audio_data * np.iinfo(np.int16).max).astype(np.int16)

	with wave.open(file_path, 'w') as wav_file:
		wav_file.setnchannels(1)  # モノラル
		wav_file.setsampwidth(2)  # 16bit
		wav_file.setframerate(sample_rate)
		wav_file.writeframes(audio_int16.tobytes())


def create_test_source() -> SoundSource:
	"""テスト用音源を作成"""
	# テスト音声データを生成
	test_data = generate_test_sine_wave(frequency=440, duration=2.0)

	source = SoundSource(source_id="test", source_type="target")
	source._audio_data = test_data
	source._original_audio_data = test_data.copy()
	source._audio_info = AudioInfo(
		file_path="test_sine.wav",
		duration=2.0,
		sample_rate=16000,
		channels=1,
		bit_depth=16,
		samples=len(test_data)
	)

	return source


# モジュール公開API
__all__ = [
	'SoundSource', 'MultiSoundSourceManager', 'SoundSourceFactory',
	'AudioFileError', 'AudioFormatError', 'SoundSourceConfigError', 'PositionCalculationError',
	'AudioInfo', 'SourceInfo',
	'load_wave_data_compatible', 'get_scale_noise_compatible', 'set_sources_coordinate2_compatible',
	'generate_test_sine_wave', 'generate_white_noise', 'create_test_audio_file', 'create_test_source'
]

# ロギング設定
logging.basicConfig(level=logging.INFO)