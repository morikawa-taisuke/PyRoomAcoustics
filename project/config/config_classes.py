"""
音響シミュレーション設定クラス群

pyroomacousticsを用いた音響シミュレーションの設定を管理するクラス群です。
各設定項目をタイプセーフに管理し、バリデーション機能を提供します。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import os
import math
import numpy as np
from pathlib import Path
import logging


# カスタム例外クラス
class ConfigValidationError(ValueError):
	"""設定値検証エラー"""
	pass


class ConfigFileError(FileNotFoundError):
	"""設定ファイルエラー"""
	pass


class ConfigCompatibilityError(ValueError):
	"""設定値互換性エラー"""
	pass


# 基底設定クラス
@dataclass(frozen=True)
class BaseConfig(ABC):
	"""設定クラスの基底クラス"""

	def validate(self) -> bool:
		"""設定値の妥当性を検証する"""
		try:
			self._validate_ranges()
			self._validate_dependencies()
			self._validate_file_paths()
			return True
		except ConfigValidationError as e:
			logging.error(f"設定検証エラー: {e}")
			raise

	@abstractmethod
	def _validate_ranges(self) -> None:
		"""数値範囲の検証"""
		pass

	@abstractmethod
	def _validate_dependencies(self) -> None:
		"""依存関係の検証"""
		pass

	def _validate_file_paths(self) -> None:
		"""ファイルパス存在の検証（基底実装は何もしない）"""
		pass

	def to_dict(self) -> Dict[str, Any]:
		"""辞書形式に変換"""
		result = {}
		for key, value in self.__dict__.items():
			if isinstance(value, BaseConfig):
				result[key] = value.to_dict()
			elif isinstance(value, (list, tuple)) and value and isinstance(value[0], BaseConfig):
				result[key] = [item.to_dict() for item in value]
			else:
				result[key] = value
		return result

	def to_json(self, indent: int = 4) -> str:
		"""JSON文字列に変換"""
		return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

	def pretty_print(self) -> None:
		"""整形された設定内容を出力"""
		print(f"\n=== {self.__class__.__name__} ===")
		for key, value in self.__dict__.items():
			if isinstance(value, BaseConfig):
				value.pretty_print()
			else:
				print(f"  {key}: {value}")

	def summary(self) -> str:
		"""重要な設定値のサマリーを返す"""
		return f"{self.__class__.__name__}: {len(self.__dict__)} parameters"


@dataclass(frozen=True)
class RoomConfig(BaseConfig):
	"""部屋の設定を管理するクラス"""

	# 部屋の物理特性
	dimensions: Tuple[float, float, float] = (5.0, 5.0, 5.0)  # [x, y, z] (m)
	reverberation_time: float = 0.5  # 残響時間 (sec)
	absorption: Optional[float] = None  # 吸収係数（自動計算される場合はNone）
	max_order: Optional[int] = None  # 反射回数上限（自動計算される場合はNone）
	sampling_rate: int = 16000  # サンプリングレート (Hz)
	material: str = "default"  # 材質設定

	def _validate_ranges(self) -> None:
		"""数値範囲の検証"""
		# 部屋の寸法チェック
		if any(dim <= 0 for dim in self.dimensions):
			raise ConfigValidationError("部屋の寸法は全て正の値である必要があります")
		if any(dim > 100 for dim in self.dimensions):
			raise ConfigValidationError("部屋の寸法が大きすぎます（最大100m）")

		# 残響時間チェック
		if self.reverberation_time <= 0:
			raise ConfigValidationError("残響時間は正の値である必要があります")
		if self.reverberation_time > 10.0:
			raise ConfigValidationError("残響時間が大きすぎます（最大10秒）")

		# 吸収係数チェック
		if self.absorption is not None:
			if not (0.0 <= self.absorption <= 1.0):
				raise ConfigValidationError("吸収係数は0.0〜1.0の範囲で指定してください")

		# 反射回数チェック
		if self.max_order is not None:
			if self.max_order < 0:
				raise ConfigValidationError("反射回数上限は0以上の値である必要があります")
			if self.max_order > 50:
				raise ConfigValidationError("反射回数上限が大きすぎます（最大50回）")

		# サンプリングレートチェック
		if self.sampling_rate <= 0:
			raise ConfigValidationError("サンプリングレートは正の値である必要があります")
		if self.sampling_rate < 8000:
			raise ConfigValidationError("サンプリングレートが低すぎます（最小8000Hz）")

	def _validate_dependencies(self) -> None:
		"""依存関係の検証"""
		# 特に依存関係は現在なし
		pass

	@property
	def volume(self) -> float:
		"""部屋の体積を計算"""
		return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

	@property
	def surface_area(self) -> float:
		"""部屋の表面積を計算"""
		x, y, z = self.dimensions
		return 2 * (x * y + y * z + z * x)

	def calculate_rt60_parameters(self) -> Tuple[float, int]:
		"""Sabineの残響式から吸収係数と反射回数上限を計算"""
		try:
			import pyroomacoustics as pa
			absorption, max_order = pa.inverse_sabine(self.reverberation_time, np.array(self.dimensions))
			return absorption, max_order
		except ImportError:
			raise ConfigValidationError("pyroomacousticsが必要です")

	@classmethod
	def default(cls) -> RoomConfig:
		"""デフォルト設定を作成"""
		return cls()

	@classmethod
	def from_legacy_params(cls, room_dim: List[float], reverbe_sec: float,
	                       sample_rate: int = 16000) -> RoomConfig:
		"""既存パラメータから設定を作成"""
		return cls(
			dimensions=tuple(room_dim),
			reverberation_time=reverbe_sec,
			sampling_rate=sample_rate
		)


@dataclass(frozen=True)
class MicArrayConfig(BaseConfig):
	"""マイクロホンアレイの設定を管理するクラス"""

	array_type: str = "linear"  # "linear" or "circular"
	num_channels: int = 2  # マイク数
	spacing: float = 0.1  # マイク間隔[m] または 円形アレイの半径[m]
	center_position: Tuple[float, float, float] = (2.5, 2.5, 2.5)  # 中心位置 [x, y, z] (m)
	orientation: float = 0.0  # 向き (ラジアン)
	rotate: bool = False  # 円形アレイの回転設定

	def _validate_ranges(self) -> None:
		"""数値範囲の検証"""
		# アレイタイプチェック
		if self.array_type not in ["linear", "circular"]:
			raise ConfigValidationError("array_typeは'linear'または'circular'である必要があります")

		# マイク数チェック
		if self.num_channels < 1:
			raise ConfigValidationError("マイク数は1以上である必要があります")
		if self.num_channels > 32:
			raise ConfigValidationError("マイク数が多すぎます（最大32チャンネル）")

		# 間隔チェック
		if self.spacing <= 0:
			raise ConfigValidationError("マイク間隔は正の値である必要があります")
		if self.spacing > 5.0:
			raise ConfigValidationError("マイク間隔が大きすぎます（最大5m）")

		# 中心位置チェック
		if any(pos < 0 for pos in self.center_position):
			raise ConfigValidationError("中心位置は全て正の値である必要があります")

		# 向きチェック
		if not (-2 * math.pi <= self.orientation <= 2 * math.pi):
			raise ConfigValidationError("向きは-2π〜2πの範囲で指定してください")

	def _validate_dependencies(self) -> None:
		"""依存関係の検証"""
		# 線形アレイの場合、マイク数が1の時は間隔は意味がない
		if self.array_type == "linear" and self.num_channels == 1 and self.spacing > 0:
			logging.warning("マイク数が1の場合、間隔設定は無視されます")

		# 円形アレイの場合、最小マイク数は3
		if self.array_type == "circular" and self.num_channels < 3:
			raise ConfigValidationError("円形アレイは最低3つのマイクが必要です")

	@classmethod
	def default_linear(cls, num_channels: int = 2) -> MicArrayConfig:
		"""デフォルトの線形アレイ設定を作成"""
		return cls(array_type="linear", num_channels=num_channels)

	@classmethod
	def default_circular(cls, num_channels: int = 4) -> MicArrayConfig:
		"""デフォルトの円形アレイ設定を作成"""
		return cls(array_type="circular", num_channels=num_channels)

	@classmethod
	def from_legacy_params(cls, num_channels: int, distance: float,
	                       mic_center: List[float], array_type: str = "linear") -> MicArrayConfig:
		"""既存パラメータから設定を作成"""
		return cls(
			array_type=array_type,
			num_channels=num_channels,
			spacing=distance,
			center_position=tuple(mic_center)
		)


@dataclass(frozen=True)
class SourceConfig(BaseConfig):
	"""音源の設定を管理するクラス"""

	# 目的音源設定
	target_directory: str = "./data/speech/"  # 目的音源ディレクトリ
	target_distance: float = 0.5  # 目的音源とマイクの距離 (m)
	target_angle: float = math.pi / 2  # 目的音源の角度 (ラジアン)

	# 雑音源設定
	noise_file: str = "./data/noise/hoth.wav"  # 雑音ファイルパス
	noise_distance: float = 0.7  # 雑音源とマイクの距離 (m)
	noise_angle: float = math.pi  # 雑音源の角度 (ラジアン)

	# SNRとその他設定
	snr_values: Tuple[float, ...] = (5.0, 10.0, 15.0)  # SNR設定値のリスト (dB)
	angle_variations: Tuple[float, ...] = (0.0, math.pi / 4, math.pi / 2)  # 角度バリエーション

	def _validate_ranges(self) -> None:
		"""数値範囲の検証"""
		# 距離チェック
		if self.target_distance <= 0:
			raise ConfigValidationError("目的音源の距離は正の値である必要があります")
		if self.noise_distance <= 0:
			raise ConfigValidationError("雑音源の距離は正の値である必要があります")
		if self.target_distance > 50.0 or self.noise_distance > 50.0:
			raise ConfigValidationError("音源の距離が大きすぎます（最大50m）")

		# 角度チェック
		if not (0 <= self.target_angle <= 2 * math.pi):
			raise ConfigValidationError("目的音源の角度は0〜2πの範囲で指定してください")
		if not (0 <= self.noise_angle <= 2 * math.pi):
			raise ConfigValidationError("雑音源の角度は0〜2πの範囲で指定してください")

		# SNRチェック
		if any(snr < -50 or snr > 50 for snr in self.snr_values):
			raise ConfigValidationError("SNRは-50〜50dBの範囲で指定してください")

		# 角度バリエーションチェック
		if any(angle < 0 or angle > 2 * math.pi for angle in self.angle_variations):
			raise ConfigValidationError("角度バリエーションは0〜2πの範囲で指定してください")

	def _validate_dependencies(self) -> None:
		"""依存関係の検証"""
		# 目的音源と雑音源が同じ位置にある場合の警告
		if (abs(self.target_distance - self.noise_distance) < 0.01 and
				abs(self.target_angle - self.noise_angle) < 0.01):
			logging.warning("目的音源と雑音源が同じ位置に設定されています")

	def _validate_file_paths(self) -> None:
		"""ファイルパス存在の検証"""
		# 目的音源ディレクトリの存在チェック
		if not os.path.exists(self.target_directory):
			raise ConfigFileError(f"目的音源ディレクトリが存在しません: {self.target_directory}")

		# 雑音ファイルの存在チェック
		if not os.path.isfile(self.noise_file):
			raise ConfigFileError(f"雑音ファイルが存在しません: {self.noise_file}")

	@classmethod
	def default(cls) -> SourceConfig:
		"""デフォルト設定を作成"""
		return cls()

	@classmethod
	def from_legacy_params(cls, target_dir: str, noise_path: str, snr: float,
	                       target_distance: float = 0.5, noise_distance: float = 0.7,
	                       target_angle: float = math.pi / 2, noise_angle: float = math.pi) -> SourceConfig:
		"""既存パラメータから設定を作成"""
		return cls(
			target_directory=target_dir,
			noise_file=noise_path,
			target_distance=target_distance,
			noise_distance=noise_distance,
			target_angle=target_angle,
			noise_angle=noise_angle,
			snr_values=(snr,)
		)


@dataclass(frozen=True)
class SimulationConfig(BaseConfig):
	"""シミュレーション実行の設定を管理するクラス"""

	sampling_rate: int = 16000  # サンプリングレート (Hz)
	output_types: Tuple[str, ...] = ("clean", "noise", "reverb", "mixed")  # 出力タイプ
	save_split: bool = False  # チャンネル分割保存
	scaling_factor: float = 15.0  # スケーリング係数
	random_seed: Optional[int] = None  # 乱数シード

	def _validate_ranges(self) -> None:
		"""数値範囲の検証"""
		# サンプリングレートチェック
		if self.sampling_rate <= 0:
			raise ConfigValidationError("サンプリングレートは正の値である必要があります")
		if self.sampling_rate < 8000:
			raise ConfigValidationError("サンプリングレートが低すぎます（最小8000Hz）")

		# 出力タイプチェック
		valid_types = {"clean", "noise", "reverb", "mixed"}
		if not all(otype in valid_types for otype in self.output_types):
			raise ConfigValidationError(f"出力タイプは{valid_types}から選択してください")

		# スケーリング係数チェック
		if self.scaling_factor <= 0:
			raise ConfigValidationError("スケーリング係数は正の値である必要があります")
		if self.scaling_factor > 100:
			raise ConfigValidationError("スケーリング係数が大きすぎます（最大100）")

	def _validate_dependencies(self) -> None:
		"""依存関係の検証"""
		# 出力タイプが空の場合
		if not self.output_types:
			raise ConfigValidationError("少なくとも1つの出力タイプを指定してください")

	@classmethod
	def default(cls) -> SimulationConfig:
		"""デフォルト設定を作成"""
		return cls()


@dataclass(frozen=True)
class OutputConfig(BaseConfig):
	"""出力設定を管理するクラス"""

	base_directory: str = "./output/"  # 出力ベースディレクトリ
	naming_convention: str = "{signal}_{noise}_{snr}dB_{rt60}sec"  # ファイル命名規則
	create_subdirectories: bool = True  # サブディレクトリ作成
	overwrite_existing: bool = False  # 既存ファイルの上書き

	def _validate_ranges(self) -> None:
		"""数値範囲の検証"""
		# 特に範囲チェックは不要
		pass

	def _validate_dependencies(self) -> None:
		"""依存関係の検証"""
		# 命名規則の基本チェック
		if not self.naming_convention:
			raise ConfigValidationError("ファイル命名規則を指定してください")

	def _validate_file_paths(self) -> None:
		"""ファイルパス存在の検証"""
		# ベースディレクトリの作成可能性チェック
		try:
			os.makedirs(self.base_directory, exist_ok=True)
		except (OSError, PermissionError) as e:
			raise ConfigFileError(f"出力ディレクトリの作成に失敗しました: {e}")

	def generate_filename(self, signal_name: str, noise_name: str = "",
	                      snr: float = 0.0, rt60: float = 0.0, **kwargs) -> str:
		"""命名規則に基づいてファイル名を生成"""
		filename = self.naming_convention.format(
			signal=signal_name,
			noise=noise_name,
			snr=snr,
			rt60=rt60,
			**kwargs
		)
		return filename + ".wav"

	@classmethod
	def default(cls) -> OutputConfig:
		"""デフォルト設定を作成"""
		return cls()


# 全体設定を統合するクラス
@dataclass(frozen=True)
class AcousticSimulationConfig(BaseConfig):
	"""音響シミュレーションの全体設定を管理するクラス"""

	room: RoomConfig = field(default_factory=RoomConfig.default)
	microphone_array: MicArrayConfig = field(default_factory=MicArrayConfig.default_linear)
	sources: SourceConfig = field(default_factory=SourceConfig.default)
	simulation: SimulationConfig = field(default_factory=SimulationConfig.default)
	output: OutputConfig = field(default_factory=OutputConfig.default)

	def _validate_ranges(self) -> None:
		"""数値範囲の検証"""
		# 各設定クラスの検証を委譲
		self.room.validate()
		self.microphone_array.validate()
		self.sources.validate()
		self.simulation.validate()
		self.output.validate()

	def _validate_dependencies(self) -> None:
		"""依存関係の検証"""
		# サンプリングレートの整合性チェック
		if self.room.sampling_rate != self.simulation.sampling_rate:
			raise ConfigValidationError("部屋とシミュレーションのサンプリングレートが一致していません")

		# マイクアレイの中心位置が部屋内にあるかチェック
		center = self.microphone_array.center_position
		dims = self.room.dimensions
		if not (0 < center[0] < dims[0] and 0 < center[1] < dims[1] and 0 < center[2] < dims[2]):
			raise ConfigValidationError("マイクアレイの中心位置が部屋の範囲外です")

	@classmethod
	def from_json_file(cls, file_path: str) -> AcousticSimulationConfig:
		"""JSONファイルから設定を読み込み"""
		try:
			with open(file_path, 'r', encoding='utf-8') as f:
				data = json.load(f)
			return cls.from_dict(data)
		except (IOError, json.JSONDecodeError) as e:
			raise ConfigFileError(f"設定ファイルの読み込みに失敗しました: {e}")

	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> AcousticSimulationConfig:
		"""辞書から設定を作成"""
		return cls(
			room=RoomConfig(**data.get('room', {})),
			microphone_array=MicArrayConfig(**data.get('microphone_array', {})),
			sources=SourceConfig(**data.get('sources', {})),
			simulation=SimulationConfig(**data.get('simulation', {})),
			output=OutputConfig(**data.get('output', {}))
		)

	def save_to_file(self, file_path: str) -> None:
		"""設定をJSONファイルに保存"""
		try:
			with open(file_path, 'w', encoding='utf-8') as f:
				f.write(self.to_json())
		except IOError as e:
			raise ConfigFileError(f"設定ファイルの保存に失敗しました: {e}")

	@classmethod
	def create_default_config_file(cls, file_path: str) -> None:
		"""デフォルト設定ファイルを作成"""
		default_config = cls()
		default_config.save_to_file(file_path)

	@classmethod
	def from_legacy_recoding2_params(cls, wave_files: List[str], out_dir: str,
	                                 snr: float, reverbe_sec: float, channel: int = 1,
	                                 distance: float = 0, angle: float = math.pi) -> AcousticSimulationConfig:
		"""既存のrecoding2関数のパラメータから設定を作成"""
		return cls(
			room=RoomConfig.from_legacy_params([5.0, 5.0, 5.0], reverbe_sec),
			microphone_array=MicArrayConfig.from_legacy_params(
				channel, distance * 0.01, [2.5, 2.5, 2.5]
			),
			sources=SourceConfig.from_legacy_params(
				os.path.dirname(wave_files[0]) if wave_files else "./data/speech/",
				wave_files[1] if len(wave_files) > 1 else "./data/noise/hoth.wav",
				snr,
				0.5, 0.7, math.pi / 2, angle
			),
			output=OutputConfig(base_directory=out_dir)
		)

	def summary(self) -> str:
		"""設定の要約を返す"""
		return (f"音響シミュレーション設定:\n"
		        f"  部屋: {self.room.dimensions} (RT60: {self.room.reverberation_time}s)\n"
		        f"  マイク: {self.microphone_array.array_type} x {self.microphone_array.num_channels}ch\n"
		        f"  SNR: {self.sources.snr_values}\n"
		        f"  出力: {self.output.base_directory}")


# ユーティリティ関数
def load_config_from_file(file_path: str) -> AcousticSimulationConfig:
	"""設定ファイルを読み込む便利関数"""
	return AcousticSimulationConfig.from_json_file(file_path)


def create_test_config() -> AcousticSimulationConfig:
	"""テスト用の設定を作成"""
	return AcousticSimulationConfig(
		room=RoomConfig(dimensions=(3.0, 3.0, 3.0), reverberation_time=0.3),
		microphone_array=MicArrayConfig(num_channels=1, spacing=0.0),
		sources=SourceConfig(snr_values=(10.0,)),
		simulation=SimulationConfig(output_types=("clean", "mixed"))
	)


# モジュールレベルの設定
__all__ = [
	'ConfigValidationError', 'ConfigFileError', 'ConfigCompatibilityError',
	'RoomConfig', 'MicArrayConfig', 'SourceConfig', 'SimulationConfig', 'OutputConfig',
	'AcousticSimulationConfig', 'load_config_from_file', 'create_test_config'
]