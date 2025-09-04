"""
マイクロホンアレイクラス群

pyroomacousticsを用いた音響シミュレーションで使用するマイクロホンアレイの
座標計算と管理を行うクラス群です。線形アレイと円形アレイに対応しています。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import math
import logging
from dataclasses import dataclass

# 設定クラスのインポート（相対インポートを想定）
try:
	from config.config_classes import MicArrayConfig
except ImportError:
	# 開発時の代替
	MicArrayConfig = None


# カスタム例外クラス
class MicArrayConfigError(ValueError):
	"""マイクアレイ設定エラー"""
	pass


class ArrayGeometryError(ValueError):
	"""アレイ幾何学エラー"""
	pass


class PhysicalConstraintError(ValueError):
	"""物理制約違反エラー"""
	pass


@dataclass
class ArrayInfo:
	"""アレイ情報を格納するデータクラス"""
	array_type: str
	num_channels: int
	total_span: float  # 全体の幅または直径
	center_position: Tuple[float, float, float]
	coordinates: np.ndarray


class MicrophoneArray(ABC):
	"""
	マイクロホンアレイの抽象基底クラス

	線形アレイ、円形アレイなどの具体的な実装の共通インターフェースを定義します。
	"""

	def __init__(self, center: Tuple[float, float, float], num_channels: int,
	             spacing: float, orientation: float = 0.0):
		"""
		マイクロホンアレイの初期化

		Args:
			center: アレイ中心座標 (x, y, z)
			num_channels: マイク数
			spacing: マイク間隔または半径
			orientation: アレイの向き（ラジアン）
		"""
		self._center = np.array(center, dtype=float)
		self._num_channels = num_channels
		self._spacing = spacing
		self._orientation = orientation
		self._coordinates = None  # キャッシュ用
		self._logger = logging.getLogger(self.__class__.__name__)

		# 基本的な検証
		self._validate_basic_parameters()

	@abstractmethod
	def get_coordinates(self) -> np.ndarray:
		"""
		マイクロホン座標を計算して返す

		Returns:
			座標配列 (3, num_channels) または (2, num_channels)
		"""
		pass

	@abstractmethod
	def _calculate_coordinates(self) -> np.ndarray:
		"""座標計算の実装（各サブクラスで実装）"""
		pass

	def validate_configuration(self) -> bool:
		"""
		設定の妥当性を検証

		Returns:
			検証結果
		"""
		try:
			self._validate_basic_parameters()
			self._validate_geometric_constraints()
			self._validate_physical_constraints()
			return True
		except Exception as e:
			self._logger.error(f"設定検証エラー: {e}")
			return False

	def get_array_info(self) -> ArrayInfo:
		"""アレイ情報を取得"""
		coordinates = self.get_coordinates()
		return ArrayInfo(
			array_type=self.__class__.__name__.replace('Array', '').lower(),
			num_channels=self._num_channels,
			total_span=self._calculate_total_span(),
			center_position=tuple(self._center),
			coordinates=coordinates
		)

	def invalidate_cache(self) -> None:
		"""座標キャッシュを無効化"""
		self._coordinates = None

	def update_center(self, new_center: Tuple[float, float, float]) -> None:
		"""アレイ中心位置を更新"""
		self._center = np.array(new_center, dtype=float)
		self.invalidate_cache()

	def update_orientation(self, new_orientation: float) -> None:
		"""アレイの向きを更新"""
		self._orientation = new_orientation
		self.invalidate_cache()

	def update_spacing(self, new_spacing: float) -> None:
		"""間隔を更新"""
		if new_spacing <= 0:
			raise MicArrayConfigError("間隔は正の値である必要があります")
		self._spacing = new_spacing
		self.invalidate_cache()

	@abstractmethod
	def _calculate_total_span(self) -> float:
		"""アレイ全体の幅/直径を計算"""
		pass

	def _validate_basic_parameters(self) -> None:
		"""基本パラメータの検証"""
		if self._num_channels < 1:
			raise MicArrayConfigError("マイク数は1以上である必要があります")
		if self._num_channels > 64:  # 実用的な上限
			raise MicArrayConfigError("マイク数が多すぎます（最大64チャンネル）")
		if self._spacing <= 0:
			raise MicArrayConfigError("間隔は正の値である必要があります")
		if self._spacing > 10.0:  # 実用的な上限
			raise MicArrayConfigError("間隔が大きすぎます（最大10m）")
		if len(self._center) != 3:
			raise MicArrayConfigError("中心位置は3次元座標で指定してください")

	def _validate_geometric_constraints(self) -> None:
		"""幾何学的制約の検証"""
		# サブクラスで必要に応じてオーバーライド
		pass

	def _validate_physical_constraints(self) -> None:
		"""物理的制約の検証"""
		if any(pos < 0 for pos in self._center):
			self._logger.warning("負の座標が指定されています")

	def _apply_rotation_matrix(self, coordinates: np.ndarray) -> np.ndarray:
		"""
		座標に回転行列を適用

		Args:
			coordinates: 回転前の座標

		Returns:
			回転後の座標
		"""
		if abs(self._orientation) < 1e-10:  # 回転が不要
			return coordinates

		cos_theta = np.cos(self._orientation)
		sin_theta = np.sin(self._orientation)

		# 2D回転行列（z軸周り）
		rotation_matrix = np.array([
			[cos_theta, -sin_theta],
			[sin_theta, cos_theta]
		])

		# 2D座標に回転を適用
		if coordinates.shape[0] >= 2:
			rotated_xy = rotation_matrix @ coordinates[:2, :]
			if coordinates.shape[0] == 3:
				return np.vstack([rotated_xy, coordinates[2:3, :]])
			else:
				return rotated_xy

		return coordinates

	# プロパティ
	@property
	def center(self) -> np.ndarray:
		"""アレイ中心座標"""
		return self._center.copy()

	@property
	def num_channels(self) -> int:
		"""マイク数"""
		return self._num_channels

	@property
	def spacing(self) -> float:
		"""間隔"""
		return self._spacing

	@property
	def orientation(self) -> float:
		"""向き"""
		return self._orientation


class LinearArray(MicrophoneArray):
	"""
	線形マイクロホンアレイクラス

	マイクロホンを一直線上に等間隔で配置します。
	既存のset_mic_coordinate関数の機能を移植しています。
	"""

	def get_coordinates(self) -> np.ndarray:
		"""線形アレイの座標を取得"""
		if self._coordinates is None:
			self._coordinates = self._calculate_coordinates()
		return self._coordinates.copy()

	def _calculate_coordinates(self) -> np.ndarray:
		"""
		線形アレイの座標を計算

		既存のset_mic_coordinate関数のロジックを移植
		"""
		# マイクの相対位置を計算（中心からのオフセット）
		if self._num_channels == 1:
			# 単一マイクの場合
			offsets = np.array([0.0])
		else:
			# 複数マイクの場合：中心を基準に対称配置
			offsets = np.array([
				self._spacing * (i - (self._num_channels - 1) / 2)
				for i in range(self._num_channels)
			])

		# 2D/3D判定
		if len(self._center) == 3:
			# 3D座標の場合
			mic_positions = np.zeros((3, self._num_channels))
			mic_positions[0, :] = offsets  # X軸方向に配置
			mic_positions[1, :] = 0.0  # Y軸方向はオフセットなし
			mic_positions[2, :] = 0.0  # Z軸方向はオフセットなし
		else:
			# 2D座標の場合
			mic_positions = np.zeros((2, self._num_channels))
			mic_positions[0, :] = offsets  # X軸方向に配置
			mic_positions[1, :] = 0.0  # Y軸方向はオフセットなし

		# 回転を適用
		rotated_positions = self._apply_rotation_matrix(mic_positions)

		# 中心位置を加算
		final_positions = rotated_positions + self._center[:rotated_positions.shape[0], np.newaxis]

		return final_positions

	def _calculate_total_span(self) -> float:
		"""線形アレイの全幅を計算"""
		if self._num_channels <= 1:
			return 0.0
		return self._spacing * (self._num_channels - 1)

	def _validate_geometric_constraints(self) -> None:
		"""線形アレイ特有の制約検証"""
		super()._validate_geometric_constraints()

		# 単一マイクの場合は間隔は意味がない
		if self._num_channels == 1 and self._spacing > 0:
			self._logger.info("単一マイクの場合、間隔設定は無視されます")


class CircularArray(MicrophoneArray):
	"""
	円形マイクロホンアレイクラス

	マイクロホンを円周上に等間隔で配置します。
	既存のset_circular_mic_coordinate関数の機能を移植しています。
	"""

	def __init__(self, center: Tuple[float, float, float], num_channels: int,
	             radius: float, orientation: float = 0.0, rotate: bool = False):
		"""
		円形アレイの初期化

		Args:
			center: アレイ中心座標
			num_channels: マイク数
			radius: 半径
			orientation: 基準角度
			rotate: 45度回転するかどうか
		"""
		super().__init__(center, num_channels, radius, orientation)
		self._radius = radius
		self._rotate = rotate

	def get_coordinates(self) -> np.ndarray:
		"""円形アレイの座標を取得"""
		if self._coordinates is None:
			self._coordinates = self._calculate_coordinates()
		return self._coordinates.copy()

	def _calculate_coordinates(self) -> np.ndarray:
		"""
		円形アレイの座標を計算

		既存のset_circular_mic_coordinate関数のロジックを移植
		"""
		# 基本角度の計算
		if not self._rotate:
			# 回転なし：0度から等間隔
			base_angles = np.linspace(0, 2 * np.pi, self._num_channels, endpoint=False)
		else:
			# 45度回転
			base_angles = np.linspace(np.pi / 4, 2 * np.pi + np.pi / 4, self._num_channels, endpoint=False)

		# 追加の向き調整を適用
		angles = base_angles + self._orientation

		# 座標計算
		if len(self._center) == 3:
			# 3D座標の場合
			x_points = self._center[0] + self._radius * np.cos(angles)
			y_points = self._center[1] + self._radius * np.sin(angles)
			z_points = np.full(self._num_channels, self._center[2])
			coordinates = np.array([x_points, y_points, z_points])
		else:
			# 2D座標の場合
			x_points = self._center[0] + self._radius * np.cos(angles)
			y_points = self._center[1] + self._radius * np.sin(angles)
			coordinates = np.array([x_points, y_points])

		return coordinates

	def _calculate_total_span(self) -> float:
		"""円形アレイの直径を計算"""
		return 2.0 * self._radius

	def _validate_geometric_constraints(self) -> None:
		"""円形アレイ特有の制約検証"""
		super()._validate_geometric_constraints()

		if self._num_channels < 3:
			raise ArrayGeometryError("円形アレイは最低3つのマイクが必要です")

		# マイク間の最小角度をチェック
		min_angle = 2 * np.pi / self._num_channels
		if min_angle < np.pi / 18:  # 10度未満
			self._logger.warning(f"マイク間の角度が小さすぎる可能性があります: {np.degrees(min_angle):.1f}度")

	@property
	def radius(self) -> float:
		"""半径"""
		return self._radius

	@property
	def rotate(self) -> bool:
		"""回転フラグ"""
		return self._rotate

	def update_radius(self, new_radius: float) -> None:
		"""半径を更新"""
		if new_radius <= 0:
			raise MicArrayConfigError("半径は正の値である必要があります")
		self._radius = new_radius
		self._spacing = new_radius  # 親クラスの_spacingも更新
		self.invalidate_cache()


class MicrophoneArrayFactory:
	"""
	マイクロホンアレイの生成を行うファクトリークラス
	"""

	@staticmethod
	def create_from_config(config: 'MicArrayConfig') -> MicrophoneArray:
		"""
		設定からマイクロホンアレイを生成

		Args:
			config: マイクアレイ設定

		Returns:
			生成されたアレイオブジェクト
		"""
		if config is None:
			raise MicArrayConfigError("設定が指定されていません")

		array_type = config.array_type.lower()

		if array_type == "linear":
			return LinearArray(
				center=config.center_position,
				num_channels=config.num_channels,
				spacing=config.spacing,
				orientation=config.orientation
			)
		elif array_type == "circular":
			return CircularArray(
				center=config.center_position,
				num_channels=config.num_channels,
				radius=config.spacing,  # 円形アレイでは spacing が半径
				orientation=config.orientation,
				rotate=config.rotate
			)
		else:
			raise MicArrayConfigError(f"未対応のアレイタイプ: {array_type}")

	@staticmethod
	def create_stereo_array(center: Tuple[float, float, float] = (0, 0, 0),
	                        spacing: float = 0.1) -> LinearArray:
		"""ステレオアレイを生成"""
		return LinearArray(center=center, num_channels=2, spacing=spacing)

	@staticmethod
	def create_linear_array(num_channels: int, spacing: float,
	                        center: Tuple[float, float, float] = (0, 0, 0)) -> LinearArray:
		"""線形アレイを生成"""
		return LinearArray(center=center, num_channels=num_channels, spacing=spacing)

	@staticmethod
	def create_circular_array(num_channels: int, radius: float,
	                          center: Tuple[float, float, float] = (0, 0, 0),
	                          rotate: bool = False) -> CircularArray:
		"""円形アレイを生成"""
		return CircularArray(center=center, num_channels=num_channels,
		                     radius=radius, rotate=rotate)

	@staticmethod
	def create_uniform_circular_array(num_channels: int, radius: float,
	                                  center: Tuple[float, float, float] = (0, 0, 0)) -> CircularArray:
		"""等間隔円形アレイを生成"""
		return CircularArray(center=center, num_channels=num_channels,
		                     radius=radius, rotate=False)

	@staticmethod
	def from_legacy_linear_params(center: List[float], num_channels: int,
	                              distance: float) -> LinearArray:
		"""
		既存の線形アレイパラメータから生成

		rec_utility.pyのset_mic_coordinate関数の呼び出しと互換
		"""
		return LinearArray(
			center=tuple(center),
			num_channels=num_channels,
			spacing=distance
		)

	@staticmethod
	def from_legacy_circular_params(center: List[float], num_channels: int,
	                                radius: float, rotate: bool = False) -> CircularArray:
		"""
		既存の円形アレイパラメータから生成

		rec_utility.pyのset_circular_mic_coordinate関数の呼び出しと互換
		"""
		return CircularArray(
			center=tuple(center),
			num_channels=num_channels,
			radius=radius,
			rotate=rotate
		)


# レガシー互換性のための関数
def set_mic_coordinate_compatible(center, num_channels: int, distance: float) -> np.ndarray:
	"""
	既存のset_mic_coordinate関数との互換性を保つラッパー関数

	Args:
		center: マイク中心座標
		num_channels: チャンネル数
		distance: マイク間距離

	Returns:
		マイク座標配列
	"""
	array = MicrophoneArrayFactory.from_legacy_linear_params(center, num_channels, distance)
	return array.get_coordinates()


def set_circular_mic_coordinate_compatible(center, num_channels: int,
                                           radius: float, rotate: bool = False) -> np.ndarray:
	"""
	既存のset_circular_mic_coordinate関数との互換性を保つラッパー関数

	Args:
		center: マイク中心座標
		num_channels: チャンネル数
		radius: 半径
		rotate: 回転フラグ

	Returns:
		マイク座標配列
	"""
	array = MicrophoneArrayFactory.from_legacy_circular_params(center, num_channels, radius, rotate)
	return array.get_coordinates()


# デバッグ・テスト支援関数
def create_test_linear_array() -> LinearArray:
	"""テスト用の線形アレイを生成"""
	return MicrophoneArrayFactory.create_linear_array(
		num_channels=4, spacing=0.05, center=(1.0, 1.0, 1.0)
	)


def create_test_circular_array() -> CircularArray:
	"""テスト用の円形アレイを生成"""
	return MicrophoneArrayFactory.create_circular_array(
		num_channels=6, radius=0.1, center=(1.0, 1.0, 1.0)
	)


def compare_with_legacy_implementation(center, num_channels: int, distance: float) -> Dict[str, Any]:
	"""
	既存実装との結果比較

	Returns:
		比較結果の辞書
	"""
	# 新実装
	new_array = MicrophoneArrayFactory.from_legacy_linear_params(center, num_channels, distance)
	new_coords = new_array.get_coordinates()

	# 既存実装（仮想的な呼び出し）
	# legacy_coords = original_set_mic_coordinate(center, num_channels, distance)

	return {
		'new_coordinates': new_coords,
		'array_info': new_array.get_array_info(),
		'total_span': new_array._calculate_total_span(),
		'validation_result': new_array.validate_configuration()
	}


# モジュール公開API
__all__ = [
	'MicrophoneArray', 'LinearArray', 'CircularArray', 'MicrophoneArrayFactory',
	'MicArrayConfigError', 'ArrayGeometryError', 'PhysicalConstraintError',
	'ArrayInfo', 'set_mic_coordinate_compatible', 'set_circular_mic_coordinate_compatible',
	'create_test_linear_array', 'create_test_circular_array', 'compare_with_legacy_implementation'
]

# ロギング設定
logging.basicConfig(level=logging.INFO)