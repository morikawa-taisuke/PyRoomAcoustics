"""
部屋クラス

pyroomacousticsを用いた音響シミュレーションで使用する部屋の
作成・管理を行うクラス群です。残響時間制御、マイクアレイ・音源配置に対応しています。
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum
import gc
import time

# pyroomacousticsのインポート
try:
	import pyroomacoustics as pa
except ImportError:
	pa = None
	logging.warning("pyroomacousticsがインポートできません。模擬モードで動作します。")

# 関連クラスのインポート（相対インポートを想定）
try:
	from project.config.config_classes import RoomConfig
	from models.microphone_array import MicrophoneArray
	from models.sound_source import SoundSource
except ImportError:
	# 開発時の代替
	RoomConfig = None
	MicrophoneArray = None
	SoundSource = None


# カスタム例外クラス
class RoomCreationError(ValueError):
	"""部屋作成エラー"""
	pass


class ReverbCalculationError(ValueError):
	"""残響計算エラー"""
	pass


class SimulationError(RuntimeError):
	"""シミュレーション実行エラー"""
	pass


class RoomConfigError(ValueError):
	"""部屋設定エラー"""
	pass


class RoomConditionType(Enum):
	"""部屋条件タイプ"""
	CLEAN = "clean"  # 清音（無残響・無雑音）
	NOISE = "noise"  # 雑音のみ（無残響）
	REVERB = "reverb"  # 残響のみ（無雑音）
	MIXED = "mixed"  # 残響＋雑音


@dataclass
class RoomCondition:
	"""部屋条件を表すデータクラス"""
	condition_type: RoomConditionType
	absorption_coefficient: float
	max_order: int
	description: str

	@classmethod
	def clean_condition(cls) -> 'RoomCondition':
		"""清音条件を作成"""
		return cls(
			condition_type=RoomConditionType.CLEAN,
			absorption_coefficient=1.0,
			max_order=0,
			description="清音室（残響なし・雑音なし）"
		)

	@classmethod
	def noise_condition(cls) -> 'RoomCondition':
		"""雑音条件を作成"""
		return cls(
			condition_type=RoomConditionType.NOISE,
			absorption_coefficient=1.0,
			max_order=0,
			description="雑音室（残響なし・雑音あり）"
		)

	@classmethod
	def reverb_condition(cls, absorption: float, max_order: int) -> 'RoomCondition':
		"""残響条件を作成"""
		return cls(
			condition_type=RoomConditionType.REVERB,
			absorption_coefficient=absorption,
			max_order=max_order,
			description=f"残響室（RT60制御, 吸収={absorption:.3f}, 反射={max_order}回）"
		)

	@classmethod
	def mixed_condition(cls, absorption: float, max_order: int) -> 'RoomCondition':
		"""混合条件を作成"""
		return cls(
			condition_type=RoomConditionType.MIXED,
			absorption_coefficient=absorption,
			max_order=max_order,
			description=f"混合室（残響あり・雑音あり, 吸収={absorption:.3f}, 反射={max_order}回）"
		)


@dataclass
class RoomStatistics:
	"""部屋統計情報を格納するデータクラス"""
	dimensions: Tuple[float, float, float]
	volume: float
	surface_area: float
	target_rt60: Optional[float]
	actual_rt60: Optional[float]
	absorption_coefficient: float
	max_order: int
	num_microphones: int
	num_sources: int
	condition_type: str
	simulation_ready: bool


class Room:
	"""
	部屋クラス

	pyroomacousticsのShoeBoxクラスをラップし、音響シミュレーション用の
	部屋を管理します。残響時間制御、マイクアレイ・音源配置を統合管理します。
	"""

	def __init__(self, config: Optional[RoomConfig] = None, room_id: str = "default"):
		"""
		部屋の初期化

		Args:
			config: 部屋設定
			room_id: 部屋識別子
		"""
		self._room_id = room_id
		self._config = config
		self._logger = logging.getLogger(f"{self.__class__.__name__}_{room_id}")

		# pyroomacoustics関連
		self._pa_room: Optional[pa.ShoeBox] = None
		self._sample_rate = config.sampling_rate if config else 16000

		# 部屋パラメータ
		self._dimensions = config.dimensions if config else (5.0, 5.0, 5.0)
		self._target_rt60 = config.reverberation_time if config else 0.5
		self._absorption = config.absorption
		self._max_order = config.max_order

		# 配置されたオブジェクト
		self._microphone_arrays: List[MicrophoneArray] = []
		self._sound_sources: List[SoundSource] = []

		# シミュレーション状態
		self._simulation_completed = False
		self._simulation_results: Optional[np.ndarray] = None

		# 残響パラメータキャッシュ
		self._reverb_params_cache: Dict[float, Tuple[float, int]] = {}

		# 部屋条件管理
		self._current_condition: Optional[RoomCondition] = None

		# 統計情報
		self._creation_time: Optional[float] = None
		self._last_simulation_time: Optional[float] = None

		if pa is None:
			self._logger.warning("pyroomacousticsが利用できません。模擬モードで動作します。")

	def create_room(self, config: Optional[RoomConfig] = None,
	                condition: Optional[RoomCondition] = None) -> 'pa.ShoeBox':
		"""
		部屋オブジェクトを作成

		Args:
			config: 部屋設定（Noneの場合は初期設定を使用）
			condition: 部屋条件（Noneの場合は設定から自動決定）

		Returns:
			作成された部屋オブジェクト

		Raises:
			RoomCreationError: 部屋作成エラー
		"""
		if pa is None:
			raise RoomCreationError("pyroomacousticsが利用できません")

		try:
			start_time = time.time()

			# 設定の更新
			if config:
				self._config = config
				self._dimensions = config.dimensions
				self._target_rt60 = config.reverberation_time
				self._sample_rate = config.sampling_rate

			# 条件の決定
			if condition is None:
				condition = self._determine_default_condition()

			self._current_condition = condition

			# 残響パラメータの計算
			if condition.condition_type in [RoomConditionType.REVERB, RoomConditionType.MIXED]:
				if condition.absorption_coefficient is None or condition.max_order is None:
					abs_coeff, max_order = self.calculate_rt60_parameters(self._target_rt60)
					condition.absorption_coefficient = abs_coeff
					condition.max_order = max_order

			# pyroomacoustics部屋オブジェクトの作成
			self._pa_room = pa.ShoeBox(
				room_dim=np.array(self._dimensions),
				fs=self._sample_rate,
				max_order=condition.max_order,
				absorption=condition.absorption_coefficient
			)

			self._creation_time = time.time() - start_time

			self._logger.info(f"部屋を作成しました: {condition.description}")
			self._logger.info(f"寸法: {self._dimensions}")
			self._logger.info(f"作成時間: {self._creation_time:.3f}秒")

			return self._pa_room

		except Exception as e:
			self._logger.error(f"部屋作成に失敗: {e}")
			raise RoomCreationError(f"部屋の作成に失敗しました: {e}")

	def calculate_rt60_parameters(self, target_rt60: float, max_iterations: int = 100,
	                              tolerance: float = 0.01) -> Tuple[float, int]:
		"""
		目標残響時間から吸収係数と反射回数上限を計算

		既存のserch_reverbe_sec関数のロジックを移植

		Args:
			target_rt60: 目標残響時間（秒）
			max_iterations: 最大反復回数
			tolerance: 許容誤差（秒）

		Returns:
			(吸収係数, 反射回数上限)

		Raises:
			ReverbCalculationError: 残響計算エラー
		"""
		if pa is None:
			raise ReverbCalculationError("pyroomacousticsが利用できません")

		# キャッシュをチェック
		cache_key = target_rt60
		if cache_key in self._reverb_params_cache:
			self._logger.info(f"キャッシュから残響パラメータを取得: RT60={target_rt60}s")
			return self._reverb_params_cache[cache_key]

		try:
			self._logger.info(f"残響パラメータを計算中: 目標RT60={target_rt60}s")

			current_rt60 = target_rt60
			iteration = 0
			best_params = None
			best_error = float('inf')

			while iteration < max_iterations:
				try:
					# Sabineの残響式から吸収係数と反射回数を計算
					absorption, max_order = pa.inverse_sabine(current_rt60, np.array(self._dimensions))

					# テスト用の部屋を作成して実際のRT60を測定
					test_room = pa.ShoeBox(
						room_dim=np.array(self._dimensions),
						fs=self._sample_rate,
						max_order=max_order,
						absorption=absorption
					)

					# ダミーのマイクロホンアレイを追加（RT60測定のため）
					center = np.array(self._dimensions) / 2
					mic_locs = center[:, np.newaxis]
					test_room.add_microphone_array(pa.MicrophoneArray(mic_locs, fs=self._sample_rate))

					# ダミー音源を追加
					dummy_signal = np.random.randn(1000)  # 短いダミー信号
					test_room.add_source(center, signal=dummy_signal)

					# シミュレーション実行
					test_room.simulate()

					# RT60測定
					measured_rt60 = test_room.measure_rt60()
					actual_rt60 = np.mean(measured_rt60) if hasattr(measured_rt60, '__len__') else measured_rt60

					error = abs(actual_rt60 - target_rt60)

					self._logger.debug(f"反復{iteration}: 目標={target_rt60:.3f}s, "
					                   f"実測={actual_rt60:.3f}s, 誤差={error:.3f}s")

					# 最良の結果を記録
					if error < best_error:
						best_error = error
						best_params = (absorption, max_order)

					# 収束判定
					if error <= tolerance:
						self._logger.info(f"残響パラメータ計算完了: {iteration + 1}回の反復で収束")
						result = (absorption, max_order)
						break

					# 次の反復のためのRT60調整
					if actual_rt60 < target_rt60:
						current_rt60 *= 1.01  # 少し増加
					else:
						current_rt60 *= 0.99  # 少し減少

				except Exception as e:
					self._logger.warning(f"反復{iteration}でエラー: {e}")

				iteration += 1

			# 収束しなかった場合は最良の結果を使用
			if iteration >= max_iterations:
				if best_params is not None:
					self._logger.warning(f"最大反復回数に達しました。最良の結果を使用: 誤差={best_error:.3f}s")
					result = best_params
				else:
					# fallbackとしてinverse_sabineの結果をそのまま使用
					absorption, max_order = pa.inverse_sabine(target_rt60, np.array(self._dimensions))
					result = (absorption, max_order)
					self._logger.warning("fallbackパラメータを使用します")

			# キャッシュに保存
			self._reverb_params_cache[cache_key] = result

			self._logger.info(f"計算完了: 吸収係数={result[0]:.6f}, 反射回数={result[1]}")
			return result

		except Exception as e:
			self._logger.error(f"残響パラメータ計算に失敗: {e}")
			raise ReverbCalculationError(f"残響パラメータの計算に失敗しました: {e}")

	def search_optimal_parameters(self, target_rt60: float, tolerance: float = 0.01) -> Tuple[float, int]:
		"""
		最適な残響パラメータを探索

		Args:
			target_rt60: 目標残響時間
			tolerance: 許容誤差

		Returns:
			(吸収係数, 反射回数上限)
		"""
		return self.calculate_rt60_parameters(target_rt60, tolerance=tolerance)

	def add_microphones(self, mic_array: MicrophoneArray) -> None:
		"""
		マイクロホンアレイを追加

		Args:
			mic_array: マイクロホンアレイ

		Raises:
			RoomConfigError: 設定エラー
		"""
		if self._pa_room is None:
			raise RoomConfigError("部屋が作成されていません")

		try:
			# マイクロホン座標を取得
			mic_coordinates = mic_array.get_coordinates()

			# 部屋の境界チェック
			self._validate_positions_in_room(mic_coordinates)

			# pyroomacousticsに追加
			mic_array_pa = pa.MicrophoneArray(mic_coordinates, fs=self._sample_rate)
			self._pa_room.add_microphone_array(mic_array_pa)

			# 内部リストに追加
			self._microphone_arrays.append(mic_array)

			self._logger.info(f"マイクロホンアレイを追加: {mic_array.num_channels}チャンネル")
			self._logger.debug(f"マイク座標:\n{mic_coordinates}")

		except Exception as e:
			self._logger.error(f"マイクロホンアレイの追加に失敗: {e}")
			raise RoomConfigError(f"マイクロホンアレイの追加に失敗しました: {e}")

	def add_sources(self, sources: List[SoundSource]) -> None:
		"""
		音源リストを追加

		Args:
			sources: 音源リスト
		"""
		for source in sources:
			self.add_source(source)

	def add_source(self, source: SoundSource) -> None:
		"""
		音源を追加

		Args:
			source: 音源

		Raises:
			RoomConfigError: 設定エラー
		"""
		if self._pa_room is None:
			raise RoomConfigError("部屋が作成されていません")

		try:
			# 音源位置の取得
			position = source.get_position()
			if position is None:
				raise RoomConfigError(f"音源{source.source_id}の位置が設定されていません")

			# 音声データの取得
			audio_data = source.get_audio_data()
			if audio_data is None:
				raise RoomConfigError(f"音源{source.source_id}の音声データがありません")

			# 部屋の境界チェック
			self._validate_positions_in_room(position.reshape(-1, 1))

			# pyroomacousticsに追加
			self._pa_room.add_source(position, signal=audio_data)

			# 内部リストに追加
			self._sound_sources.append(source)

			self._logger.info(f"音源を追加: {source.source_id} ({source.source_type})")
			self._logger.debug(f"音源位置: {position}")

		except Exception as e:
			self._logger.error(f"音源の追加に失敗: {e}")
			raise RoomConfigError(f"音源の追加に失敗しました: {e}")

	def run_simulation(self) -> np.ndarray:
		"""
		シミュレーションを実行

		Returns:
			シミュレーション結果

		Raises:
			SimulationError: シミュレーション実行エラー
		"""
		if not self.is_simulation_ready():
			raise SimulationError("シミュレーションの準備が完了していません")

		try:
			start_time = time.time()

			self._logger.info("シミュレーションを開始します")

			# pyroomacousticsシミュレーション実行
			self._pa_room.simulate()

			# 結果を取得
			self._simulation_results = self._pa_room.mic_array.signals
			self._simulation_completed = True
			self._last_simulation_time = time.time() - start_time

			self._logger.info(f"シミュレーション完了: {self._last_simulation_time:.3f}秒")
			self._logger.info(f"結果の形状: {self._simulation_results.shape}")

			return self._simulation_results.copy()

		except Exception as e:
			self._logger.error(f"シミュレーション実行に失敗: {e}")
			raise SimulationError(f"シミュレーションの実行に失敗しました: {e}")

	def get_simulation_results(self) -> Optional[np.ndarray]:
		"""シミュレーション結果を取得"""
		if self._simulation_results is not None:
			return self._simulation_results.copy()
		return None

	def is_simulation_ready(self) -> bool:
		"""シミュレーション準備完了チェック"""
		return (self._pa_room is not None and
		        len(self._microphone_arrays) > 0 and
		        len(self._sound_sources) > 0)

	def measure_actual_rt60(self) -> Optional[float]:
		"""実際のRT60を測定"""
		if self._pa_room is None or not self._simulation_completed:
			return None

		try:
			rt60_measurements = self._pa_room.measure_rt60()
			if hasattr(rt60_measurements, '__len__'):
				return float(np.mean(rt60_measurements))
			else:
				return float(rt60_measurements)
		except Exception as e:
			self._logger.warning(f"RT60測定に失敗: {e}")
			return None

	def verify_rt60_accuracy(self, tolerance: float = 0.1) -> bool:
		"""RT60精度の検証"""
		if self._target_rt60 is None:
			return True

		actual_rt60 = self.measure_actual_rt60()
		if actual_rt60 is None:
			return False

		error = abs(actual_rt60 - self._target_rt60)
		is_accurate = error <= tolerance

		if is_accurate:
			self._logger.info(f"RT60精度OK: 目標={self._target_rt60:.3f}s, "
			                  f"実測={actual_rt60:.3f}s, 誤差={error:.3f}s")
		else:
			self._logger.warning(f"RT60精度不良: 目標={self._target_rt60:.3f}s, "
			                     f"実測={actual_rt60:.3f}s, 誤差={error:.3f}s")

		return is_accurate

	def reset_simulation(self) -> None:
		"""シミュレーション状態をリセット"""
		self._simulation_completed = False
		self._simulation_results = None
		if self._pa_room is not None:
			# pyroomacousticsの状態もリセット（マイクと音源は保持）
			pass
		self._logger.info("シミュレーション状態をリセットしました")

	def recreate_room(self) -> 'pa.ShoeBox':
		"""既存設定で部屋を再作成"""
		if self._config is None:
			raise RoomConfigError("部屋設定がありません")

		# 既存の配置をクリア
		self.remove_all_microphones()
		self.remove_all_sources()

		return self.create_room(self._config, self._current_condition)

	def remove_all_microphones(self) -> None:
		"""全マイクロホンを削除"""
		self._microphone_arrays.clear()
		self._logger.info("全マイクロホンを削除しました")

	def remove_all_sources(self) -> None:
		"""全音源を削除"""
		self._sound_sources.clear()
		self._logger.info("全音源を削除しました")

	def get_room_statistics(self) -> RoomStatistics:
		"""部屋統計情報を取得"""
		actual_rt60 = self.measure_actual_rt60()

		return RoomStatistics(
			dimensions=self._dimensions,
			volume=np.prod(self._dimensions),
			surface_area=self._calculate_surface_area(),
			target_rt60=self._target_rt60,
			actual_rt60=actual_rt60,
			absorption_coefficient=self._current_condition.absorption_coefficient if self._current_condition else 0.0,
			max_order=self._current_condition.max_order if self._current_condition else 0,
			num_microphones=sum(array.num_channels for array in self._microphone_arrays),
			num_sources=len(self._sound_sources),
			condition_type=self._current_condition.condition_type.value if self._current_condition else "unknown",
			simulation_ready=self.is_simulation_ready()
		)

	def print_room_info(self) -> None:
		"""部屋情報を表示"""
		stats = self.get_room_statistics()

		print(f"\n=== 部屋情報 ({self._room_id}) ===")
		print(f"寸法: {stats.dimensions} m")
		print(f"体積: {stats.volume:.2f} m³")
		print(f"表面積: {stats.surface_area:.2f} m²")
		print(f"条件: {stats.condition_type}")
		print(f"目標RT60: {stats.target_rt60:.3f} s" if stats.target_rt60 else "目標RT60: 未設定")
		print(f"実測RT60: {stats.actual_rt60:.3f} s" if stats.actual_rt60 else "実測RT60: 未測定")
		print(f"吸収係数: {stats.absorption_coefficient:.6f}")
		print(f"反射回数: {stats.max_order}")
		print(f"マイク数: {stats.num_microphones}")
		print(f"音源数: {stats.num_sources}")
		print(f"シミュレーション準備: {'OK' if stats.simulation_ready else 'NG'}")

	def _determine_default_condition(self) -> RoomCondition:
		"""デフォルト条件を決定"""
		if self._target_rt60 and self._target_rt60 > 0.05:
			# 残響時間が設定されている場合は残響室
			abs_coeff, max_order = self.calculate_rt60_parameters(self._target_rt60)
			return RoomCondition.reverb_condition(abs_coeff, max_order)
		else:
			# それ以外は清音室
			return RoomCondition.clean_condition()

	def _validate_positions_in_room(self, positions: np.ndarray) -> None:
		"""位置が部屋内にあるかチェック"""
		if positions.ndim == 1:
			positions = positions.reshape(-1, 1)

		for i, dim in enumerate(self._dimensions):
			pos_min = np.min(positions[i, :])
			pos_max = np.max(positions[i, :])

			if pos_min < 0 or pos_max > dim:
				raise RoomConfigError(
					f"位置が部屋の範囲外です: 軸{i} 範囲[0, {dim}], 実際[{pos_min:.3f}, {pos_max:.3f}]"
				)

	def _calculate_surface_area(self) -> float:
		"""部屋の表面積を計算"""
		x, y, z = self._dimensions
		return 2 * (x * y + y * z + z * x)

	# プロパティ
	@property
	def room_id(self) -> str:
		"""部屋識別子"""
		return self._room_id

	@property
	def dimensions(self) -> Tuple[float, float, float]:
		"""部屋寸法"""
		return self._dimensions

	@property
	def target_rt60(self) -> Optional[float]:
		"""目標残響時間"""
		return self._target_rt60

	@property
	def current_condition(self) -> Optional[RoomCondition]:
		"""現在の部屋条件"""
		return self._current_condition

	@property
	def is_created(self) -> bool:
		"""部屋が作成されているか"""
		return self._pa_room is not None

	@property
	def simulation_completed(self) -> bool:
		"""シミュレーションが完了しているか"""
		return self._simulation_completed

	@property
	def pa_room(self) -> Optional['pa.ShoeBox']:
		"""pyroomacoustics部屋オブジェクト（デバッグ用）"""
		return self._pa_room


class RoomManager:
	"""
	複数部屋の管理クラス

	clean, noise, reverb, mixedの4つの異なる条件の部屋を効率的に管理します。
	"""

	def __init__(self, base_config: Optional[RoomConfig] = None):
		"""
		初期化

		Args:
			base_config: ベース設定
		"""
		self._base_config = base_config
		self._rooms: Dict[str, Room] = {}
		self._logger = logging.getLogger(self.__class__.__name__)

		# 標準的な4つの条件を準備
		self._standard_conditions = {
			"clean": RoomCondition.clean_condition(),
			"noise": RoomCondition.noise_condition(),
			"reverb": None,  # 動的に生成
			"mixed": None  # 動的に生成
		}

	def create_rooms_for_all_conditions(self, config: Optional[RoomConfig] = None) -> Dict[str, Room]:
		"""
		4つの標準条件で部屋を作成

		既存のrecoding2関数の4つの部屋作成ロジックを移植

		Args:
			config: 部屋設定

		Returns:
			条件名をキーとする部屋の辞書
		"""
		if config:
			self._base_config = config

		if not self._base_config:
			raise RoomConfigError("部屋設定が指定されていません")

		try:
			self._logger.info("4つの条件で部屋を作成開始")

			# 残響条件のパラメータを事前計算
			rt60 = self._base_config.reverberation_time
			temp_room = Room(self._base_config, "temp")
			absorption, max_order = temp_room.calculate_rt60_parameters(rt60)

			# 各条件の部屋を作成
			conditions = {
				"clean": RoomCondition.clean_condition(),
				"noise": RoomCondition.noise_condition(),
				"reverb": RoomCondition.reverb_condition(absorption, max_order),
				"mixed": RoomCondition.mixed_condition(absorption, max_order)
			}

			created_rooms = {}
			for condition_name, condition in conditions.items():
				room = Room(self._base_config, condition_name)
				room.create_room(condition=condition)
				self._rooms[condition_name] = room
				created_rooms[condition_name] = room

				self._logger.info(f"{condition_name}用部屋を作成: {condition.description}")

			self._logger.info("全ての部屋の作成が完了しました")
			return created_rooms

		except Exception as e:
			self._logger.error(f"部屋作成に失敗: {e}")
			raise

	def get_room(self, condition_name: str) -> Optional[Room]:
		"""指定条件の部屋を取得"""
		return self._rooms.get(condition_name)

	def get_all_rooms(self) -> Dict[str, Room]:
		"""全部屋を取得"""
		return self._rooms.copy()

	def add_microphones_to_all(self, mic_array: MicrophoneArray) -> None:
		"""全部屋にマイクロホンアレイを追加"""
		for room_name, room in self._rooms.items():
			try:
				room.add_microphones(mic_array)
				self._logger.info(f"{room_name}部屋にマイクを追加しました")
			except Exception as e:
				self._logger.error(f"{room_name}部屋へのマイク追加に失敗: {e}")

	def cleanup_all_rooms(self) -> None:
		"""全部屋をクリーンアップ"""
		for room in self._rooms.values():
			room.reset_simulation()

		# ガベージコレクション
		gc.collect()
		self._logger.info("全部屋をクリーンアップしました")


# レガシー互換性関数
def serch_reverbe_sec_compatible(reverbe_sec: float, channel: int = 1,
                                 angle: float = np.pi) -> Tuple[float, int]:
	"""
	既存のserch_reverbe_sec関数との互換性を保つ関数

	Args:
		reverbe_sec: 残響時間
		channel: チャンネル数（互換性のため、実際は使用しない）
		angle: 角度（互換性のため、実際は使用しない）

	Returns:
		(吸収係数, 反射回数上限)
	"""
	# デフォルト部屋設定で部屋を作成
	from project.config.config_classes import RoomConfig

	config = RoomConfig(
		dimensions=(5.0, 5.0, 5.0),
		reverberation_time=reverbe_sec,
		sampling_rate=16000
	)

	room = Room(config, "legacy_compat")
	return room.calculate_rt60_parameters(reverbe_sec)


def create_room_compatible(room_dim: List[float], rt60: float,
                           sample_rate: int = 16000) -> 'pa.ShoeBox':
	"""
	既存の部屋作成ロジックとの互換性を保つ関数

	Args:
		room_dim: 部屋寸法
		rt60: 残響時間
		sample_rate: サンプリングレート

	Returns:
		pyroomacoustics部屋オブジェクト
	"""
	from project.config.config_classes import RoomConfig

	config = RoomConfig(
		dimensions=tuple(room_dim),
		reverberation_time=rt60,
		sampling_rate=sample_rate
	)

	room = Room(config, "legacy_room")
	return room.create_room()


# テスト支援関数
def create_test_room(rt60: float = 0.3) -> Room:
	"""テスト用部屋を作成"""
	from project.config.config_classes import RoomConfig

	config = RoomConfig(
		dimensions=(3.0, 3.0, 3.0),
		reverberation_time=rt60,
		sampling_rate=16000
	)

	room = Room(config, "test_room")
	room.create_room()
	return room


def benchmark_room_creation(num_iterations: int = 10) -> Dict[str, float]:
	"""部屋作成のベンチマーク"""
	import time

	results = {}

	# 異なる条件でベンチマーク
	conditions = [
		("small_room", (2.0, 2.0, 2.0), 0.2),
		("medium_room", (5.0, 5.0, 5.0), 0.5),
		("large_room", (10.0, 10.0, 10.0), 1.0)
	]

	for name, dims, rt60 in conditions:
		times = []

		for i in range(num_iterations):
			start = time.time()

			room = create_test_room(rt60)
			room.recreate_room()

			times.append(time.time() - start)

		results[name] = {
			'mean_time': np.mean(times),
			'std_time': np.std(times),
			'min_time': np.min(times),
			'max_time': np.max(times)
		}

	return results


# モジュール公開API
__all__ = [
	'Room', 'RoomManager', 'RoomCondition', 'RoomConditionType', 'RoomStatistics',
	'RoomCreationError', 'ReverbCalculationError', 'SimulationError', 'RoomConfigError',
	'serch_reverbe_sec_compatible', 'create_room_compatible',
	'create_test_room', 'benchmark_room_creation'
]

# ロギング設定
logging.basicConfig(level=logging.INFO)