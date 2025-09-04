"""
音響シミュレーション設定管理クラス

Config.jsonの読み書きと各設定クラスの統合管理を行います。
既存コードとの互換性を保ちながら、設定の一元管理を実現します。
"""

from __future__ import annotations
import json
import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime
import threading

# 自作モジュール
from config_classes import (
	AcousticSimulationConfig, RoomConfig, MicArrayConfig, SourceConfig,
	SimulationConfig, OutputConfig, ConfigValidationError, ConfigFileError
)

# 設定ファイルのスキーマバージョン
CONFIG_SCHEMA_VERSION = "1.0.0"


class ConfigManager:
	"""
	音響シミュレーション設定の統合管理クラス

	シングルトンパターンで実装され、アプリケーション全体で
	一つの設定管理インスタンスを提供します。
	"""

	_instance: Optional[ConfigManager] = None
	_lock = threading.Lock()

	def __new__(cls) -> ConfigManager:
		"""シングルトンパターンの実装"""
		if cls._instance is None:
			with cls._lock:
				if cls._instance is None:
					cls._instance = super().__new__(cls)
		return cls._instance

	def __init__(self):
		"""初期化処理"""
		if hasattr(self, '_initialized'):
			return

		self._initialized = True
		self._config: Optional[AcousticSimulationConfig] = None
		self._config_file_path: Optional[str] = None
		self._observers: List[Callable[[AcousticSimulationConfig], None]] = []
		self._logger = logging.getLogger(__name__)
		self._backup_dir = "./config_backups/"

		# ログ設定
		logging.basicConfig(
			level=logging.INFO,
			format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
			handlers=[
				logging.StreamHandler(),
				logging.FileHandler('config_manager.log', encoding='utf-8')
			]
		)

	# =============================================================================
	# 基本的な設定ファイル管理機能
	# =============================================================================

	def load_config(self, config_path: str, validate: bool = True) -> AcousticSimulationConfig:
		"""
		設定ファイルを読み込み

		Args:
			config_path: 設定ファイルのパス
			validate: 読み込み後にバリデーションを実行するか

		Returns:
			読み込まれた設定

		Raises:
			ConfigFileError: ファイル読み込みエラー
			ConfigValidationError: 設定値検証エラー
		"""
		try:
			self._logger.info(f"設定ファイルを読み込み中: {config_path}")

			if not os.path.exists(config_path):
				raise ConfigFileError(f"設定ファイルが存在しません: {config_path}")

			config = AcousticSimulationConfig.from_json_file(config_path)

			if validate:
				self._validate_loaded_config(config)

			self._config = config
			self._config_file_path = config_path

			# 観察者に通知
			self._notify_observers(config)

			self._logger.info("設定ファイルの読み込みが完了しました")
			return config

		except Exception as e:
			self._logger.error(f"設定ファイルの読み込みに失敗: {e}")
			raise

	def save_config(self, config: Optional[AcousticSimulationConfig] = None,
	                file_path: Optional[str] = None, create_backup: bool = True) -> None:
		"""
		設定をファイルに保存

		Args:
			config: 保存する設定（Noneの場合は現在の設定）
			file_path: 保存先パス（Noneの場合は読み込み元パス）
			create_backup: バックアップを作成するか
		"""
		config = config or self._config
		file_path = file_path or self._config_file_path

		if not config:
			raise ConfigValidationError("保存する設定が指定されていません")
		if not file_path:
			raise ConfigFileError("保存先パスが指定されていません")

		try:
			# バックアップ作成
			if create_backup and os.path.exists(file_path):
				self._create_backup(file_path)

			# ディレクトリが存在しない場合は作成
			os.makedirs(os.path.dirname(file_path), exist_ok=True)

			# メタデータを追加
			config_with_meta = self._add_metadata(config)

			# ファイルに保存
			with open(file_path, 'w', encoding='utf-8') as f:
				f.write(config_with_meta)

			self._config = config
			self._config_file_path = file_path

			self._logger.info(f"設定ファイルを保存しました: {file_path}")

		except Exception as e:
			self._logger.error(f"設定ファイルの保存に失敗: {e}")
			raise ConfigFileError(f"設定ファイルの保存に失敗しました: {e}")

	def create_default_config_file(self, file_path: str) -> None:
		"""
		デフォルト設定ファイルを作成

		Args:
			file_path: 作成するファイルのパス
		"""
		try:
			default_config = AcousticSimulationConfig()
			self.save_config(default_config, file_path, create_backup=False)
			self._logger.info(f"デフォルト設定ファイルを作成しました: {file_path}")
		except Exception as e:
			self._logger.error(f"デフォルト設定ファイルの作成に失敗: {e}")
			raise

	def validate_config_file(self, file_path: str) -> bool:
		"""
		設定ファイルの妥当性を検証

		Args:
			file_path: 検証するファイルのパス

		Returns:
			検証結果
		"""
		try:
			config = AcousticSimulationConfig.from_json_file(file_path)
			return config.validate()
		except Exception as e:
			self._logger.error(f"設定ファイルの検証に失敗: {e}")
			return False

	# =============================================================================
	# 設定取得機能
	# =============================================================================

	def get_current_config(self) -> Optional[AcousticSimulationConfig]:
		"""現在の設定を取得"""
		return self._config

	def get_room_config(self) -> Optional[RoomConfig]:
		"""部屋設定を取得"""
		return self._config.room if self._config else None

	def get_mic_array_config(self) -> Optional[MicArrayConfig]:
		"""マイクアレイ設定を取得"""
		return self._config.microphone_array if self._config else None

	def get_source_config(self) -> Optional[SourceConfig]:
		"""音源設定を取得"""
		return self._config.sources if self._config else None

	def get_simulation_config(self) -> Optional[SimulationConfig]:
		"""シミュレーション設定を取得"""
		return self._config.simulation if self._config else None

	def get_output_config(self) -> Optional[OutputConfig]:
		"""出力設定を取得"""
		return self._config.output if self._config else None

	# =============================================================================
	# 既存コード互換機能
	# =============================================================================

	def from_legacy_recoding2_params(self, wave_files: List[str], out_dir: str,
	                                 snr: float, reverbe_sec: float, channel: int = 1,
	                                 distance: float = 0, angle: float = 3.14159,
	                                 is_split: bool = False) -> AcousticSimulationConfig:
		"""
		既存のrecoding2関数のパラメータから設定を作成

		Args:
			wave_files: 音声ファイルのリスト [目的音声, 雑音]
			out_dir: 出力ディレクトリ
			snr: SNR値
			reverbe_sec: 残響時間
			channel: チャンネル数
			distance: マイク間隔（cm）
			angle: 雑音源の角度
			is_split: チャンネル分割保存

		Returns:
			生成された設定
		"""
		try:
			config = AcousticSimulationConfig.from_legacy_recoding2_params(
				wave_files, out_dir, snr, reverbe_sec, channel, distance, angle
			)

			# 分割保存設定の更新
			config = self._update_config_field(config, 'simulation', 'save_split', is_split)

			self._config = config
			self._logger.info("既存パラメータから設定を作成しました")

			return config

		except Exception as e:
			self._logger.error(f"既存パラメータからの設定作成に失敗: {e}")
			raise

	def to_legacy_format(self) -> Dict[str, Any]:
		"""
		現在の設定を既存コードで使える形式に変換

		Returns:
			既存コード用のパラメータ辞書
		"""
		if not self._config:
			raise ConfigValidationError("設定が読み込まれていません")

		try:
			legacy_params = {
				'room_dim': list(self._config.room.dimensions),
				'reverbe_sec': self._config.room.reverberation_time,
				'channel': self._config.microphone_array.num_channels,
				'distance': self._config.microphone_array.spacing * 100,  # mからcmに変換
				'snr_values': list(self._config.sources.snr_values),
				'sample_rate': self._config.room.sampling_rate,
				'target_distance': self._config.sources.target_distance,
				'noise_distance': self._config.sources.noise_distance,
				'target_angle': self._config.sources.target_angle,
				'noise_angle': self._config.sources.noise_angle,
				'is_split': self._config.simulation.save_split,
				'out_dir': self._config.output.base_directory,
			}

			return legacy_params

		except Exception as e:
			self._logger.error(f"既存形式への変換に失敗: {e}")
			raise

	def extract_hardcoded_values_from_file(self, python_file_path: str) -> Dict[str, Any]:
		"""
		既存Pythonファイルからハードコーディングされた値を抽出

		Args:
			python_file_path: 解析するPythonファイルのパス

		Returns:
			抽出された値の辞書
		"""
		# 実装は複雑になるため、基本的なパターンマッチングのみ実装
		extracted_values = {}

		try:
			with open(python_file_path, 'r', encoding='utf-8') as f:
				content = f.read()

			# 基本的なパターンを検索
			import re

			# room_dimの抽出
			room_dim_match = re.search(r'room_dim\s*=\s*np\.r_\[([0-9.,\s]+)\]', content)
			if room_dim_match:
				values = [float(x.strip()) for x in room_dim_match.group(1).split(',')]
				extracted_values['room_dimensions'] = values

			# その他のパラメータも同様に抽出
			# （実際の実装では、より詳細な解析が必要）

			self._logger.info(f"ハードコーディング値を抽出しました: {len(extracted_values)}項目")
			return extracted_values

		except Exception as e:
			self._logger.error(f"ハードコーディング値の抽出に失敗: {e}")
			return {}

	# =============================================================================
	# 設定の動的変更機能
	# =============================================================================

	def update_room_config(self, updates: Dict[str, Any]) -> None:
		"""部屋設定を更新"""
		if not self._config:
			raise ConfigValidationError("設定が読み込まれていません")

		try:
			# 現在の設定を辞書に変換
			current_room = self._config.room.to_dict()
			current_room.update(updates)

			# 新しい設定を作成
			new_room_config = RoomConfig(**current_room)
			new_room_config.validate()

			# 全体設定を更新
			self._config = self._replace_config_field(self._config, 'room', new_room_config)

			# 観察者に通知
			self._notify_observers(self._config)

			self._logger.info("部屋設定を更新しました")

		except Exception as e:
			self._logger.error(f"部屋設定の更新に失敗: {e}")
			raise

	def update_mic_array_config(self, updates: Dict[str, Any]) -> None:
		"""マイクアレイ設定を更新"""
		if not self._config:
			raise ConfigValidationError("設定が読み込まれていません")

		try:
			current_mic = self._config.microphone_array.to_dict()
			current_mic.update(updates)

			new_mic_config = MicArrayConfig(**current_mic)
			new_mic_config.validate()

			self._config = self._replace_config_field(self._config, 'microphone_array', new_mic_config)
			self._notify_observers(self._config)

			self._logger.info("マイクアレイ設定を更新しました")

		except Exception as e:
			self._logger.error(f"マイクアレイ設定の更新に失敗: {e}")
			raise

	def update_source_config(self, updates: Dict[str, Any]) -> None:
		"""音源設定を更新"""
		if not self._config:
			raise ConfigValidationError("設定が読み込まれていません")

		try:
			current_source = self._config.sources.to_dict()
			current_source.update(updates)

			new_source_config = SourceConfig(**current_source)
			new_source_config.validate()

			self._config = self._replace_config_field(self._config, 'sources', new_source_config)
			self._notify_observers(self._config)

			self._logger.info("音源設定を更新しました")

		except Exception as e:
			self._logger.error(f"音源設定の更新に失敗: {e}")
			raise

	def merge_configs(self, base_config: AcousticSimulationConfig,
	                  override_config: AcousticSimulationConfig) -> AcousticSimulationConfig:
		"""
		2つの設定をマージ

		Args:
			base_config: ベースとなる設定
			override_config: 上書きする設定

		Returns:
			マージされた設定
		"""
		try:
			# 辞書形式に変換してマージ
			base_dict = base_config.to_dict()
			override_dict = override_config.to_dict()

			merged_dict = self._deep_merge_dicts(base_dict, override_dict)
			merged_config = AcousticSimulationConfig.from_dict(merged_dict)

			merged_config.validate()

			self._logger.info("設定のマージが完了しました")
			return merged_config

		except Exception as e:
			self._logger.error(f"設定のマージに失敗: {e}")
			raise

	# =============================================================================
	# バリデーション統括機能
	# =============================================================================

	def validate_all_configs(self) -> bool:
		"""全ての設定の妥当性を検証"""
		if not self._config:
			return False

		try:
			return self._config.validate()
		except Exception as e:
			self._logger.error(f"設定の検証に失敗: {e}")
			return False

	def validate_cross_dependencies(self) -> bool:
		"""設定間の依存関係を検証"""
		if not self._config:
			return False

		try:
			# サンプリングレートの整合性
			if self._config.room.sampling_rate != self._config.simulation.sampling_rate:
				self._logger.error("部屋とシミュレーションのサンプリングレートが不一致です")
				return False

			# マイクアレイ位置の妥当性
			mic_center = self._config.microphone_array.center_position
			room_dims = self._config.room.dimensions

			if not all(0 < pos < dim for pos, dim in zip(mic_center, room_dims)):
				self._logger.error("マイクアレイが部屋の範囲外に配置されています")
				return False

			return True

		except Exception as e:
			self._logger.error(f"依存関係の検証に失敗: {e}")
			return False

	def check_file_dependencies(self) -> bool:
		"""ファイル依存関係を検証"""
		if not self._config:
			return False

		try:
			# 音源ディレクトリの存在確認
			if not os.path.exists(self._config.sources.target_directory):
				self._logger.error(f"音源ディレクトリが存在しません: {self._config.sources.target_directory}")
				return False

			# 雑音ファイルの存在確認
			if not os.path.isfile(self._config.sources.noise_file):
				self._logger.error(f"雑音ファイルが存在しません: {self._config.sources.noise_file}")
				return False

			return True

		except Exception as e:
			self._logger.error(f"ファイル依存関係の検証に失敗: {e}")
			return False

	def generate_validation_report(self) -> str:
		"""検証レポートを生成"""
		if not self._config:
			return "設定が読み込まれていません"

		report = []
		report.append("=== 設定検証レポート ===")
		report.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
		report.append("")

		# 基本検証
		try:
			basic_valid = self.validate_all_configs()
			report.append(f"基本検証: {'OK' if basic_valid else 'NG'}")
		except Exception as e:
			report.append(f"基本検証: エラー - {e}")

		# 依存関係検証
		try:
			deps_valid = self.validate_cross_dependencies()
			report.append(f"依存関係検証: {'OK' if deps_valid else 'NG'}")
		except Exception as e:
			report.append(f"依存関係検証: エラー - {e}")

		# ファイル検証
		try:
			files_valid = self.check_file_dependencies()
			report.append(f"ファイル検証: {'OK' if files_valid else 'NG'}")
		except Exception as e:
			report.append(f"ファイル検証: エラー - {e}")

		# 設定サマリー
		report.append("")
		report.append("=== 設定サマリー ===")
		report.append(self._config.summary())

		return "\n".join(report)

	# =============================================================================
	# ファクトリーメソッド
	# =============================================================================

	def create_development_config(self) -> AcousticSimulationConfig:
		"""開発用設定を作成"""
		return AcousticSimulationConfig(
			room=RoomConfig(dimensions=(3.0, 3.0, 3.0), reverberation_time=0.2),
			microphone_array=MicArrayConfig(num_channels=1),
			sources=SourceConfig(snr_values=(10.0,)),
			simulation=SimulationConfig(output_types=("clean", "mixed"))
		)

	def create_production_config(self) -> AcousticSimulationConfig:
		"""本番用設定を作成"""
		return AcousticSimulationConfig()  # デフォルト設定

	def create_test_config(self) -> AcousticSimulationConfig:
		"""テスト用設定を作成"""
		return AcousticSimulationConfig(
			room=RoomConfig(dimensions=(2.0, 2.0, 2.0), reverberation_time=0.1),
			microphone_array=MicArrayConfig(num_channels=1, spacing=0.0),
			sources=SourceConfig(snr_values=(15.0,)),
			simulation=SimulationConfig(output_types=("clean",))
		)

	def create_benchmark_config(self) -> AcousticSimulationConfig:
		"""ベンチマーク用設定を作成"""
		return AcousticSimulationConfig(
			room=RoomConfig(dimensions=(10.0, 10.0, 10.0), reverberation_time=1.0),
			microphone_array=MicArrayConfig(array_type="circular", num_channels=8),
			sources=SourceConfig(snr_values=(0.0, 5.0, 10.0, 15.0, 20.0)),
			simulation=SimulationConfig(output_types=("clean", "noise", "reverb", "mixed"))
		)

	# =============================================================================
	# 観察者パターン
	# =============================================================================

	def add_observer(self, observer: Callable[[AcousticSimulationConfig], None]) -> None:
		"""設定変更の観察者を追加"""
		self._observers.append(observer)

	def remove_observer(self, observer: Callable[[AcousticSimulationConfig], None]) -> None:
		"""設定変更の観察者を削除"""
		if observer in self._observers:
			self._observers.remove(observer)

	def _notify_observers(self, config: AcousticSimulationConfig) -> None:
		"""観察者に設定変更を通知"""
		for observer in self._observers:
			try:
				observer(config)
			except Exception as e:
				self._logger.error(f"観察者への通知に失敗: {e}")

	# =============================================================================
	# プライベートヘルパーメソッド
	# =============================================================================

	def _validate_loaded_config(self, config: AcousticSimulationConfig) -> None:
		"""読み込まれた設定の検証"""
		config.validate()

	def _create_backup(self, file_path: str) -> None:
		"""設定ファイルのバックアップを作成"""
		try:
			os.makedirs(self._backup_dir, exist_ok=True)

			timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
			filename = os.path.basename(file_path)
			backup_path = os.path.join(self._backup_dir, f"{timestamp}_{filename}")

			shutil.copy2(file_path, backup_path)
			self._logger.info(f"バックアップを作成しました: {backup_path}")

		except Exception as e:
			self._logger.warning(f"バックアップの作成に失敗: {e}")

	def _add_metadata(self, config: AcousticSimulationConfig) -> str:
		"""設定にメタデータを追加してJSON文字列を生成"""
		config_dict = config.to_dict()

		metadata = {
			"_metadata": {
				"schema_version": CONFIG_SCHEMA_VERSION,
				"created_at": datetime.now().isoformat(),
				"created_by": "ConfigManager"
			}
		}

		config_dict.update(metadata)
		return json.dumps(config_dict, indent=4, ensure_ascii=False)

	def _deep_merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
		"""辞書の深いマージ"""
		result = base.copy()

		for key, value in override.items():
			if key in result and isinstance(result[key], dict) and isinstance(value, dict):
				result[key] = self._deep_merge_dicts(result[key], value)
			else:
				result[key] = value

		return result

	def _update_config_field(self, config: AcousticSimulationConfig,
	                         section: str, field: str, value: Any) -> AcousticSimulationConfig:
		"""設定の特定フィールドを更新"""
		config_dict = config.to_dict()
		if section not in config_dict:
			config_dict[section] = {}
		config_dict[section][field] = value
		return AcousticSimulationConfig.from_dict(config_dict)

	def _replace_config_field(self, config: AcousticSimulationConfig,
	                          field_name: str, new_value: Any) -> AcousticSimulationConfig:
		"""設定の特定セクションを置き換え"""
		config_dict = config.to_dict()
		config_dict[field_name] = new_value.to_dict() if hasattr(new_value, 'to_dict') else new_value
		return AcousticSimulationConfig.from_dict(config_dict)


# =============================================================================
# モジュールレベルの便利関数
# =============================================================================

# グローバルインスタンス
_global_config_manager = None


def get_config_manager() -> ConfigManager:
	"""グローバル設定管理インスタンスを取得"""
	global _global_config_manager
	if _global_config_manager is None:
		_global_config_manager = ConfigManager()
	return _global_config_manager


def load_config_from_file(file_path: str) -> AcousticSimulationConfig:
	"""設定ファイルを読み込む便利関数"""
	return get_config_manager().load_config(file_path)


def create_default_config(file_path: str = "./config/default_config.json") -> None:
	"""デフォルト設定ファイルを作成する便利関数"""
	get_config_manager().create_default_config_file(file_path)


def get_current_config() -> Optional[AcousticSimulationConfig]:
	"""現在の設定を取得する便利関数"""
	return get_config_manager().get_current_config()


# モジュール公開API
__all__ = [
	'ConfigManager', 'get_config_manager', 'load_config_from_file',
	'create_default_config', 'get_current_config'
]