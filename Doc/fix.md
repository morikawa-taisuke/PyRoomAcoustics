
# RoomClass実装改善計画

## 1. 概要
pyroomacousticsを用いた音響シミュレーションコードの問題点を解決し、クラス設計によって保守性・拡張性・再利用性を向上させる改善計画です。

## 2. 現在の問題点の整理

### 2.1 コード構造の問題
- **単一責任原則の違反**: 一つの関数が複数の責任を持っている
- **重複コードの存在**: 4つの部屋作成で同様の処理を繰り返し
- **ハードコーディング**: 設定値がコード内に直接記述されている
- **長大な関数**: recoding2関数が200行以上で可読性が低い

### 2.2 設定管理の問題
- **設定の分散**: パラメータが関数引数やローカル変数として分散
- **Config.json未活用**: ドキュメントで言及されているが実装されていない
- **実験条件変更の困難**: コード修正が必要で非効率

### 2.3 拡張性の問題
- **マイクアレイ形状の制限**: 線形アレイのみ、円形アレイが未対応
- **音源数の固定**: 目的音声＋雑音の2音源に限定
- **部屋パラメータの硬直性**: 形状や材質の変更が困難

## 3. 改善方針

### 3.1 設計原則
- **単一責任原則**: 各クラスは一つの責任のみを持つ
- **開放閉鎖原則**: 拡張に対して開いており、修正に対して閉じている
- **依存性逆転原則**: 抽象に依存し、具体に依存しない
- **設定の外部化**: 全ての設定をConfig.jsonで管理

### 3.2 クラス設計方針
- **階層化設計**: 低レベル（物理設定）から高レベル（シミュレーション実行）まで階層化
- **組み合わせ可能性**: 各コンポーネントを独立して組み合わせ可能
- **テスト容易性**: 各クラスが独立してテスト可能

## 4. 提案するクラス構造

### 4.1 設定管理層
```aiexclude
ConfigManager 
├── RoomConfig (部屋設定) 
├── MicArrayConfig (マイクアレイ設定) 
├── SourceConfig (音源設定) 
└── SimulationConfig (シミュレーション設定)
```
### 4.2 物理モデル層
```
PhysicalModel 
├── Room (部屋の物理特性) 
├── MicrophoneArray (マイクアレイ) 
│ ├── LinearArray 
│ └── CircularArray 
└── SoundSource (音源)
``` 

### 4.3 シミュレーション実行層
```
AcousticSimulator 
├── SingleConditionSimulator (単一条件) 
└── MultiConditionSimulator (複数条件)
``` 

### 4.4 ユーティリティ層
```
Utils 
├── AudioProcessor (音声処理) 
├── FileManager (ファイル管理) 
└── ValidationHelper (入力検証)
``` 

## 5. 詳細設計

### 5.1 ConfigManagerクラス
**責任**: Config.jsonの読み込みと設定値の管理
```python 
class ConfigManager: 
	load_config(config_path: str) 
	get_room_config() -> RoomConfig 
	get_mic_array_config() -> MicArrayConfig 
	get_source_config() -> SourceConfig 
	validate_config()
``` 

### 5.2 RoomConfigクラス
**責任**: 部屋に関する設定値の保持
```python 
class RoomConfig: 
	dimensions: List[float] # [x, y, z] 
	reverberation_time: float 
	absorption: float 
	max_order: int 
	material_properties: Dict
``` 

### 5.3 MicArrayConfigクラス
**責任**: マイクアレイの設定値の保持
```python 
class MicArrayConfig: 
	array_type: str # "linear" or "circular" 
	num_channels: int 
	spacing: float # 間隔 or 半径 
	center_position: List[float] 
	orientation: float
``` 

### 5.4 SourceConfigクラス
**責任**: 音源の設定値の保持
```python 
class SourceConfig: 
	arget_source: Dict # 目的音源の設定 - 
	noise_sources: List[Dict] # 雑音源のリスト 
	snr_values: List[float] # SNR設定 
	angles: List[float] # 音源角度
``` 

### 5.5 Roomクラス
**責任**: pyroomacousticsの部屋オブジェクトの管理
```python
class Room: 
	create_room(config: RoomConfig) -> pa.ShoeBox 
	add_microphones(room: pa.ShoeBox, mic_array: MicrophoneArray) 
	add_sources(room: pa.ShoeBox, sources: List[SoundSource]) 
	calculate_rt60() -> float
``` 

### 5.6 MicrophoneArrayクラス（抽象基底クラス）
**責任**: マイクアレイの座標計算
```python
class MicrophoneArray(ABC): 
	@abstractmethod
	def get_coordinates() -> np.ndarray
class LinearArray(MicrophoneArray): 
	def get_coordinates() -> np.ndarray
class CircularArray(MicrophoneArray): 
	def get_coordinates() -> np.ndarray
``` 

### 5.7 SoundSourceクラス
**責任**: 音源の管理
```python 
class SoundSource: 
	load_audio(file_path: str) 
	set_position(coordinates: List[float]) 
	set_angle(elevation: float, azimuth: float) 
	apply_snr(target: np.ndarray, snr: float) -> np.ndarray
``` 

### 5.8 AcousticSimulatorクラス
**責任**: シミュレーションの実行統括
```python 
class AcousticSimulator: 
	setup_simulation(config: ConfigManager) 
	run_clean_simulation() -> np.ndarray 
	run_noise_simulation() -> np.ndarray 
	nun_reverb_simulation() -> np.ndarray 
	run_mixed_simulation() -> np.ndarray 
	save_results(output_dir: str)
``` 

## 6. Config.json設計例
```json 
{ "room": { 
	"dimensions": [5.0, 5.0, 5.0],
	"reverberation_time": 0.5, 
	"material": "default" }, 
	"microphone_array": { 
		"type": "linear", 
		"num_channels": 2, 
		"spacing": 0.1, 
		"center_position": [2.5, 2.5, 2.5],
		"orientation": 0.0
	}, 
	"sources": { 
		"target": { 
			"directory": "./data/speech/", 
			"distance": 0.5, 
			"angle": 90
		}, 
		"noise": {
			"file": "./data/noise/hoth.wav", 
			"distance": 0.7, 
			"angle": 180, 
			"snr_values": [5, 10, 15]
		}
	},
	"simulation": { 
		"sampling_rate": 16000, 
		"output_types": ["clean", "noise", "reverb", "mixed"], 
		"save_split": false
	}, 
	"output": { 
		"base_directory": "./output/", 
		"naming_convention": "{signal}_{noise}_{snr}dB_{rt60}sec"
	}
}
``` 

## 7. 実装手順

### Phase 1: 基盤クラスの実装
1. ConfigManagerとConfig系クラスの実装
2. 基本的なvalidation機能の追加
3. Config.jsonの読み込み機能

### Phase 2: 物理モデルクラスの実装
1. Room, MicrophoneArray系クラスの実装
2. SoundSourceクラスの実装
3. 既存機能との互換性確認

### Phase 3: シミュレーター実装
1. AcousticSimulatorクラスの実装
2. 既存のrecoding2関数の置き換え
3. エラーハンドリングの追加

### Phase 4: 拡張機能とテスト
1. 円形アレイ対応の実装
2. 複数音源対応の実装
3. 単体テストの作成

## 8. 期待される効果

### 8.1 保守性の向上
- **責任の明確化**: 各クラスの役割が明確
- **変更の局所化**: 修正の影響範囲を限定
- **コードの理解容易性**: 構造が整理され理解しやすい

### 8.2 拡張性の向上
- **新機能の追加容易性**: インターフェースを通じた拡張
- **設定の柔軟性**: Config.jsonで様々な条件を設定可能
- **アルゴリズムの変更容易性**: 実装の入れ替えが容易

### 8.3 再利用性の向上
- **モジュール単位での再利用**: 各クラスが独立して利用可能
- **設定の再利用**: 同じ設定で異なる音声を処理可能
- **実験の効率化**: 条件変更が設定ファイル編集のみで完了

### 8.4 品質の向上
- **バグの削減**: 構造化によりバグの発生を抑制
- **テスタビリティ**: 各コンポーネントの独立テストが可能
- **ドキュメント化**: クラス設計により仕様が明確

## 9. 移行戦略

### 9.1 段階的移行
- **既存コードとの並行運用**: 新クラス実装中も既存機能を維持
- **機能単位での置き換え**: 一度に全てを変更せず段階的に移行
- **テストによる検証**: 各段階での動作確認を徹底

### 9.2 後方互換性
- **既存インターフェースの維持**: 可能な限り既存の呼び出し方法をサポート
- **設定の移行支援**: 既存のハードコーディング値をConfig.jsonに移行するツール

## 10. 今後の拡張計画

### 10.1 短期拡張
- **GUI対応**: 設定変更をGUIで行える機能
- **バッチ処理**: 複数条件の一括処理機能
- **結果分析**: シミュレーション結果の自動分析機能

### 10.2 長期拡張
- **機械学習連携**: 学習データセット生成の自動化
- **分散処理**: 大規模シミュレーションの並列実行
- **3D可視化**: 音場の可視化機能

