"""
【役割】
WAVファイルのリサンプリング処理やサンプリングレートの確認を行うモジュール
"""
import argparse
from pathlib import Path
import soundfile as sf
import resampy
import numpy as np
from tqdm import tqdm


# --- 機能1: サンプリングレートのチェック ---
def check_samplerates_in_dir(dir_path: Path):
	"""
    指定されたディレクトリ内のすべてのWAVファイルのサンプリングレートを出力する。

    Args:
        dir_path (Path): 検索対象のディレクトリパス。
    """
	if not dir_path.is_dir():
		print(f"エラー: ディレクトリが見つかりません: {dir_path}")
		return

	print(f"ディレクトリ '{dir_path}' 内のWAVファイルのサンプリングレートをチェックします...")
	wav_files = sorted(list(dir_path.rglob("*.wav")))

	if not wav_files:
		print("WAVファイルが見つかりませんでした。")
		return

	samplerates = set()
	for wav_file in wav_files:
		try:
			info = sf.info(wav_file)
			print(f"{info.samplerate} Hz\t: {wav_file.relative_to(dir_path)}")
			samplerates.add(info.samplerate)
		except Exception as e:
			print(f"エラー\t: {wav_file.relative_to(dir_path)} ({e})")

	print("\n--- まとめ ---")
	if samplerates:
		print(f"検出されたサンプリングレート: {sorted(list(samplerates))}")
	else:
		print("有効なWAVファイルからサンプリングレートを検出できませんでした。")


# --- 機能2: サンプリングレートの変換 ---
def resample_wav(input_path: Path, output_path: Path, target_sr: int, overwrite: bool):
	"""
    単一のWAVファイルを指定されたサンプリングレートにリサンプリングする。

    Args:
        input_path (Path): 入力WAVファイルのパス。
        output_path (Path): 出力WAVファイルのパス。
        target_sr (int): ターゲットサンプリングレート。
        overwrite (bool): 出力ファイルが既に存在する場合に上書きするかどうか。
    """
	if not overwrite and output_path.exists():
		# print(f"スキップ: {output_path} は既に存在します。")
		return

	try:
		info = sf.info(input_path)
		if info.samplerate == target_sr:
			# サンプリングレートが同じ場合はファイルをコピーする
			output_path.parent.mkdir(parents=True, exist_ok=True)
			import shutil
			shutil.copy2(input_path, output_path)
			# print(f"コピー: {input_path} -> {output_path} (サンプリングレート同一)")
			return

		# 音声データを読み込む
		data, original_sr = sf.read(input_path, always_2d=True)

		# リサンプリング (resampyを使用)
		# resampyは (N, C) ではなく (C, N) を期待するため転置(T)する
		resampled_data = resampy.resample(data.T, original_sr, target_sr, axis=1)

		# 元の (N, C) 形式に戻す
		resampled_data = resampled_data.T

		# 保存
		output_path.parent.mkdir(parents=True, exist_ok=True)
		sf.write(output_path, resampled_data, target_sr)
		# print(f"変換完了: {input_path} ({original_sr}Hz) -> {output_path} ({target_sr}Hz)")

	except Exception as e:
		print(f"エラー: {input_path} の処理中にエラーが発生しました: {e}")


def convert_samplerates(input_path: Path, output_dir: Path, target_sr: int, overwrite: bool):
	"""
    指定されたファイルまたはディレクトリ内のWAVファイルをリサンプリングする。
    """
	if not input_path.exists():
		print(f"エラー: 入力パスが見つかりません: {input_path}")
		return

	if input_path.is_dir():
		wav_files = sorted(list(input_path.rglob("*.wav")))
		print(f"ディレクトリ '{input_path}' 内の {len(wav_files)} 個のWAVファイルを処理します...")
	else:
		wav_files = [input_path]
		print(f"ファイル '{input_path}' を処理します...")

	if not wav_files:
		print("処理対象のWAVファイルが見つかりませんでした。")
		return

	for wav_file in tqdm(wav_files, desc="Resampling"):
		# 出力パスを決定 (入力のディレクトリ構造を維持)
		relative_path = wav_file.relative_to(input_path.parent if input_path.is_file() else input_path)
		output_path = output_dir / relative_path
		resample_wav(wav_file, output_path, target_sr, overwrite)

	print(f"\n処理が完了しました。出力先: {output_dir.resolve()}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="WAVファイルのサンプリングレートをチェックまたは変換します。")
	subparsers = parser.add_subparsers(dest="command", help="実行するコマンド")

	# 'check' コマンド
	parser_check = subparsers.add_parser("check", help="ディレクトリ内のWAVファイルのサンプリングレートをチェックする")
	parser_check.add_argument("directory", type=str, help="チェック対象のディレクトリ")

	# 'convert' コマンド
	parser_convert = subparsers.add_parser("convert", help="WAVファイルのサンプリングレートを変換する")
	parser_convert.add_argument("--input_path", default="C:/Users/kataoka-lab/Desktop/sound_data/sample_data/noise/DEMAND", type=str, help="変換対象のWAVファイルまたはディレクトリ")
	parser_convert.add_argument("--output_dir", default="C:/Users/kataoka-lab/Desktop/sound_data/sample_data/noise/DEMAND_16kHz", type=str, help="変換後のファイルの出力先ディレクトリ")
	parser_convert.add_argument("-r", "--rate", default=16000, type=int, help="ターゲットサンプリングレート (例: 16000)")
	parser_convert.add_argument("--overwrite", default=True, action="store_true", help="出力ファイルが既に存在する場合に上書きする")

	args = parser.parse_args()

	if args.command == "check":
		check_samplerates_in_dir(Path(args.directory))
	elif args.command == "convert":
		convert_samplerates(
			Path(args.input_path),
			Path(args.output_dir),
			args.rate,
			args.overwrite
		)
