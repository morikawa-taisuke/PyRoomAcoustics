"""
【役割】
リポジトリのディレクトリ構造をテキスト出力するためのユーティリティ
"""
import os
import sys


def visualize_directory_tree(path, file_stream, indent=''):
	"""
    指定されたパスのディレクトリ構造をツリー形式で可視化し、
    ファイルストリームに書き込みます。

    Args:
        path (str): 可視化するディレクトリのパス。
        file_stream: 出力を書き込むファイルオブジェクト。
        indent (str): ツリーのインデント文字列。
    """
	if not os.path.isdir(path):
		# file_stream.write() でエラーメッセージを書き込む
		file_stream.write(f"エラー: 指定されたパス '{path}' はディレクトリではありません。\n")
		return

	# file_stream.write(f"📁 {os.path.basename(os.path.abspath(path))}/\n")

	items = sorted(os.listdir(path))

	for i, item in enumerate(items):
		item_path = os.path.join(path, item)
		is_last = (i == len(items) - 1)

		# プレフィックスの決定
		prefix = '└── ' if is_last else '├── '

		# ディレクトリかファイルかの判定
		if os.path.isdir(item_path):
			file_stream.write(f"{indent}{prefix}📁 {item}/\n")
			# サブディレクトリを再帰的に呼び出し
			new_indent = indent + ('    ' if is_last else '│   ')
			visualize_directory_tree(item_path, file_stream, new_indent)
		else:
			file_stream.write(f"{indent}{prefix}📄 {item}\n")


if __name__ == '__main__':
	target_path = "C:/Users/kataoka-lab/Desktop/sound_data/sample_data/speech/speeker_DEMAND"
	output_file_path = os.path.join(target_path, "directory_structure.txt")

	# 書き込みモード ('w') でファイルを開く
	try:
		with open(output_file_path, 'w', encoding='utf-8') as f:
			visualize_directory_tree(target_path, f)
		print(f"ディレクトリ構造が '{output_file_path}' に保存されました。")
	except Exception as e:
		print(f"ファイルの書き込み中にエラーが発生しました: {e}")