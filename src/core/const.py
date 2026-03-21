from pathlib import Path

"""
construction
このファイルは，PyroomAcousticsで使用する音源データのディレクトリ構造を定義するファイルです．

"""
# SOUND_DATA_DIR = Path('/Users/a/Documents/sound_data/')
SOUND_DATA_DIR = Path('C:/Users/adomin/Documents/sound_data/')

SAMPLE_DATA_DIR = SOUND_DATA_DIR / 'sample_data'
SPEECH_DATA_DIR = SAMPLE_DATA_DIR / 'speaker'
NOISE_DATA_DIR = SAMPLE_DATA_DIR / 'noise'

MIX_DATA_DIR = SOUND_DATA_DIR / 'mix_data'
PARMS_DATA_DIR = SOUND_DATA_DIR / 'preconpute_params'
RIR_DIR = SOUND_DATA_DIR / 'RIR'

