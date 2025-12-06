from pathlib import Path

"""
construction
設定

"""
SOUND_DATA_DIR = '/Users/a/Documents/sound_data/'
# SOUND_DATA_DIR = 'C:/Users/kataoka-lab/Desktop/sound_data/'
SAMPLE_DATA_DIR = Path(SOUND_DATA_DIR) / 'sample_data'
SPEECH_DATA_DIR = Path(SAMPLE_DATA_DIR) / 'speech'
NOISE_DATA_DIR = Path(SAMPLE_DATA_DIR) / 'noise'
MIX_DATA_DIR = Path(SOUND_DATA_DIR) / 'mix_data'
PARMS_DATA_DIR = Path(SOUND_DATA_DIR) / 'preconpute_params'

