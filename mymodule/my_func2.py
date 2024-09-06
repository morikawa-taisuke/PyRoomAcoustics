import os
import numpy as np
import wave
import array
from pathlib import Path


def record(text):
  print('save_txt')
  f = open('./result/00state.txt','w')
  f.write(text)
  f.close()

def record_room(room_dim, rt60, max_order, e_absorption):
  text = f'room_dim[m]:{room_dim}\n' \
         f'残響時間[sec]:{rt60}\n' \
         f'反射上限[回]:{max_order}\n' \
         f'壁の材質:{e_absorption}'

  record(text)

