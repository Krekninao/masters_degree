import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from side import read_file
def spectr(signal, sampl_num, sampl_rate):
  sig_fft = fft(signal)
  freqs = fftfreq(sampl_num, 1/sampl_rate)

  # Отображаем только положительные частоты
  positive_freqs_mask = freqs > 0
  freqs = freqs[positive_freqs_mask]
  sig_fft = np.abs(sig_fft[positive_freqs_mask])

  # Выводим результаты
  plt.plot(freqs, sig_fft)
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Amplitude')
  plt.show()

  # Определяем частоту с наибольшей амплитудой
  max_amp_freq = freqs[np.argmax(sig_fft)]
  print("Max amplitude frequency: {:.2f} Hz".format(max_amp_freq))
  return max_amp_freq

num_file = 19
path = r'C:\Users\user\Desktop\Магистерская\Быстровка_07_07_10_Круг_6км\all_Unit\3seans_19'
path = path.replace('\\', '/')
# считывание данных из файлов
sampl_rate_x, sampl_num_x, T_x, t_x, signal_x = read_file(path + f'/U.{num_file}x')

#fx = spectr(signal_x[:600*200], sampl_num_x, sampl_rate_x)
fx = spectr(signal_x[2*600*200:3*600*200], len(signal_x[2*600*200:3*600*200]), sampl_rate_x)