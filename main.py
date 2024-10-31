import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from side import read_file, quad
def spectr(signal, sampl_num, sampl_rate):
  sig_fft = fft(signal)
  freqs = fftfreq(sampl_num, 1/sampl_rate)

  # Отображаем только положительные частоты
  positive_freqs_mask = freqs > 0
  freqs = freqs[positive_freqs_mask]
  sig_fft = np.abs(sig_fft[positive_freqs_mask])

  # Выводим результаты
  # plt.plot(freqs, sig_fft)
  # plt.xlabel('Frequency (Hz)')
  # plt.ylabel('Amplitude')
  # plt.show()

  # Определяем частоту с наибольшей амплитудой
  max_amp_freq = freqs[np.argmax(sig_fft)]
  #print("Max amplitude frequency: {:.2f} Hz".format(max_amp_freq))
  return max_amp_freq

num_file = 19
path = r'C:\Users\user\Desktop\Магистерская\Быстровка_07_07_10_Круг_6км\all_Unit\3seans_19'
path = path.replace('\\', '/')
# считывание данных из файлов
sampl_rate_x, sampl_num_x, T_x, t_x, signal_x = read_file(path + f'/U.{num_file}x')

#for i in range(len(signal_x))
#размер окна
win_size = 5
window = 300
seconds = list(range(0, int(t_x[-1]) + 1, win_size))
phi_total = []
for start, finish in zip(seconds, seconds[1:]):
  #обдумать момент с концом интервала!
  sig_part = signal_x[start * sampl_rate_x : finish * sampl_rate_x-1]
  max_fr = spectr(sig_part, len(sig_part), sampl_rate_x)
  R_x, phi_x, x1_list = quad(sig_part, max_fr, window, sampl_rate_x)
  phi_total.extend(phi_x)

plt.figure(figsize=(15, 6))
plt.plot(phi_total)
plt.show()