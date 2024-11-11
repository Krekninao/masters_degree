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

#num_file = 19
#path = r'C:\Users\user\Desktop\Магистерская\Быстровка_07_07_10_Круг_6км\all_Unit\3seans_19'
path = r'C:\Users\user\Desktop\Магистерская\ДАННЫЕ СТУДЕНТАМ\Грузовые поезда\freq_1_2_19_10302210.U8822Vk00'
#path = r'C:\Users\user\Desktop\Магистерская\ДАННЫЕ СТУДЕНТАМ\Грузовые поезда\freq_19_10301410.U8822Vk00'
#path = r'C:\Users\user\Desktop\Магистерская\ДАННЫЕ СТУДЕНТАМ\Электрички\x_elec_19_10302210.U8822Vk00'

#зашумлённые данные
#path = r'C:\Users\user\Desktop\Магистерская\Зашумленные данные\0100200271.01x'
#path = r'C:\Users\user\Desktop\Магистерская\Зашумленные данные\0100200272.01y'
#path = r'C:\Users\user\Desktop\Магистерская\Зашумленные данные\0100200273.01z'


path = path.replace('\\', '/')
# считывание данных из файлов
#sampl_rate_x, sampl_num_x, T_x, t_x, signal_x = read_file(path + f'/U.{num_file}x')
sampl_rate_x, sampl_num_x, T_x, t_x, signal_x = read_file(path)

# #зашумленный сгенерированный сигнал
# # Параметры сигнала
# fs = 1000  # Частота дискретизации, Гц
# t = np.linspace(0, 10, fs * 10)  # Время от 0 до 10 секунд
# freq_signal = 1  # Частота полезного сигнала, Гц
# amplitude_signal = 5  # Амплитуда полезного сигнала
# amplitude_noise = amplitude_signal / 5  # Амплитуда шума для соотношения 5:1
#
# # Генерация белого шума
# noise = amplitude_noise * np.random.normal(0, 1, len(t))
#
# # Создание пустого массива для полезного сигнала
# useful_signal = np.zeros_like(t)
#
# # Задаем интервалы времени, где присутствует полезный сигнал (например, от 2 до 3 и от 5 до 6 секунд)
# signal_start_times = [2, 5]
# signal_end_times = [3, 6]
#
# # Заполнение массива полезного сигнала
# for start, end in zip(signal_start_times, signal_end_times):
#     index_start = np.searchsorted(t, start)
#     index_end = np.searchsorted(t, end)
#     useful_signal[index_start:index_end] = amplitude_signal * np.sin(2 * np.pi * freq_signal * t[index_start:index_end])
#
# # Комбинирование полезного сигнала и шума
# combined_signal = useful_signal + noise
# t_x, signal_x = t, combined_signal


#for i in range(len(signal_x))
#размер окна
win_size = 10
window = 300
seconds = list(range(0, int(t_x[-1]) + 1, win_size))
phi_total = []
for start, finish in zip(seconds, seconds[1:]):
  #выделяем сигнал на отрезке в 5 секунд
  #обдумать момент с концом интервала!
  sig_part = signal_x[start * sampl_rate_x : finish * sampl_rate_x-1]
  #определяем на этом отрезке доминирующую частоту
  max_fr = spectr(sig_part, len(sig_part), sampl_rate_x)
  print(max_fr)
  #передаём эту частоту как опорную в квадратурный алгоритм
  R_x, phi_x, x1_list = quad(sig_part, max_fr, window, sampl_rate_x)
  #присоединяем возвращённый кусочек накоплений в общий список накоплений
  phi_total.extend(R_x)

plt.figure(figsize=(15, 6))
#накопилось на последнем участке
#plt.plot(phi_x)
#первые 10_000 значений
#plt.plot(phi_total[:10000])
#plt.title('Сигнал и накопление')
plt.subplot(2, 1, 1)
plt.plot(t_x, signal_x)
plt.subplot(2, 1, 2)
plt.plot(t_x[:len(phi_total)], phi_total)
plt.show()