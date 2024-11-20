import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os
from side import read_file, quad
import numpy as np
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

#path = r'C:\Users\user\Desktop\Магистерская\Быстровка_07_07_10_Круг_6км\all_Unit\3seans_19'
path = r'C:\Users\user\Desktop\Магистерская\ДАННЫЕ СТУДЕНТАМ\Грузовые поезда — копия\1\x'
#path = r'C:\Users\user\Desktop\Магистерская\ДАННЫЕ СТУДЕНТАМ\Грузовые поезда — копия\freq_19_10301410.U8822Vk00'
#path = r'C:\Users\user\Desktop\Магистерская\ДАННЫЕ СТУДЕНТАМ\Электрички\x_elec_19_10302210.U8822Vk00'

#зашумлённые данные
#path = r'C:\Users\user\Desktop\Магистерская\Зашумленные данные\0100200271.01x'
#path = r'C:\Users\user\Desktop\Магистерская\Зашумленные данные\0100200272.01y'
#path = r'C:\Users\user\Desktop\Магистерская\Зашумленные данные\0100200273.01z'
#path = r'C:\Users\user\Desktop\Магистерская\ДАННЫЕ СТУДЕНТАМ\Электрички'

#изменение направление слэшей для корректной работы
path = path.replace('\\', '/')
# считывание данных из файлов
sampl_rate_x, sampl_num_x, T_x, t_x, signal_x = read_file(path)

# #суперсигнал
# path = r'C:\Users\user\Desktop\Магистерская\ДАННЫЕ СТУДЕНТАМ\Зашумленные данные'
# path_x = path + '\\x'
# path_y = path + '\\y'
# path_z = path + '\\z'
# sampl_rate_x, sampl_num_x, T_x, t_x, signal_x = read_file(path_x)
# sampl_rate_x, sampl_num_x, T_x, t_x, signal_y = read_file(path_y)
# sampl_rate_x, sampl_num_x, T_x, t_x, signal_z = read_file(path_z)
# signal_x = signal_x + signal_y + signal_z

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


# #размер окна
# #win_size = 1
# win_sizes = [1, 2, 3, 5, 10]
# window = 300
# for win_size in win_sizes:
#   for file_name in ['x', 'y', 'z']:
#     path = r'C:\Users\user\Desktop\Магистерская\ДАННЫЕ СТУДЕНТАМ\Зашумленные данные' + f'\\{file_name}'
#     path = path.replace('\\', '/')
#     sampl_rate_x, sampl_num_x, T_x, t_x, signal_x = read_file(path)
#
#     seconds = list(range(0, int(t_x[-1]) + 1, win_size))
#     phi_total = []
#     for start, finish in zip(seconds, seconds[1:]):
#       #выделяем сигнал на отрезке в 5 секунд
#       #обдумать момент с концом интервала!
#       sig_part = signal_x[start * sampl_rate_x : finish * sampl_rate_x-1]
#       #определяем на этом отрезке доминирующую частоту
#       max_fr = spectr(sig_part, len(sig_part), sampl_rate_x)
#       #передаём эту частоту как опорную в квадратурный алгоритм
#       R_x, phi_x, x1_list = quad(sig_part, max_fr, window, sampl_rate_x)
#       #присоединяем возвращённый кусочек накоплений в общий список накоплений
#       phi_total.extend(R_x)
#
#     plt.figure(figsize=(15, 6))
#     #накопилось на последнем участке
#     #plt.plot(phi_x)
#     #первые 10_000 значений
#     #plt.plot(phi_total[:10000])
#     #plt.title('Сигнал и накопление')
#     plt.subplot(2, 1, 1)
#     plt.plot(t_x, signal_x)
#     plt.title('Оригинальный сигнал')
#     plt.ylabel('Амплитуда')
#     plt.xlabel('Время, с')
#     plt.subplot(2, 1, 2)
#     plt.plot(t_x[:len(phi_total)], phi_total)
#     plt.title('Накопление')
#     plt.ylabel('Амплитуда')
#     plt.xlabel('Время, с')
#
#     # Установка расстояния между подграфиками
#     plt.subplots_adjust(hspace=0.5)
#     #plt.show()
#
#     # file_title = f'Грузовые поезда 1, {file_name}-компонента, размер окна - {win_size}c'
#     # directory = r'C:\Users\user\Desktop\Магистерская\Результаты квадратурного фильтра\Квадратурный фильтр' \
#     #             + f'\Грузовые поезда 1\\{win_size}'
#
#     file_title = f'Зашумленные данные, {file_name}-компонента, размер окна - {win_size}c'
#     directory = r'C:\Users\user\Desktop\Магистерская\Результаты квадратурного фильтра\Квадратурный фильтр' \
#                 + f'\Зашумленные данные\\{win_size}'
#     if not os.path.exists(directory):
#       os.mkdir(directory)
#     plt.savefig(directory + f'\\{file_title}')

#размер окна
#win_size = 1

win_sizes = [10.78]
window = 300
for win_size in win_sizes:
    # Используем np.arange для создания списка секунд с плавающей запятой
    seconds = list(
        np.arange(0, int(t_x[-1]) + win_size, win_size))  # int(t_x[-1]) + 1 заменено на int(t_x[-1]) + win_size
    phi_total = []

    for start, finish in zip(seconds, seconds[1:]):
        # выделяем сигнал на отрезке
        sig_part = signal_x[int(start * sampl_rate_x): int(finish * sampl_rate_x) - 1]
        # определяем на этом отрезке доминирующую частоту
        max_fr = spectr(sig_part, len(sig_part), sampl_rate_x)
        # передаем эту частоту как опорную в квадратурный алгоритм
        R_x, phi_x, x1_list = quad(sig_part, max_fr, window, sampl_rate_x)
        # присоединяем возвращенный кусочек накоплений в общий список накоплений
        phi_total.extend(R_x)

    plt.figure(figsize=(15, 6))
    #накопилось на последнем участке
    #plt.plot(phi_x)
    #первые 10_000 значений
    #plt.plot(phi_total[:10000])
    #plt.title('Сигнал и накопление')
    plt.subplot(2, 1, 1)
    plt.plot(t_x, signal_x)
    plt.title('Оригинальный сигнал')
    plt.ylabel('Амплитуда')
    plt.xlabel('Время, с')
    plt.subplot(2, 1, 2)
    plt.plot(t_x[:len(phi_total)], phi_total)
    plt.title('Накопление')
    plt.ylabel('Амплитуда')
    plt.xlabel('Время, с')

    # Установка расстояния между подграфиками
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    # file_title = f'Грузовые поезда 1, {file_name}-компонента, размер окна - {win_size}c'
    # directory = r'C:\Users\user\Desktop\Магистерская\Результаты квадратурного фильтра\Квадратурный фильтр' \
    #             + f'\Грузовые поезда 1\\{win_size}'

    # file_title = f'Зашумленные данные(поезд), размер окна - {win_size}c'
    # directory = r'C:\Users\user\Desktop\Магистерская\Результаты квадратурного фильтра\Суперсигналы' \
    #             + f'\Зашумленные данные(поезд)\\{win_size}'
    # if not os.path.exists(directory):
    #   os.mkdir(directory)
    # plt.savefig(directory + f'\\{file_title}')