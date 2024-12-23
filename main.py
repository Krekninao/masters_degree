import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os
from side import read_file, quad
import numpy as np
import pandas as pd
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

def draw_accum(signal, t, accums):
    ''' Отрисовка графиков исходного сигнала и результата функции quad_finder - накоплений амплитуды (аргумент accums)'''
    plt.figure(figsize=(15, 6))
    # накопилось на последнем участке
    # plt.plot(phi_x)
    # первые 10_000 значений
    # plt.plot(phi_total[:10000])
    # plt.title('Сигнал и накопление')
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Оригинальный сигнал')
    plt.ylabel('Амплитуда')
    plt.xlabel('Время, с')
    plt.subplot(2, 1, 2)
    plt.plot(t[:len(accums)], accums)
    plt.title('Накопление')
    plt.ylabel('Амплитуда')
    plt.xlabel('Время, с')

    # Установка расстояния между подграфиками
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    #file_title = f'Электричка, {file_name}-компонента, размер окна - {win_size}c'
    # file_title = f'Электричка, {file_name} - компонента, размер окна - 0_5c'
    # directory = r'C:\Users\user\Desktop\Магистерская\Результаты квадратурного фильтра\Квадратурный фильтр' \
    #              + f'\Электричка\\0_5'

    # file_title = f'Зашумленные данные(поезд), размер окна - {win_size}c'
    # directory = r'C:\Users\user\Desktop\Магистерская\Результаты квадратурного фильтра\Суперсигналы' \
    #             + f'\Зашумленные данные(поезд)\\{win_size}'
    # file_title = f'Электрички, размер окна - 0_5с'
    # directory = r'C:\Users\user\Desktop\Магистерская\Результаты квадратурного фильтра\Суперсигналы' \
    #             + f'\Электрички\\0_5'
    # if not os.path.exists(directory):
    #   os.mkdir(directory)
    # plt.savefig(directory + f'\\{file_title}')

def quad_finder(signal, t, sampl_rate, win_sizes = [1], window = 300, file_title = '---'):
    '''Обнаружение полезного сигнала при помощи накоплений
        Аргументы:
            - signal - значения исходного сигнала
            - t - значения времени (временная ось)
            - sampl_rate - частота дискретизации исходного сигнала
            - win_sizes - размеры промежутков квадратурных накоплений
            - window - окно для квадратурного алгоритма (T) УТОЧНИТЬ!!! '''
    # Создаем экземпляр ExcelWriter один раз
    with pd.ExcelWriter(f'{file_title}.xlsx', engine='xlsxwriter') as writer:
        for win_size in win_sizes:
            # Используем np.arange для создания списка секунд с плавающей запятой
            seconds = list(
                np.arange(0, int(t[-1]) + win_size, win_size))  # int(t_x[-1]) + 1 заменено на int(t_x[-1]) + win_size
            R_total, phi_total = [], []
            spectrs = []
            time_intervals = list(zip(seconds, seconds[1:]))
            for start, finish in time_intervals:
                # выделяем сигнал на отрезке
                sig_part = signal[int(start * sampl_rate): int(finish * sampl_rate) - 1]
                # определяем на этом отрезке доминирующую частоту
                max_fr = spectr(sig_part, len(sig_part), sampl_rate)
                spectrs.append(max_fr)
                # передаем эту частоту как опорную в квадратурный алгоритм
                R_x, phi_x, x1_list = quad(sig_part, max_fr, window, sampl_rate)
                # присоединяем возвращенный кусочек накоплений в общий список накоплений
                R_total.extend(R_x)
                phi_total.extend(phi_x)
            #plt.figure(figsize=(15, 8))
            #plt.plot(phi_total)
            #plt.show()
            # После завершения обработки всех временных интервалов
            # Создаем DataFrame с началом, концом и частотой
            df = pd.DataFrame({
                'Начало, с': seconds[:-1],  # Начало интервала
                'Конец, с': seconds[1:],  # Конец интервала
                'Частота, Гц': spectrs  # Список частот
            })

            # Записываем DataFrame в Excel-файл на отдельный лист
            df.to_excel(writer, index=False, sheet_name=str(win_size))

            # Получаем объект workbook и worksheet
            workbook = writer.book
            worksheet = writer.sheets[str(win_size)]

            # Создаем график
            chart = workbook.add_chart({'type': 'column'})  # 'column' для столбчатого графика

            # Настраиваем график
            chart.add_series({
                'categories': '={}!$A$2:$A${}'.format(str(win_size), len(df) + 1),  # Начало интервала
                'values': '={}!$C$2:$C${}'.format(str(win_size), len(df) + 1),  # Частота
                'name': 'Значение модальной частоты'
            })

            # Скрытие легенды
            chart.set_legend({'position': 'none'})

            # Устанавливаем размер графика
            chart.set_size({'width': 800, 'height': 400})

            # Задаем заголовок графика
            title = f'Динамика модальной частоты исходного сигнала (размер интервала накопления - {win_size}с)'
            chart.set_title({'name': title})
            chart.set_x_axis({'name': 'Время, с'})
            chart.set_y_axis({'name': 'Частота, Гц'})

            # Вставляем график в лист
            worksheet.insert_chart('E2', chart)  # Позиция вставки графика (E2)

            #строим график для модальных частот и сохраняем его
            # plt.figure(figsize=(15, 6))
            # plt.bar(seconds[:-1], spectrs)
            # plt.title(title)
            # plt.xlabel('Время, с')
            # plt.ylabel('Частота, Гц')
            # plt.savefig(f'{file_title} (окно - {win_size}с)')
            #plt.savefig(f'Электричка (окно - 0_5с)')
            #plt.show()

            #Отрисовака графика накоплений
            #draw_accum(signal, t, R_total)
            return seconds[:-1], spectrs

#path = r'C:\Users\user\Desktop\Магистерская\Быстровка_07_07_10_Круг_6км\all_Unit\3seans_19'
#path = r'C:\Users\user\Desktop\Магистерская\ДАННЫЕ СТУДЕНТАМ\Грузовые поезда — копия\1\x'
#path = r'C:\Users\user\Desktop\Магистерская\ДАННЫЕ СТУДЕНТАМ\Грузовые поезда — копия\freq_19_10301410.U8822Vk00'
#path = r'C:\Users\user\Desktop\Магистерская\ДАННЫЕ СТУДЕНТАМ\Электрички\x_elec_19_10302210.U8822Vk00'

#зашумлённые данные
#path = r'C:\Users\user\Desktop\Магистерская\Зашумленные данные\0100200271.01x'
#path = r'C:\Users\user\Desktop\Магистерская\Зашумленные данные\0100200272.01y'
#path = r'C:\Users\user\Desktop\Магистерская\Зашумленные данные\0100200273.01z'
#path = r'C:\Users\user\Desktop\Магистерская\ДАННЫЕ СТУДЕНТАМ\Электрички'

#изменение направление слэшей для корректной работы
#path = path.replace('\\', '/')
# считывание данных из файлов
#sampl_rate_x, sampl_num_x, T_x, t_x, signal_x = read_file(path)

# #суперсигнал
# path = r'C:\Users\user\Desktop\Магистерская\ДАННЫЕ СТУДЕНТАМ\Электрички — копия'
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
def add_noise(signal, target_snr):
    # Рассчитываем необходимый стандартное отклонение шума
    noise_std = signal.max() / target_snr
    # Генерируем шум
    noise = np.random.normal(0, noise_std, signal.shape)
    # Возвращаем сигнал с добавленным шумом
    return signal + noise

def get_starts_stops(spectrs):
    #скользящее окно определяющее начало и конец полезного сигнала
    series = pd.Series(spectrs)
    roll = series.rolling(10).mean()

    found = roll < 7
    result = (found != found.shift(-1))[:-1]
    startstop = result[result].index

    plt.figure()
    plt.bar(range(len(spectrs)), spectrs)
    plt.scatter(startstop, series[startstop], color='red', marker='*')
    plt.savefig('data.png')

    plt.figure()
    roll.plot()
    plt.scatter(startstop, roll[startstop], color='red', marker='*')
    plt.savefig('dataroll.png')

    for a, b in zip(startstop[::2], startstop[1::2]):
        print(f'{a}:{b}')


#размер окна
#win_size = 1

path = r'C:\Users\user\Desktop\Магистерская\ДАННЫЕ СТУДЕНТАМ\Грузовые поезда — копия\1\x'
#path = r'C:\Users\user\Desktop\Магистерская\ДАННЫЕ СТУДЕНТАМ\Грузовые поезда — копия\2\x'
#path = r'C:\Users\user\Desktop\Магистерская\ДАННЫЕ СТУДЕНТАМ\Электрички — копия\z'
#path = r'C:\Users\user\Desktop\Магистерская\ДАННЫЕ СТУДЕНТАМ\Зашумленные данные\x'

#изменение направление слэшей для корректной работы
path = path.replace('\\', '/')
# считывание данных из файлов
sampl_rate_x, sampl_num_x, T_x, t_x, signal_x = read_file(path)

#параметр - соотношение сигнал:шум
target_snr = 10
signal_x = add_noise(signal_x, target_snr)
plt.figure(figsize=(15, 8))
# real_snr = signal_x.max() / signal_x.std()

plt.title(f'Соотношение сигнал:шум = {target_snr}')
plt.plot(signal_x)
plt.show()
win_sizes = [1]
sec, spectrs = quad_finder(signal_x, t_x, sampl_rate_x, win_sizes, file_title = f'-Грузовые поезда 1(зашумление с соотношением {target_snr})')

#plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 1)
plt.plot(t_x, signal_x)
plt.title(f'Соотношение сигнал:шум = {target_snr}')
plt.ylabel('Амплитуда')
plt.xlabel('Время, с')
plt.subplot(2, 1, 2)
plt.plot(sec, spectrs)
plt.title('Модальные частоты')
plt.ylabel('Частота, Гц')
plt.xlabel('Время, с')

# Установка расстояния между подграфиками
plt.subplots_adjust(hspace=0.5)
#plt.show()

get_starts_stops(spectrs)
