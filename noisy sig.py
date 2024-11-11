import numpy as np
import matplotlib.pyplot as plt

# Параметры сигнала
fs = 1000  # Частота дискретизации, Гц
t = np.linspace(0, 10, fs * 10)  # Время от 0 до 10 секунд
freq_signal = 1  # Частота полезного сигнала, Гц
amplitude_signal = 5  # Амплитуда полезного сигнала
amplitude_noise = amplitude_signal / 5  # Амплитуда шума для соотношения 5:1

# Генерация белого шума
noise = amplitude_noise * np.random.normal(0, 1, len(t))

# Создание пустого массива для полезного сигнала
useful_signal = np.zeros_like(t)

# Задаем интервалы времени, где присутствует полезный сигнал (например, от 2 до 3 и от 5 до 6 секунд)
signal_start_times = [2, 5]
signal_end_times = [3, 6]

# Заполнение массива полезного сигнала
for start, end in zip(signal_start_times, signal_end_times):
    index_start = np.searchsorted(t, start)
    index_end = np.searchsorted(t, end)
    useful_signal[index_start:index_end] = amplitude_signal * np.sin(2 * np.pi * freq_signal * t[index_start:index_end])

# Комбинирование полезного сигнала и шума
combined_signal = useful_signal + noise

# Построение графика
plt.figure(figsize=(14, 6))
plt.plot(t, combined_signal, label='Полезный сигнал + шум', color='blue')
plt.plot(t, useful_signal, label='Полезный сигнал', color='green')
plt.title('Полезный сигнал с шумом (активен в определенные сроки)')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid()
plt.show()
