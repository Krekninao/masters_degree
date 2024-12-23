import numpy as np
import matplotlib.pyplot as plt
from main import quad_finder

# Параметры
duration = 50  # Длительность сигнала в секундах
sample_rate = 100  # Частота дискретизации
A = 1.0  # Амплитуда
omega_0 = 2 * np.pi * 2  # Начальная угловая частота (2 Гц)
b = 0.5  # Коэффициент для изменения фазы Гц/c
start_time = 17  # Время, когда сигнал начинает появляться (в секундах)

# Временная ось
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Генерация шумовой части
noise = 0.1 * A * np.random.normal(size=t.shape)  # Уровень шума 10%

# Генерация сигнала
signal = A * np.cos(omega_0 * t + (b * t**2) / 2)

# Условие для того, чтобы сигнал начинал появляться только после start_time
# Мы используем np.where, чтобы сохранить только значения до указанного времени
registrated_signal = np.where(t >= start_time, signal + noise, noise)

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(t, registrated_signal, label='Регистрируемый сигнал', color='blue')
plt.axvline(x=start_time, color='red', linestyle='--', label='Начало сигнала')  # Вертикальная линия для указания времени появления сигнала
plt.title('Имитация сигнала разгоняющегося автомобиля')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid()
plt.xlim(0, duration)  # Ограничиваем ось x по длине сигнала
plt.ylim(-1.5, 1.5)    # Ограничиваем ось y для лучшей видимости
plt.legend()
plt.show()

win_sizes = [1, 2, 3, 5, 10]
quad_finder(registrated_signal, t, sample_rate, win_sizes, window = 300, file_title = 'Имитация сигнала разгоняющегося автомобиля')
