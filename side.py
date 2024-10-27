import numpy as np
import matplotlib.pyplot as plt


def read_file(file_name):
    import struct
    # with open('/content/drive/MyDrive/Магистерская/test_x.U8822Vk00', 'rb') as f:
    with open(file_name, 'rb') as f:
        # my_lines = f.readlines()
        ID = f.read(2)  # Идентификатор формата
        reserv = f.read(4)  # Зарезервировано
        width = struct.unpack('<f', f.read(4))[0]  # Широта
        longitude = struct.unpack('<f', f.read(4))[0]  # Долгота
        scale = struct.unpack('<d', f.read(8))[0]  # Коэффициент пересчета в физическую величину
        year = ord(struct.unpack('<c', f.read(1))[0])  # Год начала записи
        month = ord(struct.unpack('<c', f.read(1))[0])  # Месяц начала записи
        day = ord(struct.unpack('<c', f.read(1))[0])  # День начала записи
        hour = ord(struct.unpack('<c', f.read(1))[0])  # Час начала записи
        minute = ord(struct.unpack('<c', f.read(1))[0])  # Минута начала записи
        second = ord(struct.unpack('<c', f.read(1))[0])  # Секунда начала записи
        mcsecond = struct.unpack('<i', f.read(4))[0]  # Поправка в микросекундах к времени начала записи
        sampl_rate = struct.unpack('<h', f.read(2))[0]  # Частота дискретизации в Гц
        sampl_num = struct.unpack('<i', f.read(4))[0]  # Число отсчетов в трассе
        sampl_type = struct.unpack('<h', f.read(2))[0]  # Длина и тип отсчета:
        tr_num = ord(struct.unpack('<c', f.read(1))[0])  # Номер трассы в исходном файле
        reserved_end = ord(struct.unpack('<c', f.read(1))[0])  # Зарезервировано
        import numpy as np

        # Задаем параметры системы
        fs = sampl_rate  # Частота дискретизации
        f0 = 10  # Частота гармонического сигнала ???
        # T = sampl_type  # Длительность сигнала
        # T = 5000
        T = sampl_num // fs
        # t = np.linspace(0, T, T*fs)  # Временная ось ???
        t = np.linspace(0, T, sampl_num)

        # Генерируем сигнал
        x = np.zeros(sampl_num)
        for i in range(sampl_num):
            if sampl_type == 4:
                x[i] = struct.unpack('<i', f.read(4))[0]
            if sampl_type == 2:
                x[i] = struct.unpack('<h', f.read(2))[0]
        # Возвращаем основную информацию: частоту дискретизации, число отсчётов, длительность сигнала, временную ось и сами значения сигнала
        return sampl_rate, sampl_num, T, t, x


# квадратурный алгоритм
def quad(input_sig, target_freq, T, sample_rate):
    target_omega = 2 * np.pi * target_freq
    input_size = len(input_sig)
    dt = 1 / sample_rate
    t = np.arange(0, input_size * dt, dt)
    z_x = input_sig * np.sin(target_omega * t)
    z_y = input_sig * np.cos(target_omega * t)
    gamma = dt / T

    x = 0
    y = 0
    R = np.zeros(input_size)
    phi = np.zeros(input_size)
    #список значений x
    x_list = np.zeros(input_size)
    for i in range(input_size):
        x = x + gamma * (z_x[i] - x)
        y = y + gamma * (z_y[i] - y)
        R[i] = np.sqrt(x ** 2 + y ** 2)
        x_list[i] = x
        if x != 0:
            phi[i] = np.arctan(y / x)
        else:
            phi[i] = 0
    # Далее должен быть код для построения графика амплитуды сигнала во времени, но он закомментирован
    # plt.plot(t_x, phi)
    # plt.show()
    return R, phi, x_list


# # путь к папке
# path = r'C:\Users\user\Desktop\Магистерская\Быстровка_07_07_10_Круг_6км\all_Unit\3seans_19'
# path = path.replace('\\', '/')
# # считывание данных из файлов
# sampl_rate_x, sampl_num_x, T_x, t_x, signal_x = read_file(path + f'/U.19x')
# sampl_rate_y, sampl_num_x, T_y, t_y, signal_y = read_file(path + f'/U.19y')
# #
# # # для демонстрации работы вадратурного алгоритма
# R_x, phi_x, x_list = quad(signal_x, 9.5, 100, 200)
# R_y, phi_y, x_list = quad(signal_y, 9.5, 100, 200)

##################################################################################################################
# перебор всех частот, окон и файлов - построение графиков x от y

# искомые частоты
#list_target_freq = [8, 8.5, 9, 9.5, 10, 10.5]
list_target_freq = [9.5]
#list_target_freq = [8]
# ширина окна
#list_windows = [50, 100, 200, 300, 400, 500, 600]
list_windows = [300]
# номера файлов Быстровки типа U19.x
#num_files = [19, 20, 24, 25, 26, 27, 28]
num_files = [19]

for target_freq in list_target_freq:
    for num_file in num_files:
        for window in list_windows:
            print(f'Частота: {target_freq}, номер файла: U{num_file}, размер окна: {window}')

            # путь к папке
            path = r'C:\Users\user\Desktop\Магистерская\Быстровка_07_07_10_Круг_6км\all_Unit\3seans_19'
            path = path.replace('\\', '/')
            # считывание данных из файлов
            sampl_rate_x, sampl_num_x, T_x, t_x, signal_x = read_file(path + f'/U.{num_file}x')
            sampl_rate_y, sampl_num_y, T_y, t_y, signal_y = read_file(path + f'/U.{num_file}y')

            # применение квадратурного фильтра
            R_x, phi_x, x1_list = quad(signal_x, target_freq, window, 200)  # как автоматически определять 200???
            R_y, phi_y, x2_list = quad(signal_y, target_freq, window, 200)

            # номер частоты (первая - под номером 0)
            k = (target_freq - 8) // 0.5
            # длина кусочка с учётом частоты дискретизации
            lenght_part = len(t_x) // 6

            start = int(lenght_part * k)
            finish = int(lenght_part * (k + 1))
            print(start, finish)
            plt.figure()  # создание нового окна для графика
            #графики для накоплений амплитуды
            #plt.scatter(R_x[start: finish], R_y[start: finish], s=1)
            #графики для накоплений начальных фаз
            plt.scatter(phi_x[2000*200: 2500*200], phi_y[2000*200: 2500*200], s=1)
            # plt.title(f'Частота: {target_freq}, номер файла: U{num_file}, размер окна: {window}')
            # plt.xlabel(f'U.{num_file}x')
            # plt.ylabel(f'U.{num_file}y')
            # plt.grid()
            #
            # # формирование имени файла
            # name_file = str(int(target_freq))
            # # если частота нецелая - дописываем 5
            # if target_freq != int(target_freq):
            #     name_file += '_5'
            # name_file += f'Гц_имя_U{num_file}_размер_окна_{window}'
            # # plt.savefig(r'C:\Users\user\Desktop\Магистерская\Результаты Быстровка 6 км' \
            # #             + f'\\{name_file}')
            plt.show()
