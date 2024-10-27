import matplotlib.pyplot as plt
from statistics import mean, stdev
from math import pi
import numpy as np

#Функция определяющая порог для определения начала накопления
def get_threshold_R(r, n=100000, k=2):
    beg = r[:n]
    # end = r[-n:]
    # avg_r = mean(beg) # среднее по отрезку
    s = 0
    for r in beg:
        s += (r - avg_r) ** 2
    d = (s / (n - 1)) ** 0.5
    # print(f'Сигма = {d}')
    # print(f'Среднее R = {avg_r}')
    # print(round(d, 2)*k)
    # st_dev_r = min(stdev(end), stdev(beg))
    st_dev_r = stdev(beg)
    threshold_r = k * st_dev_r
    print(f'k = {k}, порог = {round(threshold_r, 2)}')
    return avg_r, threshold_r

# Функция для чтения данных из файла
def read_data(file_path):
    x = []
    values = []
    with open(file_path, 'r') as file:
        next(file)  # Пропускаем заголовок
        for line in file:
            parts = line.split()
            x.append(float(parts[0]))
            values.append(float(parts[1]))
    return x, values
def draw_chart(x_r, r, x_phi, phi, x_start, x_end, frame):
    # Построение графиков
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    v_discr = 200
    plt.plot(x_r[(x_start - frame)*v_discr:(x_end + frame)*v_discr], r[(x_start - frame)*v_discr:(x_end + frame)*v_discr], 'r', label='R от x')
    #plt.plot(x_r[(x_start - 2500)*250:(x_end + frame)*250], r[(x_start - 2500)*250:(x_end + frame)*250], 'r', label='R от x')
    plt.xlabel('x')
    plt.ylabel('R')
    plt.legend()

    # Добавление меток и вертикальной линии на первом графике
    plt.axvline(x=x_start, color='b', linestyle='--')
    plt.axvline(x=x_end, color='b', linestyle='--')
    plt.text(x_start, plt.ylim()[1], f'{x_start}', ha='center', va='bottom')
    plt.text(x_end, plt.ylim()[1], f'{x_end}', ha='center', va='bottom')

    plt.subplot(2, 1, 2)
    v_discr = 200
    plt.plot(x_phi[(x_start - frame)*v_discr:(x_end + frame)*v_discr], phi[(x_start - frame)*v_discr:(x_end + frame)*v_discr], 'k', label='phi от x')
    plt.xlabel('x')
    plt.ylabel('phi')
    plt.legend()

    # Добавление меток и вертикальной линии на втором графике
    plt.axvline(x=x_start, color='b', linestyle='--')
    plt.axvline(x=x_end, color='b', linestyle='--')
    plt.text(x_start, plt.ylim()[1], f'{x_start}', ha='center', va='bottom')
    plt.text(x_end, plt.ylim()[1], f'{x_end}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
def get_avg_phi(r_file_path, phi_file_path, range_acc=600, target_freq = 8):
    """
        Главная функция
        range_acc - переменная, задающая диапазон рассмотрения, на котором происходит накопление
    """
    # Путь к файлам
    #r_file_path = 'C:/Users/user/Desktop/Магистерская/Тренировочные данные/x/R_values.txt'
    #phi_file_path = 'C:/Users/user/Desktop/Магистерская/Тренировочные данные/x/phi_values.txt'

    # Чтение данных из файлов
    x_r, r = read_data(r_file_path)
    x_phi, phi = read_data(phi_file_path)
    #Перевод в градусы
    to_degrees = 180/pi
    phi = list(map(lambda x: to_degrees * x, phi))

    res_x = []
    res_r = []
    #avg_r, threshold = get_threshold_R(r, n = 150000, k = 2.962948577038649000)
    # при k = 2.9629485770 - начало в 50 с, при k = 2.9629485771 - начало в 642

    # for i in range(len(x_r)):
    #     if r[i] - avg_r > threshold: # сравниваем с разбросом
    #         res_x.append(x_r[i])
    #         res_r.append(r[i])
    #x_start, x_end = int(res_x[0]), int(res_x[-1])
    x_start = (target_freq - 8)//0.5 * 600
    x_end = x_start + range_acc
    frame = 50 # количество секунд до и после интервала накопления
    # print(x_phi.index(x_start))
    # print(x_phi.index(x_end))
    #print(f'Среднее значение phi на интервале накопления = {round(mean(phi[x_phi.index(x_start):(x_phi.index(x_end)+1)]), 2)}')
    # print(f'Начало накопления: {x_start} c.\nКонец накопления: {x_end} c.')
    # print(f'Среднее значение R на интервале накопления = {round(mean(res_r), 2)}')
    #
    # Построение графиков
    #draw_chart(x_r, r, x_phi, phi, x_start, x_end, frame)
    return round(mean(phi[x_phi.index(x_start):(x_phi.index(x_end)+1)]), 2)
    #обработать концы нормально!!!
    #return round(mean(phi[x_phi.index(x_start):]), 2)

def get_info(num_file = 0, range_acc=600, target_freq = 8):
    phi_x, phi_y, phi_z = 'Not defined', 'Not defined', 'Not defined'

    path = r'C:\Users\user\Desktop\Магистерская\Тренировочные данные\Быстровка_07_07_10_Круг_6км\8.5' \
            + f'\\U{num_file}_окно40'
    #path = r'C:\Users\user\Desktop\Магистерская\Тренировочные данные\Быстровка_11_09_12_Круг_12км\12_09111855.U883d'


    phi_x = get_avg_phi(r_file_path = path + '/x/R_values.txt',\
                phi_file_path = path + '/x/phi_values.txt', range_acc = range_acc, target_freq = target_freq)
    phi_y = get_avg_phi(r_file_path = path + '/y/R_values.txt',\
                phi_file_path = path + '/y/phi_values.txt', range_acc = range_acc, target_freq = target_freq)
    # phi_z = get_avg_phi(r_file_path = 'C:/Users/user/Desktop/Магистерская/Тренировочные данные/z/R_values.txt',\
    #            phi_file_path = 'C:/Users/user/Desktop/Магистерская/Тренировочные данные/z/phi_values.txt')

    # print('phi_x = ', phi_x)
    # print('phi_y = ', phi_y)
    # print('phi_z = ', phi_z)

    return (phi_x, phi_y)
#искомые частоты
#list_target_freq = [8, 8.5, 9, 9.5, 10, 10.5]
list_target_freq = [9.5]
#диапазоны рассмотрения накопления
#list_range_acc = [50, 100, 200, 300, 400, 500, 600]
list_range_acc = [600]
#названия файлов U.19x
#num_files = [19, 20, 24, 25, 26, 27, 28]
num_files = [19]
for target_freq in list_target_freq:
    for range_acc in list_range_acc:
        print(target_freq, range_acc)
        phi_x_list = []
        phi_y_list = []
        for num_file in num_files:
            phi_x, phi_y = get_info(num_file, range_acc, target_freq)
            phi_x_list.append(phi_x)
            phi_y_list.append(phi_y)

        phi_x_list = np.array(phi_x_list)
        phi_y_list = np.array(phi_y_list)

        # Вычисление средних значений
        mean_x = np.mean(phi_x_list)
        mean_y = np.mean(phi_y_list)

        plt.figure()  # создание нового окна для графика
        plt.scatter(phi_x_list, phi_y_list) #Основные точки
        # Добавление крупной красной точки с координатами средних значений
        plt.scatter(mean_x, mean_y, color='red', s=100, label='Среднее значение')
        plt.xlabel('phi_x') #подписи осей
        plt.ylabel('phi_y')
        plt.legend(loc='upper left') #вывод легенды на экран

        #формирование имени файла
        name_file = str(int(target_freq))
        #если частота нецелая - дописываем 5
        if target_freq != int(target_freq):
            name_file += '_5'
        name_file += f'Гц_{range_acc}'

        plt.savefig(r'C:\Users\user\Desktop\Магистерская\Результаты Быстровка 6 км' \
                    + f'\\{name_file}')
        #plt.show()

