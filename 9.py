import matplotlib.pyplot as plt
from statistics import mean, stdev


def get_threshold_R(r, n=100000, k=2):
    beg = r[:n]
    s = 0
    for r_val in beg:
        s += (r_val - avg_r) ** 2
    d = (s / (n - 1)) ** 0.5
    st_dev_r = stdev(beg)
    threshold_r = k * st_dev_r
    print(f'k = {k}, threshold = {round(threshold_r, 2)}')
    return avg_r, threshold_r


def read_data(file_path):
    x = []
    values = []
    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            parts = line.split()
            x.append(float(parts[0]))
            values.append(float(parts[1]))
    return x, values


def draw_chart(ax, x, y, color, label, x_start, x_end, frame):
    v_discr = 200
    ax.plot(x[(x_start - frame) * v_discr:(x_end + frame) * v_discr], y[(x_start - frame) * v_discr:(x_end + frame) * v_discr], color,
            label=label)
    ax.axvline(x=x_start, color='b', linestyle='--')
    ax.axvline(x=x_end, color='b', linestyle='--')
    ax.text(x_start, ax.get_ylim()[1], f'{x_start}', ha='center', va='bottom')
    ax.text(x_end, ax.get_ylim()[1], f'{x_end}', ha='center', va='bottom')


def get_avg_phi(r_file_path, phi_file_path, ax_r, ax_phi):
    x_r, r = read_data(r_file_path)
    x_phi, phi = read_data(phi_file_path)

    x_start = 2400
    x_end = x_start + 600
    frame = 50

    draw_chart(ax_r, x_r, r, 'r', 'R от x', x_start, x_end, frame)
    draw_chart(ax_phi, x_phi, phi, 'k', 'phi от x', x_start, x_end, frame)


fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, figsize=(12, 18))
path = r'C:\Users\user\Desktop\Магистерская\Тренировочные данные\Быстровка_07_07_10_Круг_6км\U19_окно40'

get_avg_phi(path + '/x/R_values.txt', path + '/x/phi_values.txt', ax1, ax2)
get_avg_phi(path + '/y/R_values.txt', path + '/y/phi_values.txt', ax3, ax4)
get_avg_phi('C:/Users/user/Desktop/Магистерская/Тренировочные данные/z/R_values.txt',
            'C:/Users/user/Desktop/Магистерская/Тренировочные данные/z/phi_values.txt', ax5, ax6)

plt.tight_layout()
plt.show()
