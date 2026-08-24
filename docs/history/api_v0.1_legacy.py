import numpy as np
import pandas as pd
import scipy
import scipy.integrate
import scipy.ndimage
import scipy.signal
from derivative import dxdt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math
from decimal import Decimal

import seaborn as sns

eps = 10 ** (-6)

# для анализа скоростей глаз при длительном рассматривании
# логнормальное распределение
def lognormal_distribution(x, mu, sigma=0.8):
    return 1 / (x * sigma * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))

def lognormal_distribution_no_sigma(x, mu):
    sigma=0.78
    return 1 / (x * sigma * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))


#класс для аппроксимации ломаной
class biline_regression():
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.par_0 = [0, 0, 0]
        self.a_0, self.a_1, self.b_0 = *(self.par_0),

    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        X = X[~np.isnan(Y)]
        Y = Y[~np.isnan(Y)]
        X = X[~np.isinf(Y)]
        Y = Y[~np.isinf(Y)]
        X = X[1:]
        Y = Y[1:]
        base_tr = min(X) + self.threshold*(max(X)-min(X))
        base_tr = np.argmin(abs(X-base_tr))
        self.threshold = base_tr/len(X)
        res = minimize(
            lambda x: np.sum((X[:int(len(X)*self.threshold)]*x[0]+x[2] - Y[:int(len(X)*self.threshold)])**2)
                  + np.sum((x[1]*X[int(len(X)*self.threshold):]+X[int(len(X)*self.threshold)]*(x[0]-x[1])+x[2] - Y[int(len(X)*self.threshold):])**2),
            (self.a_0, self.a_1, self.b_0), method='nelder-mead')
        self.a_0, self.a_1, self.b_0 = *(res.x),
        sns.set(rc={'figure.figsize':(14,5)}) ## здесь размер картинки
        sns.set(font_scale=1) ## а здесь размер подписей
        fig, ax = plt.subplots()
        ax.set_xlabel('ln(Δt), c')
        ax.set_ylabel('ln(f(Δt))')

        fig = sns.scatterplot(x=X, y=Y)
        fig = sns.lineplot(x=X[:int(len(X)*self.threshold)+1], y=self.a_0*X[:int(len(X)*self.threshold)+1]+self.b_0)
        fig = sns.lineplot(x=X[int(len(X)*self.threshold):], y=self.a_1*X[int(len(X)*self.threshold):]
                           +X[int(len(X)*self.threshold)]*(self.a_0-self.a_1)+self.b_0)
    def get_der(self):
        return (self.a_0, self.a_1)

# расчет скорости с приведением к выбранной частоте
def eye_velocity_by_fr(time_s: np.array, eye_data_x, eye_data_y, fr=250):
    """
    # скорость осредненная по выбранной частоте
    :param time_s: вектор времени, время может быть неравномерное
    :param eye_data_x: x координата
    :param eye_data_y: y координата
    :param fr: частота, на которой будет расчитана скорость
    :return:  моменты времени - вектор, мгновенные скорости - осредененные по частоте Fr
    """
    fr_max = len(time_s) / (np.max(time_s) - np.min(time_s))
    if fr > fr_max:
        fr = int(fr_max)
    t = np.linspace(np.min(time_s), np.max(time_s), int((np.max(time_s) - np.min(time_s)) * fr))
    eye_x = np.interp(t, time_s, eye_data_x)
    eye_y = np.interp(t, time_s, eye_data_y)
    dx = eye_x[1:] - eye_x[:-1]
    dy = eye_y[1:] - eye_y[:-1]
    distance = [np.sqrt(i ** 2 + j ** 2) for i, j in zip(dx, dy)]
    velocity = [i * fr for i in distance]
    return t, velocity


def ksv(time_s, eye_data_x, eye_data_y, v0=27, fr=250):
    """
    :param time_s: вектор времени, время может быть неравномерное
    :param eye_data_x: x координата
    :param eye_data_y: y координата
    :param v0: скорость, до которой считается, что взор стабилен
    :param fr: частота, на которой будет расчитана скорость
    :return: коэффицент стабилизации взора
    """
    t_stab = 0
    t, v = eye_velocity_by_fr(time_s, eye_data_x, eye_data_y, fr)
    for i in range(len(v)):
        if v[i] <= v0:
            t_stab += 1 / fr
    t_0 = np.max(time_s) - np.min(time_s)
    k = t_stab / t_0
    return k


def tr_length(eye_data, t1, t2, keyx, keyy):
    '''
    eye_data -- содержит 'T_s', keyx, keyy
    t1 -- начало интервала анализа
    t2 -- конец интервала анализа
    '''
    # Фильтрация массива по значению времени в диапазоне [t1, t2]
    filtered_data = eye_data[(eye_data['T_s'] >= t1) & (eye_data['T_s'] <= t2)].copy()
    filtered_data = filtered_data.reset_index(drop=True)
    # Вычисление разностей координат x и y для соседних точек
    dx = [i - j for i, j in
          zip(filtered_data[keyx][1:len(filtered_data)], filtered_data[keyx][0:len(filtered_data) - 1])]
    dy = [i - j for i, j in
          zip(filtered_data[keyy][1:len(filtered_data)], filtered_data[keyy][0:len(filtered_data) - 1])]
    # Вычисление длины каждого сегмента и суммирование
    segment_lengths = [np.sqrt(i ** 2 + j ** 2) for i, j in zip(dx, dy)]
    total_length = np.sum(segment_lengths)

    return total_length


def find_std_fx(eye_data, t1, t2, key_pos):
    """
    :param eye_data: данные глаз с ключами: 'key_pos' и  'T_s'
    :param t1:  время начала отрезка анализа
    :param t2: время окончания отрезка анализа
    :param key_pos: ключ данных которые надо обработать
    :return: std
    """
    filtered_data = eye_data[(eye_data['T_s'] >= t1) & (eye_data['T_s'] <= t2)].copy()
    return np.std(filtered_data[key_pos])


def max_increment_deg(fr, v_max=900):
    """
    :param fr: частота, на которой будет проходить анализ
    :param v_max: макисмальноя скорость допустимая для глаза
    :return: максимальное допустимое приращение координат на заданной частоте
    """
    return v_max / fr


def eye_params(fr, n=10):
    """
    #вычисление максимальной детектируемой скорости фиксации и продолжительности "дырок в фиксациях"
    fr -- частота съема данных
    n  -- минимальное чило точек на детектируемой саккаде
    """
    if n < 5:
        n = 5
    t = n * 1 / fr
    t_min = 0.01
    a_min = (t - t_min) / 2
    a_min_0 = 0.1
    if a_min <= a_min_0:
        a_min = a_min_0
        vs_max = 15
        vf_max = 5
        max_delta_time = (n / 2) * t_min
    else:
        vs_max = np.sqrt(a_min) * 100
        vf_max = 5 / a_min_0 * a_min
        max_delta_time = (n / 2) * (1 / fr)
    return a_min * 180 / np.pi, vs_max, vf_max, max_delta_time


def find_full_angel(eye_data_cur, x_key, y_key, k=1, x0=0, y0=0, ):
    """
    :param eye_data_cur: даныне глаза, минимально есть x_key, y_key
    :param k: калибровочный коэффициент для перевода px -> mm (по умолчанию не требуется)
    :param x0: нейтральгное положение по x, по умолчанию = 0
    :param y0: нейтральгное положение по y, по умолчанию = 0
    :param x_key: название колонки с x координатой
    :param y_key: название колонки с y координатой
    :return: dp -- смещение глаза в метрических величинах, alpha -- угол относитльно начального положения x0, y0
    """
    dp = [k * np.sqrt((x - x0) ** 2 + (y - y0) ** 2) for x, y in zip(eye_data_cur[x_key], eye_data_cur[y_key])]
    s_alpha = [(y - y0) / np.sqrt((x - x0) ** 2 + (y - y0) ** 2) for x, y in
               zip(eye_data_cur[x_key], eye_data_cur[y_key])]
    c_alpha = [(x - x0) / np.sqrt((x - x0) ** 2 + (y - y0) ** 2) for x, y in
               zip(eye_data_cur[x_key], eye_data_cur[y_key])]
    alpha = np.zeros(len(s_alpha))
    for i, a in enumerate(alpha):
        alpha[i] = np.atan2(s_alpha[i], c_alpha[i])
    return dp, alpha


def make_full_coord(eye_data, key_x, key_y, k=1, x0=0, y0=0):
    """
    :param eye_data: даныне глаза, минимально есть x_key, y_key
    :param k: калибровочный коэффициент для перевода px -> mm (по умолчанию не требуется)
    :param x0: нейтральгное положение по x, по умолчанию = 0
    :param y0: нейтральгное положение по y, по умолчанию = 0
    :param key_x: название колонки с x координатой
    :param key_y: название колонки с y координатой
    """
    dp, alpha = find_full_angel(eye_data, k, x0, y0, key_x, key_y)
    full_coord = pd.DataFrame(columns=['T_s', 'pos', 'dpos'])
    full_coord['T_s'] = eye_data['T_s']
    full_coord['pos'] = scipy.signal.medfilt(np.array([np.atan2(i, 12) * 180 / np.pi for i in dp]), 3)
    full_coord['alpha'] = scipy.signal.medfilt(alpha, 3)
    full_coord['dpos'] = dxdt(full_coord['pos'].to_numpy(), eye_data['T_s'].to_numpy(), kind="savitzky_golay",
                              left=.05, right=.05, order=3)
    return full_coord


def find_saccade(full_coord, max_peak=15, win=20):
    """
    :param full_coord: full_coord['T_s', 'pos', 'alpha', 'dpos'] -- полные смещения глазного яблока
    :param max_peak: --
    :param win:
    :return:
    """
    peak_ind = scipy.signal.find_peaks(np.abs(full_coord['dpos']), height=max_peak)
    saccade_list = np.zeros([7, len(peak_ind[0])]).transpose()
    for n, i in enumerate(peak_ind[0]):
        if (i > win) and ((i + win) < len(full_coord['dpos'])):
            a1 = full_coord['dpos'][i - win:i]
            a2 = full_coord['dpos'][i + 1:i + win]
            a0 = full_coord['dpos'][i]
            try:
                ts_s = full_coord['T_s'][np.flip(a1)[np.flip(a1) * a0 > eps].index[0]]
                ts_f = full_coord['T_s'][a2[a2 * a0 > eps].index[0]]
                a_s = full_coord['pos'][np.flip(a1)[np.flip(a1) * a0 > eps].index[0]]
                a_f = full_coord['pos'][a2[a2 * a0 > eps].index[0]]
                alpha_s = full_coord['alpha'][np.flip(a1)[np.flip(a1) * a0 > eps].index[0]]
                alpha_f = full_coord['alpha'][a2[a2 * a0 > eps].index[0]]
                saccade_list[n] = [ts_s, ts_f, a0, a_s, a_f, alpha_s, alpha_f]
            except:
                pass
    saccade_list = pd.DataFrame(saccade_list,
                                columns=['t_start', 't_finish', 'max_vel', 'A_s', 'A_f', 'alpha_s', 'alpha_f'])
    saccade_list = saccade_list.loc[(saccade_list['t_finish'] != 0)]
    saccade_list = saccade_list.reset_index(drop=True)
    return saccade_list


def find_fixation(full_coord, saccade_list, max_vel=5, max_delta_time=0.006):
    """
    full_coord -- np.array [T_s        pos      dpos]
    saccade_list -- список выделенных саккад, морганий, явных нефиксаций [t_start, t_finish]
    max_vel -- максимальная скорость галаза на фиксации
    max_delta_time -- максимальный интервал времени на котором может нарушаться скорость
    """
    fixations_list = np.zeros([2, 10 * len(saccade_list)]).transpose()
    n = 0
    for t1, t2 in zip([saccade_list['t_start'][0] - 10 * max_delta_time] + saccade_list['t_finish'].to_list(),
                      saccade_list['t_start'].to_list() + [len(saccade_list['lat']) - 1, ]):
        fixations_max = full_coord[np.abs(full_coord['dpos']) < max_vel]
        fixations_max = fixations_max[fixations_max['T_s'] >= t1]
        fixations_max = fixations_max[fixations_max['T_s'] <= t2]
        fixations_max = fixations_max.reset_index(drop=True)
        if len(fixations_max) > 2:
            dfixations_max = np.array([i - j for i, j in zip(fixations_max['T_s'][1:], fixations_max['T_s'][:-1])])
            fixations_list[n][0] = fixations_max['T_s'][0]
            for i, dt in enumerate(dfixations_max):
                if dt < max_delta_time:
                    pass
                else:
                    fixations_list[n][1] = fixations_max['T_s'][i + 1]
                    n = n + 1
                    fixations_list[n][0] = fixations_max['T_s'][i + 1] + dt
            fixations_list[n][1] = fixations_max['T_s'][len(fixations_max['T_s']) - 1]
            n = n + 1
    fixations_list = pd.DataFrame(fixations_list, columns=['t_start', 't_finish'])
    fixations_list = fixations_list.loc[(fixations_list['t_finish'] != 0)]
    fixations_list = fixations_list.reset_index(drop=True)
    return fixations_list

def star_eye_smoothing(a: np.array, #окулографические данные
                       saccade_solid, # максимальное приращение на саккаде
                       eps, # отступ от максимального и минимального значения (область поиска глаза)
                       max_data=None, #максимальное достоверное значение (недостижимое)
                       min_data=None, #минимальное достоверное значение (недостижимое)
):
    '''
    сглаживает данные в любы единицах
    '''
    if max_data == None:
        max_data=max(a)
    if min_data == None: 
        min_data=min(a)    
    b=[a[0]]
    for k, i, j in zip(a[0:-3], a[1:-2], a[2:-1]):
        if min_data+eps<i<max_data-eps:
            if abs(i-k)<saccade_solid:
                 b.append(np.median([b[-1], i, j]))
            else:
                 b.append(b[-1])
        else: b.append(b[-1])
    return b


def vive_eye_smoothing(x, y, saccade_solid, eps, z=None, max_data=None):
    '''
    x - eye coord in RAID-axis (to ear)
    y - eye coord in RAID-axis (to head top)
    z - eye coord optional (forward)
    saccade_solid - максимальное приращение угла на саккаде в зависимости от частоты записи -- из фунции eye_params
    eps - отступ от максимального и минимального значения (область поиска глаза)
    max_data - максимальное достоверное значение (недостижимое)
    '''
    x_=[x[0]+star_eye_smoothing(x,  saccade_solid, eps, max_data)]
    y_=[y[0]+star_eye_smoothing(y,  saccade_solid, eps, max_data)]
    x_.append(x[-1])
    y_.append(y[-1])
    if z != None :
        z_=[z[0]+star_eye_smoothing(z,  saccade_solid, eps, max_data)]
        z_.append(z[-1])
    else:
        z_ = None
    return [x_, y_, z_]

def find_central_position(x, y, z=None):
    x_c = np.nanmedian(x)
    y_c = np.nanmedian(y)
    if z:
        z_c = np.nanmedian(z)
    else:
        z_c = None
    return [x_c, y_c, z_c]

def make_okulogram(x, x_c = None, r_eye=12):
    '''
    функция возращает откулограмму в градусах
    x - данные о смещении глаза в милиметрах
    x_c - среднее положение, если оно записывалось отдельно
    '''
    if x_c == None:
        x_c = np.nanmedian(x)
    res=[np.arctan2((i-x_c), r_eye)*180/np.pi for i in x]
    return res


#######################################################
# Блок для анализ данных типа "средний глаз" ==  gaze


def find_gaze_fixation(full_coord, max_vel=15, max_delta_time=0.07, max_peak = 25, fr = None):
    """
    full_coord -- pd.DataFrame ['T_s', 'pos', 'dpos'] ([time/angle/derev 1 from angle])
    max_vel -- максимальная скорость взора на фиксации в градусах, если 'pos', 'dpos' в град
    max_delta_time -- максимальный интервал времени на котором может нарушаться скорость в единицах что и 'T_s'
    max_peak -- минимальная скорость взора, которая считается быстрым движением
    fr -- частота
    """
    if fr:
        if max_delta_time<1/fr: max_delta_time = 2*1/fr
            
    full_coord['dpos'] = scipy.signal.medfilt(full_coord['dpos'], 3)
    peak_ind = scipy.signal.find_peaks(np.abs(full_coord['dpos']), height=max_peak)
    fast_move_list = np.zeros([5, len(peak_ind[0])]).transpose()
    Fr = len(full_coord['T_s'])/np.max(full_coord['T_s'])
    win = 2*int(max_delta_time*Fr)
    i_end = 0
    for n, i in enumerate(peak_ind[0]):
        if (i > win) and ((i + win) < len(full_coord['dpos'])) and (i_end < i-win):
            i_max, v_max = np.argmax(np.abs(full_coord['dpos'][i - win:i + win])), np.max(np.abs(full_coord['dpos'][i - win:i + win]))
            i_max += i - win
            #win_pross = ищется исходя из v_max и Fr -- 1/2 максимльного времени соотв саккады
            win_pross = win
            if i_max < win_pross:
                ts_s = full_coord['T_s'][0]
                a_s = full_coord['pos'][0]
            else:
                ts_s = full_coord['T_s'][i_max - win_pross]
                a_s = full_coord['pos'][i_max - win_pross]
            if i_max + win_pross > len(full_coord['dpos']):
                ts_f = full_coord['T_s'][-1]
                a_f = full_coord['pos'][-1]
            else:
                ts_f = full_coord['T_s'][i_max + win_pross]
                a_f = full_coord['pos'][i_max + win_pross]
            if ts_f - ts_s > 2*max_delta_time:
                fast_move_list[n] = [ts_s, ts_f, full_coord['dpos'][i_max], a_s, a_f]
            i_end = i_max + win_pross
    fast_move_list = pd.DataFrame(fast_move_list,
                                columns=['t_start', 't_finish', 'max_vel', 'A_s', 'A_f'])
    fast_move_list = fast_move_list.loc[(fast_move_list['t_finish'] != 0)]
    fast_move_list = fast_move_list.reset_index(drop=True)
    fast_move_list['lat'] = fast_move_list['t_finish'] - fast_move_list['t_start']

    fixations_list = np.zeros([2, 10*len(fast_move_list)]).transpose()
    n = 0
    for t1, t2 in zip([fast_move_list['t_start'][0] - 10 * max_delta_time] + fast_move_list['t_finish'].to_list(),
                      fast_move_list['t_start'].to_list() + [len(fast_move_list['t_start']) - 1, ]):
        fixations_max = full_coord[np.abs(full_coord['dpos']) < max_vel]
        fixations_max = fixations_max[fixations_max['T_s'] >= t1]
        fixations_max = fixations_max[fixations_max['T_s'] <= t2]
        fixations_max = fixations_max.reset_index(drop=True)
        if len(fixations_max) > 2:
            dfixations_max = np.array([i - j for i, j in zip(fixations_max['T_s'][1:], fixations_max['T_s'][:-1])])
            fixations_list[n][0] = fixations_max['T_s'][0]
            for i, dt in enumerate(dfixations_max):
                if dt < max_delta_time:
                    pass
                else:
                    fixations_list[n][1] = fixations_max['T_s'][i + 1]
                    n = n + 1
                    fixations_list[n][0] = fixations_max['T_s'][i + 1] + dt
            fixations_list[n][1] = fixations_max['T_s'][len(fixations_max['T_s']) - 1]
            n = n + 1
    fixations_list = pd.DataFrame(fixations_list, columns=['t_start', 't_finish'])
    fixations_list = fixations_list.loc[(fixations_list['t_finish'] != 0)]
    fixations_list = fixations_list.reset_index(drop=True)
    fixations_list['lat'] = fixations_list['t_finish'] - fixations_list['t_start']
    return fast_move_list, fixations_list



## это убирается из этого модуля
#########################################################
#Кватернионная алгебра для стандартных способов записи масс-айтрекеров

from typing import NamedTuple
import math


class Vector3(NamedTuple):
    x: float
    y: float
    z: float

    @staticmethod
    def empty():
        return Vector3(0.0, 0.0, 0.0)

    def __add__(self, other):
        if isVector3(other):
            return Vector3(*[k + v for k, v in zip(self, other)])
        else:
            raise TypeError("Vector3 should be given")
    __radd__ = __add__

    def __neg__(self):
        return Vector3(*[- k for k in self])

    def __sub__(self, other):
        if isVector3(other):
            return Vector3(*[k - v for k, v in zip(self, other)])
        else:
            raise TypeError("Vector3 should be given")

    def __rsub__(self, other):
        if isVector3(other):
            return Vector3(*[k - v for k, v in zip(other, self)])
        else:
            raise TypeError("Vector3 should be given")

    def __mul__(self, other):
        if isNumber(other):
            return Vector3(*[k * other for k in self])
        elif isVector3(other):
            return sum(k * v for k, v in zip(self, other))
    __rmul__ = __mul__

    def __div__(self, other):
        if isNumber(other):
            return Vector3(*[k / (1. * other) for k in self])
        else:
            raise TypeError("Number should be given")
    __truediv__ = __div__

    def __rdiv__(self, other):
        raise TypeError("Can't divide by a vector")

    def __len__(self):
        return 3

    @property
    def norm(self) -> float:
        return math.sqrt(math.pow(self.x, 2) + math.pow(self.y, 2) + math.pow(self.z, 2))

    @property
    def normalized(self):
        n = self.norm
        if n == 0:
            raise ZeroDivisionError("Can't normalize a zero-length vector")
        return self / n

    @staticmethod
    def cross(v1, v2):
        if isVector3(v1) and isVector3(v2):
            return Vector3(v1.y * v2.z - v1.z * v2.y,
                           v1.z * v2.x - v1.x * v2.z,
                           v1.x * v2.y - v1.y * v2.x)
        else:
            raise TypeError("Cross product with non-vector")

    @staticmethod
    def triple_prod(v1, v2, v3) -> float:
        if all(map(lambda x: isVector3(x), [v1, v2, v3])):
            return v1 * Vector3.cross(v2, v3)
        else:
            raise TypeError("Triple product with non-vector")


class Matrix(NamedTuple):
    a11: float
    a12: float
    a13: float
    a21: float
    a22: float
    a23: float
    a31: float
    a32: float
    a33: float

    @staticmethod
    def empty():
        return Matrix(0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0)


class Quaternion(NamedTuple):
    w: float
    x: float
    y: float
    z: float

    def __add__(self, other):
        if isQuaternion(other):
            return Quaternion(*[k + v for k, v in zip(self, other)])
        else:
            raise TypeError("Quaternion should be given")
    __radd__ = __add__

    def __sub__(self, other):
        if isQuaternion(other):
            return Quaternion(*[k - v for k, v in zip(self, other)])
        else:
            raise TypeError("Quaternion should be given")

    def __rsub__(self, other):
        if isQuaternion(other):
            return Quaternion(*[k - v for k, v in zip(other, self)])
        else:
            raise TypeError("Quaternion should be given")

    def __mul__(self, other):
        if isNumber(other):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        elif isQuaternion(other):
            w = (self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z)
            x = (self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y)
            y = (self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x)
            z = (self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w)
            return Quaternion(w=w, x=x, y=y, z=z)
        else:
            raise ValueError('Not yet implemented')

    def __rmul__(self, other):
        if isNumber(other):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        return self.__class__(other) * self

    def __div__(self, other):
        if isNumber(other):
            return Quaternion(*[k / (1. * other) for k in self])
        else:
            raise TypeError("Number should be given")
    __truediv__ = __div__

    def __rdiv__(self, other):
        raise ValueError('Division by quaternions is not allowed')

    def __eq__(self, other):
        """Returns true if the following is true for each element:
        `absolute(a - b) <= (atol + rtol * absolute(b))`
        """
        if not isinstance(other, type(self)):
            return NotImplemented

        r_tol = 1.0e-13
        a_tol = 1.0e-14
        return all(math.fabs(item - iota) <= (a_tol + r_tol * math.fabs(iota)) for item, iota in zip(self, other))

    def __len__(self):
        return 4

    @property
    def vector(self) -> Vector3:
        return Vector3(self.x, self.y, self.z)

    @property
    def scalar(self) -> float:
        return self.w

    @property
    def conjugate(self):
        return Quaternion(w=self.w, x=self.x * -1., y=self.y * -1., z=self.z * -1.)

    @property
    def norm(self):
        return math.sqrt(sum(map(lambda x: math.pow(x, 2), [k for k in self])))

    @property
    def normalized(self):
        n = self.norm
        if n == 0:
            raise ZeroDivisionError("Can't normalize with a zero quaternion")
        return self / n

    @property
    def inverse(self):
        return self.conjugate / math.pow(self.norm, 2)

    @staticmethod
    def empty():
        return Quaternion(w=1, x=0.0, y=0.0, z=0.0)

    @staticmethod
    def get_rot_quaternion(m: Matrix):
        if m == Matrix.empty():
            return Quaternion.empty()

        eta = math.expm1(1e-5)

        if (m.a11 + m.a22 + m.a33) > eta:
            q1 = (1 / 2) * math.sqrt(1 + (m.a11 + m.a22 + m.a33))
        else:
            q1 = (1 / 2) * math.sqrt(
                (
                        math.pow((m.a32 - m.a23), 2)
                        + math.pow((m.a13 - m.a31), 2)
                        + math.pow((m.a21 - m.a12), 2)
                ) / (3 - (m.a11 + m.a22 + m.a33))
            )

        if (m.a11 - m.a22 - m.a33) > eta:
            q2 = (1 / 2) * math.sqrt(1 + (m.a11 - m.a22 - m.a33))
        else:
            q2 = (1 / 2) * math.sqrt(
                (
                        math.pow((m.a32 - m.a23), 2)
                        + math.pow((m.a12 + m.a21), 2)
                        + math.pow((m.a31 + m.a13), 2)
                ) / (3 - (m.a11 - m.a22 - m.a33))
            )

        if (- m.a11 + m.a22 - m.a33) > eta:
            q3 = (1 / 2) * math.sqrt(1 + (- m.a11 + m.a22 - m.a33))
        else:
            q3 = (1 / 2) * math.sqrt(
                (
                        math.pow((m.a13 - m.a31), 2)
                        + math.pow((m.a12 + m.a21), 2)
                        + math.pow((m.a23 + m.a32), 2)
                ) / (3 - (- m.a11 + m.a22 - m.a33))
            )

        if (- m.a11 - m.a22 + m.a33) > eta:
            q4 = (1 / 2) * math.sqrt(1 + (- m.a11 - m.a22 + m.a33))
        else:
            q4 = (1 / 2) * math.sqrt(
                (
                        math.pow((m.a21 - m.a12), 2)
                        + math.pow((m.a31 + m.a13), 2)
                        + math.pow((m.a32 + m.a23), 2)
                ) / (3 - (- m.a11 - m.a22 + m.a33))
            )

        return Quaternion(w=q1, x=q2, y=q3, z=q4)

    @staticmethod
    def get_angle(q1, q2):
        if isQuaternion(q1) and isQuaternion(q2):
            q12 = q1.conjugate * q2
            angle = 2 * math.atan2(q12.vector.norm, q12.scalar)
            return angle
        else:
            raise TypeError("Quaternions should be given")


def isVector3(item) -> bool:
    return isinstance(item, Vector3)


def isMatrix(item) -> bool:
    return isinstance(item, Matrix)


def isQuaternion(item) -> bool:
    return isinstance(item, Quaternion)


def isNumber(item) -> bool:
    return isinstance(item, (int, float))

def quat_scale(q, val):
    res= Quaternion(q.w * val, q.x * val, q.y * val, q.z * val)
    return res

def quat_length(q):
    quat_length = (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)**0.5
    return    quat_length


def quat_normalize(q):
    n = quat_length(q)
    quat_normalize = quat_scale(q, 1 / n)
    return quat_normalize

def quat_invert(q):
    res =Quaternion(q.w, -q.x, -q.y, -q.z)
    quat_invert = quat_normalize(res)
    return quat_invert

    
def quat_mul_vector(a, b):
    c=Quaternion(-a.x * b.x - a.y * b.y - a.z * b.z,
                 a.w * b.x + a.y * b.z - a.z * b.y,
                 a.w * b.y - a.x * b.z + a.z * b.x,
                 a.w * b.z + a.x * b.y - a.y * b.x)
    return c
    
def quat_mul_quat(a, b):
    res=Quaternion(a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
                   a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                   a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                   a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w)
    return res

def quat_transform_vector(q, v):
    t = quat_mul_vector(q, v)
    t1 = quat_mul_quat(t, quat_invert(q))
    res=Vector3(t1.x, t1.y, t1.z)
    return res



def get_x(data):
    return [item.x for item in data]


def get_y(data):
    return [item.y for item in data]


def get_z(data):
    return [item.z for item in data]


def get_w(data):
    return [item.w for item in data]


def str2bool(v):
    return v.lower() in ("yes", "true", 'y', "t", "1")


def declist_from_vec(s):
    return [Decimal(i) for i in s.strip('()').split(',')]


def floatlist_from_vec(s):
    return [float(i) for i in s.strip('()').split(',')]


def get_vec3_from_list(list, convert_l2r=False) -> Vector3:
    if convert_l2r:
        return Vector3(x=list[0],
                       y=list[2],
                       z=list[1])
    else:
        return Vector3(x=list[0],
                       y=list[1],
                       z=list[2])


def get_quat_from_list(list, convert_l2r=False) -> Quaternion:
    if convert_l2r:
        return Quaternion(x=list[0],
                          y=list[2],
                          z=list[1],
                          w=list[3])
    else:
        return Quaternion(x=list[0],
                          y=list[1],
                          z=list[2],
                          w=list[3])

def euler_from_quaternion(quat: Quaternion, convert2deg=False):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (quat.w * quat.x + quat.y * quat.z)
    t1 = +1.0 - 2.0 * (quat.x * quat.x + quat.y * quat.y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (quat.w * quat.y - quat.z * quat.x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (quat.w * quat.z + quat.x * quat.y)
    t4 = +1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
    yaw_z = math.atan2(t3, t4)
    
    if convert2deg:
        return Vector3(x=math.degrees(roll_x), y=math.degrees(pitch_y), z=math.degrees(yaw_z)) # in radians
    else:
        return Vector3(x=roll_x, y=pitch_y, z=yaw_z) # in radians

def get_euler_from_quats(list, convert2deg=False):
    return [euler_from_quaternion(i, convert2deg) for i in list]

