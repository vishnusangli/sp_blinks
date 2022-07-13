"""
This file holds the mothods for fitting and characterizing a blink
Here numerous fitting methods could be applied using different fit functions


"""

import numpy as np
import neurokit2 as nk
from scipy.optimize import curve_fit
from scipy.stats import gamma

class fit_gamma:
    norm_p0 = [1, 1, 1]
    num_params = 3
    def func(x, loc, a, scale):
        x = nk.rescale(x, to=[0, 10])
        gamma = gamma.pdf(x, a=a, loc=loc, scale=scale)
        y = gamma / np.max(gamma)
        return y

class fit_scr:
    norm_p0 = [1, 1, 1, 1]
    num_params = 4
    def func(x, time_peak, rise, decay1, decay2):
        x = nk.rescale(x, to=[0, 10])
        gt = np.exp(-((x - time_peak) ** 2) / (2 * rise ** 2))
        ht = np.exp(-x / decay1) + np.exp(-x / decay2)

        ft = np.convolve(gt, ht)
        ft = ft[0 : len(x)]
        y = ft / np.max(ft)
        return y


class paper_func:
    norm_p0 = [ 1.07681633e-04, -1.10660735e-06,  2.19228462e-09,  6.31139713e-01,
       -1.90463734e-01]
    num_params = 4
    #file:///C:/Users/vishn/Downloads/brosch-blink-characterization-using-2017.pdf
    def func(x, a0, a1, a2, b, c):
        return (a0 * np.power(x, 2)) + (a1 * np.power(x, 3)) + (a2 * np.power(x, 4)) + np.exp(-b * np.power(x, c))

def fitfunc_wrapper(func, p0):
    """
    Decorator function that vectorizes a fit function to corresponding 
    parameters
    """
    first = lambda x: func(x, *p0)
    sec = np.vectorize(first)
    return sec

def lsqr_fit(blink_eog: list, func_class, p0: list = None, sampling_rate = 1e3, centering = True):
    """
    Center and fit the blink to the function using least squares
    """
    x = np.linspace(0, np.divide(len(blink_eog) - 1, sampling_rate), len(blink_eog))
    if centering:
        max_index = np.where(blink_eog== np.max(blink_eog))[0]
        x = x - x[max_index] # Center blink at 0
    if p0 is None:
        p0 = func_class.norm_p0
    try:
        popt, pcov = curve_fit(func_class.func, x, blink_eog, p0 = p0)
        temp_func = fitfunc_wrapper(func_class.func, popt)
        return popt, pcov, x, temp_func(x)
    except Exception as e:
        print("Optimum not found")
        print(e)
    return None, None, None, None