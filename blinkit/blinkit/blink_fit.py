"""
This file holds the mothods for fitting and characterizing a blink
Here numerous fitting methods could be applied using different fit functions


"""

import numpy as np
import scipy
import neurokit2 as nk


def fit_gamma(x, loc, a, scale):
    x = nk.rescale(x, to=[0, 10])
    gamma = scipy.stats.gamma.pdf(x, a=a, loc=loc, scale=scale)
    y = gamma / np.max(gamma)
    return y


def fit_scr(x, time_peak, rise, decay1, decay2):
    x = nk.rescale(x, to=[0, 10])
    gt = np.exp(-((x - time_peak) ** 2) / (2 * rise ** 2))
    ht = np.exp(-x / decay1) + np.exp(-x / decay2)

    ft = np.convolve(gt, ht)
    ft = ft[0 : len(x)]
    y = ft / np.max(ft)
    return y

#file:///C:/Users/vishn/Downloads/brosch-blink-characterization-using-2017.pdf
def fit_func(x, a0, a1, a2, b, c):
    return (a0 * np.power(x, 2)) + (a1 * np.power(x, 3)) + (a2 * np.power(x, 4)) + np.exp(-b * np.power(x, c))