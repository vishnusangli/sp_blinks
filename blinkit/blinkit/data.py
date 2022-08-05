# %%
import pandas as pd
import numpy as np

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
np.set_printoptions(precision=8)

def clean(df: pd.DataFrame, row = 'CH1\tCH2'):
    """
    Clean the raw DataFrame from input file
    
    """
    vals = []
    for elem in df.iloc[:, 0]:
        newval = elem
        if type(newval) == str:
            newval = newval.split()
        vals.append(newval)

    vals_start = np.where(df.iloc[:, 0] == row)[0][0]
    info = vals[:vals_start - 1]
    cols, data = vals[vals_start], vals[vals_start + 1:]
    data = np.array(data, dtype=float)
    df = pd.DataFrame(data[100:], columns = cols)
    return info, df


def approx_equal(x, val, e = 1e-5):
    """
    Approximation Equality checker, checking: \\
        abs(x - val) <= e
        
    """
    return np.power(x, 2) - val <= np.power(e, 2)

def point_locate(arr: list, points: list):
    """
    Selects the arr's points indexed at points 
    """
    return_arr = [arr[i] for i in points]
    return return_arr

def range_separate(arr: list, points: list):
    """
    Slices arr based on points' break point indeces

    Returns  2 lists of lists: \n
        `arr_list`: sliced arrays
        `elem_list`: original index values of `arr_list`
    """
    arr_list = []
    elem_list = []

    first_elem = min(0, points[0])
    for end_elem in points:
        if end_elem > first_elem:
            elem_list.append(np.arange(first_elem, end_elem))
            arr_list.append(arr[first_elem: end_elem])

            first_elem = end_elem
    return arr_list, elem_list

def include_ends(arr, points):
    use_arr = point_locate(arr, points)
    use_points = list(points.copy())
    if points[0] != 0:
        use_arr.insert(0, arr[0])
        use_points.insert(0, 0)
    if points[-1] != len(arr) -1:
        end_val = len(arr)-1
        use_arr.append(arr[end_val])
        use_points.append(end_val)
    return use_points, use_arr


def exclude_points(arr, exclude_points, window_threshold =20):
    """
    Return points in array that are not in the excluded points. \n
    This is the unoptimized counterpart to `complement_ends` function.
    """
    elems = []
    new_arr = []
    for i in range(len(arr)):
        if i not in exclude_points:
            if len(elems) == 0 or i - elems[-1][-1] > window_threshold:
                elems.append([i])
                new_arr.append([arr[i]])
            else:
                elems[-1].append(i)
                new_arr[-1].append(arr[i])
    return elems, new_arr

def complement_ends(arr: list, exclude_points: list, window_threshold: int = 20):
    """
    Output ranges of given array that does not exist within excluded points. If 
    valid point exists at most `window_threshold` distance from the previous range, 
    it is included in the previous range. \n
    
    When excluding points with low gradients, this method outputs ranges of 
    high variance, thereby providing potential blinks.
    """
    copy_exclude_points = exclude_points.copy()
    lims = []
    recent_inclusion = -1
    for i in range(len(arr)):
        if i != copy_exclude_points[0]:
            if len(lims) == 0:
                lims.append([i])
            elif i - recent_inclusion > window_threshold: 
                lims[-1].append(recent_inclusion)
                lims.append([i])
            recent_inclusion = i
        else:
            copy_exclude_points = copy_exclude_points[1:]
    if len(lims) > 0 and len(lims[-1]) == 1:
        lims[-1].append(i)
    return lims

def normalize(arr):
    """Basic normlization that divides by range
    """
    minimum, maximum = np.min(arr), np.max(arr)
    return np.divide(arr, maximum)

class ChangePointDetect:
    def savgol_method(arr, e = 1e-14):
        """
        Uses savgol_filter first derivative to find local 
        maximas and minimas.

        Returns a list of minimas, maximas
        """
        first_deriv = savgol_filter(arr, window_length= 31, polyorder= 2, deriv = 1)
        second_deriv = savgol_filter(arr, window_length= 31, polyorder= 2, deriv = 2)

        chg_points = np.where(approx_equal(first_deriv, 0, e =e))[0]
        def group_blinks(arr, first_deriv, sec_deriv, points):
            """
            Supposed to compress the blink special points into single points -
            """
            point_groups = [] #grouping similar points together
            minimas = []
            maximas = []

            for elem in points:
                if len(point_groups) == 0:
                    point_groups.append([elem])
                    continue
                if elem - point_groups[-1][0] > 30: #Sample number difference
                    point_groups.append([elem])
                else:
                    point_groups[-1].append(elem)
            for grp in point_groups:
                if sec_deriv[grp[0]] <= 0: #maxima
                    maximas.append(max(grp))
                else:
                    minimas.append(min(grp))
            return minimas, maximas
        minimas, maximas = group_blinks(arr, first_deriv, second_deriv, chg_points)
        return minimas, maximas
    
    def window_grad(arr, threshold_rate = 1e-10, window_size = 20, check_size = 10, back = True):
        """
        find places where the absolute rate changes through the 
        threshold rate within the widnow size
        Here, we take back derivative/slope by default

        Gradient of a point is calculated w/ respect to another point (window_size) away
        Change in gradient is found with the gradients of two points (check_size) apart

        Returns minimas, maximas
        """
        forward_slope = lambda x: np.divide(arr[x + window_size] - arr[x], window_size)
        backward_slope = lambda x: np.divide(arr[x] - arr[x-window_size], window_size)
        increase_arr = []
        decrease_arr = []
        slope_calc = forward_slope
        if back:
            slope_calc = backward_slope
        for elem in range(len(arr)):
            if slope_calc(elem) < threshold_rate and slope_calc(min(elem + check_size, len(arr) - 1)) > threshold_rate:
                increase_arr.append(elem)
            
            if slope_calc(elem) < -threshold_rate and slope_calc(min(elem + check_size, len(arr) - 1)) > -threshold_rate:
                increase_arr.append(elem)
            
            if slope_calc(elem) > threshold_rate and slope_calc(min(elem + check_size, len(arr) - 1)) <= threshold_rate:
                decrease_arr.append(elem)
        return increase_arr, decrease_arr

def padded_interp(arr: list, points: list, kind: str = "linear"):
    """
    Performs interpolation on points array, 
    returning exact arry for elem indeces outside points range

    Standardization form this results in a complete cancellation 
    of regions outside range
    """
    min_elem, max_elem = points[0], points[-1]
    interp_func = interp1d(points, point_locate(arr, points), kind = kind)
    def sub_func(x):
        if x < min_elem or x > max_elem:
            return arr[x]
        return interp_func(x)
    sub_func = np.vectorize(sub_func)
    return sub_func(range(len(arr)))


#### BLINKS ####
class Episode:
    def __init__(self) -> None:
        """
        Creates an Episode object with 
        """
        pass
class Blink:
    def __init__(self) -> None:
        pass

def center_blink(blink_arr, value = 0):
    """
    With time series of blinks as input, this method outputs 
    a shifted x, y pair such that the blink maximum is centered
    at `value`. \n
    Default `value` = 0
    """
    max_val = np.where(blink_arr == np.max(blink_arr))[0]
    elem_arr = np.arange(0, len(blink_arr)) - max_val + value
    return elem_arr, blink_arr

def lims_change(arr: list, blink_lims: list, e = 1e-2):
    """
    Minor function to tweak blink windows such that the blinks 
    appropriately end when voltage values approximate to 0 within 
    `e` threshold.
    """
    for elem in range(int(np.average(blink_lims)), blink_lims[-1]):
        if arr[elem] <= e:
            blink_lims = [blink_lims[0], elem]
    for elem in range(blink_lims[0], int(np.average(blink_lims))):
        if arr[elem] >= e:
            blink_lims = [elem, blink_lims[1]]
            break
    return blink_lims

def filter_blinks(arr: list, blink_lims:list, threshold = 0.5, duration = 1000):
    """
    Primary-step blink filter that discriminates based on duration and peak voltage.
        `threshold`: float in [0, 1] above which blinks are accepted.
        `duration`: array-index-wise filter within which blinks are accepted.
    """
    def filt(lims):
        if lims[1] - lims[0] <= 30:
            return False
        return np.all(~pd.isna(lims)) and lims[1] - lims[0] <= duration and np.max(arr[int(lims[0]):int(lims[1])]) >= threshold
    return np.array([filt(i) for i in blink_lims])

def group_blinks(arr, blink_lims, e = 50):
    """
    An attempt at recognizing double blinks. If any blinks are within `e`
    indeces of eather other, they are grouped together.
    """
    new_blink_lims = []
    for i in range(len(blink_lims)):
        if len(new_blink_lims) == 0:
            new_blink_lims.append(blink_lims[i])

        if blink_lims[i][0] - new_blink_lims[-1][1] <= e:
            #If this blink is close enough to the previous
            new_blink_lims[-1][1] = blink_lims[i][1] #Combine blinks
        else:
            new_blink_lims.append(blink_lims[i])
    return new_blink_lims

def detrend_standardize(arr, detrend_method = ChangePointDetect.savgol_method):
    minimas, maximas = ChangePointDetect.savgol_method(arr, e = 1e-6)
    detrend_arr = padded_interp(arr, minimas)
    dtd_data = arr - detrend_arr
    return normalize(dtd_data)
# %%
