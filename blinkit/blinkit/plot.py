import matplotlib.patches as patches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from blinkit import data, blink_fit

def create_rect(arr: list, elem_lim: list, fill:bool = False, alpha: float = 1, linewidth: int = 2):
    rec_width = elem_lim[1]- elem_lim[0]
    max_y, min_y = max(arr[elem_lim[0]: elem_lim[1]]), min(arr[elem_lim[0]],  arr[elem_lim[1]])
    rect_height = max_y - min_y
    anchor_pt = (elem_lim[0], min_y)
    rect_obj = patches.Rectangle(anchor_pt, rec_width, rect_height, fill = fill, alpha = alpha, lw = linewidth)
    return rect_obj

def convert_img(elem_arr, blink_arr, ylims = [0, 1, 0.002], xlims = [-200, 200, 1], fill= False):
    """
    
    """
    y_num = int(np.divide(ylims[1] - ylims[0], ylims[2]))
    x_num = int(np.divide(xlims[1] - xlims[0], xlims[2]))
    img = np.zeros(shape = (y_num, x_num)) #ylim inclusive, so 1 is reachable
    def y_convert(val):
        conv_val = np.divide(val, ylims[2])
        if conv_val < 0: return 0
        elif conv_val >= y_num: return y_num-1
        return int(conv_val)
    xrange_min = int(np.where(elem_arr == max(elem_arr[0], xlims[0]))[0])
    xrange_max = int(np.where(elem_arr == min(elem_arr[-1], xlims[1]))[0])
    for elem in range(xrange_min, xrange_max, xlims[2]):
        use_elem = int(np.divide(elem_arr[elem] - xlims[0], xlims[2]))
        pos = y_num - y_convert(blink_arr[elem])
        #pos = ynum - int(ynum * blink_arr[elem])
        if not fill:
            img[pos, use_elem] = 1
        else:
            img[pos:, use_elem] = 1
    return img

def plot_blinks(blink_imgs, name = None, figsize = (10, 8), numplots = (2, 3)):
    fig, ax = plt.subplots(nrows = numplots[0], ncols= numplots[1], 
                            figsize = figsize, facecolor = 'gray')
    for i in range(numplots[0] * numplots[1]):
        r = i//numplots[1]
        c = i % numplots[1]
        ax[r][c].set_xticks([])
        ax[r][c].set_yticks([])
        if i >= len(blink_imgs):
            continue
        ax[r][c].imshow(blink_imgs[i], cmap ='gray')

    plt.tight_layout()
    if name != None:
        plt.suptitle(f"Valid blinks for {name}")
    plt.show()

def plot_compare(blink_eog: list, fits_df: pd.DataFrame, blinks_df: pd.DataFrame, blink_num: int, fit_func):
    curr_blink = blinks_df.iloc[blink_num]
    curr_fit = fits_df.iloc[blink_num]
    num_params = fit_func.num_params

    p0 = [curr_fit[f'param_{i}'] for i in range(num_params)]
    start, end = int(curr_blink["Onsets"]), int(curr_blink["Offsets"])
    plt.plot(data.normalize( blink_eog[start:end]), label = "blink", alpha = 0.7)
    temp = blink_fit.fitfunc_wrapper(fit_func.func, p0)
    plt.plot(temp(range(end - start)), label = "fit", alpha = 0.7)