import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt

def create_rect(arr: list, elem_lim: list, fill:bool = False, alpha: float = 1, linewidth: int = 2):
    anchor_pt = (elem_lim[0], arr[elem_lim[0]])
    rec_width = elem_lim[1]- elem_lim[0]
    rect_height = max(arr[elem_lim[0]: elem_lim[1]])
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