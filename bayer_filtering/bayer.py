import numpy as np
from scipy.signal import convolve2d

def get_bayer_masks(n_rows, n_cols):
    rows = np.tile(np.array([[0,1], [0, 0]], dtype=bool), n_cols // 2)
    if n_cols % 2:
        rows = np.append(rows, np.array([[0], [0]], dtype=bool), axis=1)
    r = np.tile(rows, (n_rows // 2, 1))
    if n_rows % 2 == 1:
        r = np.append(r, [rows[0]], axis = 0)

    rows = np.tile(np.array([[0, 0], [1, 0]], dtype=bool), n_cols // 2)
    if n_cols % 2:
        rows = np.append(rows, np.array([[0], [1]], dtype=bool), axis=1)
    b = np.tile(rows, (n_rows // 2, 1))
    if n_rows % 2 == 1:
        b = np.append(b, [rows[0]], axis = 0)

    rows = np.tile(np.array([[1,0], [0, 1]], dtype=bool), n_cols // 2)
    if n_cols % 2:
        rows = np.append(rows, np.array([[1], [0]], dtype=bool), axis=1)
    g = np.tile(rows, (n_rows // 2, 1))
    if n_rows % 2 == 1:
        g = np.append(g, [rows[0]], axis = 0)
    return np.dstack((r, g, b))

def get_colored_img(raw_img):
    n_rows = np.shape(raw_img)[0]
    n_cols = np.shape(raw_img)[1]
    masks = get_bayer_masks(n_rows, n_cols)
    r = [[raw_img[i][j] * masks[i][j][0] for j in range(n_cols)] for i in range(n_rows)]
    g = [[raw_img[i][j] * masks[i][j][1] for j in range(n_cols)] for i in range(n_rows)]
    b = [[raw_img[i][j] * masks[i][j][2] for j in range(n_cols)] for i in range(n_rows)]
    return np.dstack((r, g, b))

def bilinear_interpolation(colored_img):
    n_rows = np.shape(colored_img)[0]
    n_cols = np.shape(colored_img)[1]
    masks = get_bayer_masks(n_rows, n_cols)
    colored_img = np.array(np.multiply(colored_img, masks), dtype='uint32')
    kernel_green = [[0, 1, 0], [1, 4, 1], [0, 1, 0]]
    kernel_br = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    r = convolve2d(colored_img[:, :, 0], kernel_br, mode='same') // 4
    g = convolve2d(colored_img[:, :, 1], kernel_green, mode = 'same') // 4
    b = convolve2d(colored_img[:, :, 2], kernel_br, mode = 'same') // 4
    return np.dstack((r, g, b))

def improved_interpolation(raw_img):
    n_rows = np.shape(raw_img)[0]
    n_cols = np.shape(raw_img)[1]
    masks = get_bayer_masks(n_rows, n_cols)
    mask_red =masks[:, :, 0]
    mask_green_odd = masks[:, :, 1]
    mask_green_even = np.copy(masks[:, :, 1])
    mask_blue = masks[:, :, 2]
    imgred = np.multiply(raw_img, mask_red).astype('int64')
    imgblue = np.multiply(raw_img, mask_blue).astype('int64')
    imggreen = np.multiply(raw_img, mask_green_even).astype('int64')
    mask_green_even[::2] = [False] * n_cols
    mask_green_odd[1::2] = [False] * n_cols

    kernel_grb = np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], dtype='int64')
    kernel_rb = np.array([[0, 0, -3, 0, 0], [0, 4, 0, 4, 0], [-3, 0, 12, 0, -3], [0, 4, 0, 4, 0], [0, 0, -3, 0, 0]], dtype='int64')
    kernel_rows = np.array([[0, 0, 1, 0, 0], [0, -2, 0, -2, 0], [-2, 8, 10, 8, -2], [0, -2, 0, -2, 0], [0, 0, 1, 0, 0]], dtype='int64')
    kernel_cols = np.array([[0, 0, -2, 0, 0], [0, -2, 8,-2, 0], [1, 0, 10, 0, 1], [0, -2, 8, -2, 0], [0, 0, -2, 0, 0]], dtype='int64')

    g = (imggreen + np.multiply(mask_red + mask_blue, convolve2d(imgred + imgblue + imggreen, kernel_grb, mode='same')) // 8)
    r = (imgred + np.multiply(mask_blue, convolve2d(imgred + imgblue, kernel_rb, mode='same')) // 16 + np.multiply(mask_green_odd, convolve2d(imgred + imggreen, kernel_rows, mode='same')) // 16 + np.multiply(mask_green_even, convolve2d(imgred + imggreen, kernel_cols, mode='same')) // 16)
    b = (imgblue + np.multiply(mask_red, convolve2d(imgred + imgblue, kernel_rb, mode='same')) // 16 + np.multiply(mask_green_even, convolve2d(imgblue + imggreen, kernel_rows, mode='same')) // 16 + np.multiply(mask_green_odd, convolve2d(imgblue + imggreen, kernel_cols, mode='same')) // 16)
    r[r < 0] = 0
    g[g < 0] = 0
    b[b < 0] = 0
    r[r > 255] = 255
    g[g > 255] = 255
    b[b > 255] = 255
    return np.dstack((r.astype('uint8'), g.astype('uint8'), b.astype('uint8')))

def compute_psnr(img_pred, img_gt):
    img_pred = img_pred.astype('float64')
    img_gt = img_gt.astype('float64')
    n_rows = np.shape(img_gt)[0]
    n_cols = np.shape(img_gt)[1]
    n_chs = np.shape(img_gt)[2]
    mse = np.sum((img_pred - img_gt)**2) / (n_rows * n_cols * n_chs)
    if mse == 0:
        raise ValueError
    return 10 * np.log10((img_gt**2).max() / mse)
