import numpy as np

def mse(chf, chs, diffx, diffy):
    n_rows = np.shape(chf)[0]
    n_cols = np.shape(chf)[1]
    img1 = chf[max(0,diffx):min(n_rows, n_rows+diffx):, max(0,diffy):min(n_cols, n_cols+diffy):]
    img2 = chs[max(0,-diffx):min(n_rows, n_rows-diffx):, max(0,-diffy):min(n_cols, n_cols-diffy):]
    return np.sum(((img1 - img2) ** 2)) / (np.shape(img1)[0] * np.shape(img1)[1])

def psnr(chf, chs, diffx, diffy):
    n_rows = np.shape(chf)[0]
    n_cols = np.shape(chf)[1]
    img1 = chf[max(0,diffx):min(n_rows, n_rows+diffx):, max(0,diffy):min(n_cols, n_cols+diffy):]
    img2 = chs[max(0,-diffx):min(n_rows, n_rows-diffx):, max(0,-diffy):min(n_cols, n_cols-diffy):]
    return 10 * np.log10(np.max(img1 ** 2) / (np.sum(((img1 - img2) ** 2)) / (np.shape(img1)[0] * np.shape(img1)[1])))

def norm(chf, chs, diffx, diffy):            
    n_rows = np.shape(chf)[0]
    n_cols = np.shape(chf)[1]
    img1 = chf[max(0,diffx):min(n_rows, n_rows+diffx):, max(0,diffy):min(n_cols, n_cols+diffy):]
    img2 = chs[max(0,-diffx):min(n_rows, n_rows-diffx):, max(0,-diffy):min(n_cols, n_cols-diffy):]
    return np.sum(img1 * img2) / np.sqrt(np.sum(img1 ** 2) * np.sum(img2 ** 2))

def pyramid(img):
    if np.max(np.shape(img)) <= 500:
        diffx_red = 0
        diffy_red = 0
        diffx_blue = 0
        diffy_blue = 0
        err_red = 0.0
        err_blue = 0.0
        for diffx in range(-15,16):
            for diffy in range(-15,16): 
                new_err_blue = norm(img[1], img[0], diffx, diffy)
                new_err_red = norm(img[1], img[2], diffx, diffy)
                if err_blue < new_err_blue:
                    err_blue = new_err_blue
                    diffx_blue = diffx
                    diffy_blue = diffy
                if err_red < new_err_red:
                    err_red = new_err_red
                    diffx_red = diffx
                    diffy_red = diffy
        return diffx_red, diffy_red, diffx_blue, diffy_blue
    else:
        diffx_red, diffy_red, diffx_blue, diffy_blue = pyramid(img[:, ::2, ::2])
        diffx_red *= 2
        diffy_red *= 2
        diffx_blue *= 2
        diffy_blue *= 2
        diffx_red_ch = 0
        diffy_red_ch = 0
        diffx_blue_ch = 0
        diffy_blue_ch = 0
        err_red = 0.0
        err_blue = 0.0
        for diffx in range(-1, 2):
            for diffy in range(-1,2): 
                new_err_blue = norm(img[1], img[0], diffx_blue + diffx, diffy_blue + diffy)
                new_err_red = norm(img[1], img[2], diffx_red + diffx, diffy_red + diffy)
                if err_blue < new_err_blue:
                    err_blue = new_err_blue
                    diffx_blue_ch = diffx
                    diffy_blue_ch = diffy
                if err_red < new_err_red:
                    err_red = new_err_red
                    diffx_red_ch = diffx
                    diffy_red_ch = diffy
        diffx_red += diffx_red_ch
        diffy_red += diffy_red_ch
        diffx_blue += diffx_blue_ch
        diffy_blue += diffy_blue_ch
        return diffx_red, diffy_red, diffx_blue, diffy_blue

def align(raw_img, g_coord):
    len_vert = np.shape(raw_img)[0] // 3
    len_horiz = np.shape(raw_img)[1]
    edge_vert = (int)(len_vert * 0.05)
    edge_horiz = (int)(len_horiz * 0.05)

    ch_blue = np.array(raw_img[edge_vert:(len_vert - edge_vert):, edge_horiz:(len_horiz - edge_horiz):])
    ch_green = np.array(raw_img[(len_vert+edge_vert):(2 * len_vert - edge_vert):, edge_horiz:(len_horiz - edge_horiz):])
    ch_red = np.array(raw_img[(2*len_vert+edge_vert):(3*len_vert - edge_vert):, edge_horiz:(len_horiz - edge_horiz):])

    diffx_red, diffy_red, diffx_blue, diffy_blue = pyramid(np.array([ch_blue, ch_green, ch_red]))

    n_rows = np.shape(ch_red)[0]
    n_cols = np.shape(ch_red)[1]
    g = ch_green[max(0,diffx_blue, diffx_red):min(n_rows, n_rows+diffx_blue, n_rows+diffx_red):, max(0,diffy_blue, diffy_red):min(n_cols, n_cols+diffy_blue, n_cols+diffy_red):]
    r = ch_red[max(0,diffx_blue - diffx_red, -diffx_red):min(n_rows, n_rows+(diffx_blue - diffx_red), n_rows-diffx_red):, max(0,diffy_blue - diffy_red, -diffy_red):min(n_cols, n_cols+(diffy_blue - diffy_red), n_cols-diffy_red):]
    b = ch_blue[max(0,diffx_red - diffx_blue, -diffx_blue):min(n_rows, n_rows+(diffx_red - diffx_blue), n_rows-diffx_blue):, max(0,diffy_red - diffy_blue, -diffy_blue):min(n_cols, n_cols+(diffy_red - diffy_blue), n_cols-diffy_blue):]

    b_pnt = (g_coord[0] - diffx_blue - len_vert, g_coord[1] - diffy_blue)
    r_pnt = (g_coord[0] - diffx_red + len_vert, g_coord[1] - diffy_red)
    return np.dstack((r, g, b)), b_pnt, r_pnt