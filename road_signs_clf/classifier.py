import numpy as np
from scipy.signal import convolve
from cv2 import imread

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    if phi < 0:
        phi += np.pi
    return rho, phi

def extract_hog(img, cell_rows=8, cell_cols=8, bin_count=9, block_row_cells=2, block_col_cells=2):
    eps = 1

    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    dx = np.array([[-1, 0, 1]])
    dy = np.array([[-1], [0], [1]])
    
    ix = convolve(img, np.dstack((sx, sx, sx)), mode='same')
    iy = convolve(img, np.dstack((sy, sy, sy)), mode='same')
    mag = np.zeros(np.shape(ix))
    angle = np.zeros(np.shape(ix))
    for r in range(len(ix)):
        for c in range(len(ix[0])):
            for ch in range(3):
                mag[r][c][ch], angle[r][c][ch] = cart2pol(ix[r, c, ch], iy[r, c, ch])

    step = np.pi / bin_count
    hists = None
    for r in range(0, len(angle), cell_rows):
        hists_row = None
        for c in range(0, len(angle[0]), cell_cols):
            hists_ch = np.zeros((3, bin_count))
            for ch in range(3):
                hist = np.zeros(bin_count)
                for row in range(r, min(len(angle), r + cell_rows)):
                    for col in range(c, min(len(angle[0]), c + cell_cols)):
                        magnitude = mag[row][col][ch]
                        ang = angle[row][col][ch]
                        hist[(int)(np.round(ang / step)) % 9] += magnitude
                hists_ch[ch] = hist
            if len(np.shape(hists_row)) == 0:
                hists_row = [hists_ch]
            else:
                hists_row = np.append(hists_row, [hists_ch], axis=0)
        if len(np.shape(hists)) == 0:
            hists = [hists_row]
        else:
            hists = np.append(hists, [hists_row], axis=0)

    block_vectors = np.array([])
    for r in range(len(hists) - block_row_cells + 1):
        for c in range(len(hists[0]) - block_col_cells + 1):
            for ch in range(3):
                vec = np.array([])
                for row in range(r, r + block_row_cells):
                    for col in range(c, c + block_col_cells):
                        vec = np.concatenate((vec, hists[row, col, ch]), axis=None)
                vec = vec / np.sqrt(np.sum(vec ** 2) + eps)
                block_vectors = np.concatenate((block_vectors, vec), axis=None)
    return block_vectors

img = imread('tests/00_test_img_input/train/39208.png')
print(extract_hog(img))