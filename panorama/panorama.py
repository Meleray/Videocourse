import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.transform import ProjectiveTransform
from skimage.transform import warp
from skimage.filters import gaussian
from numpy.linalg import inv

DEFAULT_TRANSFORM = ProjectiveTransform


def find_orb(img, keypoints=500):
    """Find keypoints and their descriptors in image.

    img ((W, H, 3)  np.ndarray) : 3-channel image
    n_keypoints (int) : number of keypoints to find

    Returns:
        (N, 2)  np.ndarray : keypoints
        (N, 256)  np.ndarray, type=np.bool  : descriptors
    """
    extractor = ORB(n_keypoints=keypoints)
    grey_img = rgb2gray(img)
    extractor.detect_and_extract(grey_img)
    return extractor.keypoints, extractor.descriptors


def center_and_normalize_points(points):
    """Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).

    points ((N, 2) np.ndarray) : the coordinates of the image points

    Returns:
        (3, 3) np.ndarray : the transformation matrix to obtain the new points
        (N, 2) np.ndarray : the transformed image points
    """
    pointsh = np.row_stack([np.matrix.transpose(np.array(points)), np.ones((np.shape(points)[0]), )])
    matrix = np.zeros((3, 3))
    center = np.sum(points, axis=0).astype('float') / len(points)
    N = 0
    for pnt in points:
        N += np.sqrt(np.sum((pnt - center) ** 2))
    N = np.sqrt(2) / N
    matrix[2][2] = 1
    matrix[0][0] = N
    matrix[1][1] = N
    matrix[0][2] = -N * center[0]
    matrix[1][2] = -N * center[1]
    pointsh = np.matrix.transpose(np.dot(matrix, pointsh))[:, :2:]
    return matrix, pointsh


def find_homography(src_keypoints, dest_keypoints):
    """Estimate homography matrix from two sets of N (4+) corresponding points.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates

    Returns:
        ((3, 3) np.ndarray) : homography matrix
    """

    src_matrix, src = center_and_normalize_points(src_keypoints)
    dest_matrix, dest = center_and_normalize_points(dest_keypoints)

    H = np.zeros((3, 3))
    A = []
    for i in range(len(src)):
        x1 = src[i, 0]
        y1 = src[i, 1]
        x2 = dest[i, 0]
        y2 = dest[i, 1]
        ax = [-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2]
        ay = [0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2]
        A.append(ax)
        A.append(ay)
    U, S, V = np.linalg.svd(A)
    h = V[-1]
    H[0] = h[:3:]
    H[1] = h[3:6:]
    H[2] = h[6::]
    return np.dot(np.linalg.inv(dest_matrix), np.dot(H, src_matrix))


def ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, max_trials=2000, residual_threshold=1.0, return_matches=False):
    """Match keypoints of 2 images and find ProjectiveTransform using RANSAC algorithm.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    src_descriptors ((N, 256) np.ndarray) : source descriptors
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates
    dest_descriptors ((N, 256) np.ndarray) : destination descriptors
    max_trials (int) : maximum number of iterations for random sample selection.
    residual_threshold (float) : maximum distance for a data point to be classified as an inlier.
    return_matches (bool) : if True function returns matches

    Returns:
        skimage.transform.ProjectiveTransform : transform of source image to destination image
        (Optional)(N, 2) np.ndarray : inliers' indexes of source and destination images
    """

    matches = match_descriptors(src_descriptors, dest_descriptors)
    m_count = 0
    final_ids = []
    rng = np.random.default_rng()
    for k in range(max_trials):
        ids = rng.choice(len(matches), size=4)
        src = src_keypoints[matches[ids, 0]]
        dst = dest_keypoints[matches[ids, 1]]
        transform = ProjectiveTransform(find_homography(src, dst))
        new_dst = transform(src_keypoints[matches[:, 0]])
        counter = 0
        ids = []
        for i in range(len(matches)):
            if np.linalg.norm(new_dst[i] - dest_keypoints[matches[i, 1]]) < residual_threshold:
                counter += 1
                ids.append(i)
        if m_count < counter:
            m_count = counter
            final_ids = ids.copy()
    if return_matches:
        return ProjectiveTransform(find_homography(src_keypoints[matches[final_ids, 0]], dest_keypoints[matches[final_ids, 1]])), matches[final_ids]
    else:
        return ProjectiveTransform(find_homography(src_keypoints[matches[final_ids, 0]], dest_keypoints[matches[final_ids, 1]]))


def find_simple_center_warps(forward_transforms):
    """Find transformations that transform each image to plane of the central image.

    forward_transforms (Tuple[N]) : - pairwise transformations

    Returns:
        Tuple[N + 1] : transformations to the plane of central image
    """
    image_count = len(forward_transforms) + 1
    center_index = (image_count - 1) // 2

    result = [None] * image_count
    result[center_index] = DEFAULT_TRANSFORM()
    for i in range(center_index + 1, len(result)):
        result[i] = ProjectiveTransform(np.linalg.inv(forward_transforms[i - 1].params)) + result[i - 1]
    for i in range(center_index - 1, -1, -1):
        result[i] = forward_transforms[i] + result[i + 1]
    return tuple(result)


def get_corners(image_collection, center_warps):
    """Get corners' coordinates after transformation."""
    for img, transform in zip(image_collection, center_warps):
        height, width, _ = img.shape
        corners = np.array([[0, 0],
                            [height, 0],
                            [height, width],
                            [0, width]])

        yield transform(corners)[:, ::-1]


def get_min_max_coords(corners):
    """Get minimum and maximum coordinates of corners."""
    corners = np.concatenate(corners)
    return corners.min(axis=0), corners.max(axis=0)


def get_final_center_warps(image_collection, simple_center_warps):
    """Find final transformations.

        image_collection (Tuple[N]) : list of all images
        simple_center_warps (Tuple[N])  : transformations unadjusted for shift

        Returns:
            Tuple[N] : final transformations
        """
    corners = [corner for corner in get_corners(image_collection, simple_center_warps)]
    mi, ma = get_min_max_coords(corners)
    shape = ma - mi
    mi[0] = min(mi[0], 0)
    mi[1] = min(mi[1], 0)
    matrix = np.array([[1, 0, -mi[1]],
            [0, 1, -mi[0]],
            [0, 0, 1]])
    return tuple([tr + ProjectiveTransform(matrix) for tr in simple_center_warps]), np.ceil(shape[::-1]).astype('int')


def rotate_transform_matrix(transform):
    """Rotate matrix so it can be applied to row:col coordinates."""
    matrix = transform.params[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)


def warp_image(image, transform, shape):
    """Apply transformation to an image and its mask

    image ((W, H, 3)  np.ndarray) : image for transformation
    transform (skimage.transform.ProjectiveTransform): transformation to apply
    output_shape (int, int) : shape of the final pano

    Returns:
        (W, H, 3)  np.ndarray : warped image
        (W, H)  np.ndarray : warped mask
    """
    mask = warp(np.ones(np.shape(image)[:2:]).astype('bool'), rotate_transform_matrix(ProjectiveTransform(np.linalg.inv(transform.params))), output_shape=shape)
    chs = [warp(image[:, :, i], rotate_transform_matrix(ProjectiveTransform(np.linalg.inv(transform.params))), output_shape=shape) for i in range(3)]
    return np.dstack((chs[0], chs[1], chs[2])), mask


def merge_pano(image_collection, final_center_warps, output_shape):
    """ Merge the whole panorama

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano

    Returns:
        (output_shape) np.ndarray: final pano
    """
    result = np.zeros((output_shape[0], output_shape[1], 3))
    result_mask = np.zeros(output_shape, dtype=np.bool8)
    for i in range(len(final_center_warps)):
        new_image, new_mask = warp_image(image_collection[i], final_center_warps[i], output_shape)
        new_mask = np.logical_and(new_mask, np.logical_not(result_mask))
        new_image = np.multiply(new_image, np.dstack((new_mask, new_mask, new_mask)))
        result_mask = np.logical_or(result_mask, new_mask)
        result = np.add(result, new_image)
    return result


def get_gaussian_pyramid(image, n_layers, sigma):
    """Get Gaussian pyramid.

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Gaussian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Gaussian pyramid

    """
    # your code here
    pass


def get_laplacian_pyramid(image, n_layers, sigma):
    """Get Laplacian pyramid

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Laplacian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Laplacian pyramid
    """
    # your code here
    pass


def merge_laplacian_pyramid(laplacian_pyramid):
    """Recreate original image from Laplacian pyramid

    laplacian pyramid: tuple of np.array (h, w, 3)

    Returns:
        np.array (h, w, 3)
    """
    return sum(laplacian_pyramid)


def increase_contrast(image_collection):
    """Increase contrast of the images in collection"""
    result = []

    for img in image_collection:
        img = img.copy()
        for i in range(img.shape[-1]):
            img[:, :, i] -= img[:, :, i].min()
            img[:, :, i] /= img[:, :, i].max()
        result.append(img)

    return result


def gaussian_merge_pano(image_collection, final_center_warps, output_shape, n_layers, image_sigma, merge_sigma):
    """ Merge the whole panorama using Laplacian pyramid

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano
    n_layers (int) : number of layers in Laplacian pyramid
    image_sigma (int) :  sigma for Gaussian filter for images
    merge_sigma (int) : sigma for Gaussian filter for masks

    Returns:
        (output_shape) np.ndarray: final pano
    """
    # your code here
    pass

