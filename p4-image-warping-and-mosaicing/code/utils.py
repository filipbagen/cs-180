import scipy as sp
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.feature import corner_harris, peak_local_max


def compute_H(im1_pts: np.ndarray, im2_pts: np.ndarray) -> np.ndarray:
    num_pts: int = im1_pts.shape[0]

    A: list = []

    for i in range(num_pts):
        x1, y1 = im1_pts[i][0], im1_pts[i][1]
        x2, y2 = im2_pts[i][0], im2_pts[i][1]

        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])  # x'
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])  # y'

    A = np.array(A)
    _, _, V = np.linalg.svd(A)  # Perform SVD
    H = V[-1].reshape(3, 3)

    return H / H[-1, -1]  # Normalize so H[2,2] is 1


def warp_image(im: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, int, int]:
    height, width = im.shape[:2]

    # Define corners of the original image in homogeneous coordinates
    corners = np.array([
        [0, 0, 1], [width - 1, 0, 1],
        [width - 1, height - 1, 1], [0, height - 1, 1]
    ]).T

    # Warp the corners using the homography matrix
    warped_corners = H @ corners
    warped_corners /= warped_corners[2, :]  # Normalize homogeneous coordinates

    # Calculate the bounding box for the warped image
    min_x = int(np.floor(min(warped_corners[0, :])))
    min_y = int(np.floor(min(warped_corners[1, :])))
    max_x = int(np.ceil(max(warped_corners[0, :])))
    max_y = int(np.ceil(max(warped_corners[1, :])))

    # Create a meshgrid for the output coordinates
    X, Y = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
    warped_coords = np.vstack((X.ravel(), Y.ravel(), np.ones(X.size)))

    # Compute inverse homography to map to input image coordinates
    H_inv = np.linalg.inv(H)
    input_coords = H_inv @ warped_coords
    input_coords /= input_coords[2, :]  # Normalize homogeneous coordinates

    input_X = input_coords[0, :].reshape(max_y - min_y, max_x - min_x)
    input_Y = input_coords[1, :].reshape(max_y - min_y, max_x - min_x)

    # Interpolate pixel values for each channel
    warped_im = np.zeros((max_y - min_y, max_x - min_x,
                         im.shape[2]), dtype=im.dtype)

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    for channel in range(im.shape[2]):
        # interpolate using griddata
        values = im[..., channel].ravel()
        points = np.vstack((grid_x.ravel(), grid_y.ravel())).T

        warped_im[..., channel] = sp.interpolate.griddata(
            points,
            values,
            (input_X, input_Y),
            method='linear',
            fill_value=0
        )

    return warped_im, min_x, min_y


def warp_points(points: np.ndarray, H: np.ndarray, min_x: int, min_y: int) -> np.ndarray:
    num_points = points.shape[0]
    homogeneous_points = np.hstack([points, np.ones((num_points, 1))])

    warped_points = H @ homogeneous_points.T
    warped_points /= warped_points[2, :]

    warped_points[0, :] -= min_x
    warped_points[1, :] -= min_y

    return warped_points[:2, :].T


def create_blending_mask(warped_left: np.ndarray, panorama_right: np.ndarray) -> np.ndarray:
    mask_left = np.zeros(warped_left.shape[:2], dtype=np.uint8)
    mask_right = np.zeros(panorama_right.shape[:2], dtype=np.uint8)

    mask_left[np.sum(warped_left, axis=2) > 0] = 1
    mask_right[np.sum(panorama_right, axis=2) > 0] = 1

    # Compute the distance transform
    dtrans_left = distance_transform_edt(mask_left)
    dtrans_right = distance_transform_edt(mask_right)

    # Create an alpha mask
    # Avoid division by zero
    alpha_mask = dtrans_left / (dtrans_left + dtrans_right + 1e-10)
    # Ensure the values are between 0 and 1
    alpha_mask = np.clip(alpha_mask, 0, 1)

    return alpha_mask


def blend_images(warped_left: np.ndarray, panorama_right: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
    alpha_mask = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)

    blended_image = warped_left * alpha_mask + \
        panorama_right * (1 - alpha_mask)
    blended_image = blended_image.astype(np.uint8)

    return blended_image


def apply_affine_transform(image: np.ndarray, transformation_matrix: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    output_image = np.zeros(
        (output_shape[0], output_shape[1], image.shape[2]), dtype=image.dtype)
    inverse_transform = np.linalg.inv(
        np.vstack([transformation_matrix, [0, 0, 1]]))[:2, :]

    # Iterate through each pixel in the output image
    for y in range(output_shape[0]):
        for x in range(output_shape[1]):
            source_coords = np.dot(inverse_transform, [x, y, 1])
            src_x, src_y = source_coords[0], source_coords[1]

            # Bilinear interpolation
            if 0 <= src_x < image.shape[1] and 0 <= src_y < image.shape[0]:
                x0, y0 = int(src_x), int(src_y)
                dx, dy = src_x - x0, src_y - y0

                # Ensure the coordinates are within bounds
                x1 = min(x0 + 1, image.shape[1] - 1)
                y1 = min(y0 + 1, image.shape[0] - 1)

                # Get the four neighboring pixel values
                pixel_00 = image[y0, x0]
                pixel_01 = image[y0, x1]
                pixel_10 = image[y1, x0]
                pixel_11 = image[y1, x1]

                # Perform bilinear interpolation
                output_image[y, x] = (
                    pixel_00 * (1 - dx) * (1 - dy) +
                    pixel_01 * dx * (1 - dy) +
                    pixel_10 * (1 - dx) * dy +
                    pixel_11 * dx * dy
                )

    return output_image


def align_and_blend_images(im1: np.ndarray, im2: np.ndarray, warped_points: np.ndarray, target_points: np.ndarray, visualise_mask: bool = False) -> np.ndarray:
    # Compute the rotation angle
    angle_warped = np.arctan2(warped_points[1, 1] - warped_points[0, 1],
                              warped_points[1, 0] - warped_points[0, 0])
    angle_target = np.arctan2(target_points[1, 1] - target_points[0, 1],
                              target_points[1, 0] - target_points[0, 0])

    rotation_angle = angle_target - angle_warped

    # Compute the translation
    translation_x = target_points[0, 0] - (
        np.cos(rotation_angle) * warped_points[0, 0] -
        np.sin(rotation_angle) * warped_points[0, 1]
    )
    translation_y = target_points[0, 1] - (
        np.sin(rotation_angle) * warped_points[0, 0] +
        np.cos(rotation_angle) * warped_points[0, 1]
    )

    # Build the affine transformation matrix
    transformation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle), translation_x],
        [np.sin(rotation_angle),  np.cos(rotation_angle), translation_y]
    ])

    height, width = im2.shape[:2]
    corners = np.array([
        [0, 0],
        [im1.shape[1], 0],
        [im1.shape[1], im1.shape[0]],
        [0, im1.shape[0]]
    ])
    transformed_corners = np.dot(
        np.c_[corners, np.ones(4)], transformation_matrix.T)[:, :2]

    min_x_transformed = min(transformed_corners[:, 0])
    max_x_transformed = max(transformed_corners[:, 0])
    min_y_transformed = min(transformed_corners[:, 1])
    max_y_transformed = max(transformed_corners[:, 1])

    # Size of the canvas
    canvas_width = int(max(max_x_transformed, width) -
                       min(min_x_transformed, 0))
    canvas_height = int(max(max_y_transformed, height) -
                        min(min_y_transformed, 0))

    # Overlay the aligned image over im2
    result_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    offset_x = int(-min(min_x_transformed, 0))
    offset_y = int(-min(min_y_transformed, 0))
    result_image[offset_y:offset_y + height,
                 offset_x:offset_x + width] = im2  # Add the base image

    translation_x_centered = translation_x + offset_x
    translation_y_centered = translation_y + offset_y

    # Update the transformation matrix with the centered translation
    transformation_matrix_centered = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle),
         translation_x_centered],
        [np.sin(rotation_angle),  np.cos(
            rotation_angle), translation_y_centered]
    ])

    # Affine Transformation
    aligned_image_centered = apply_affine_transform(
        im1, transformation_matrix_centered, (canvas_height, canvas_width))

    # Create the blending mask using the aligned images
    alpha_mask = create_blending_mask(aligned_image_centered, result_image)

    if visualise_mask:
        # Visualize the alpha mask
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(alpha_mask, cmap='gray')
        ax.set_title('Alpha Mask')
        ax.axis('off')
        plt.show()

    # Blend the images using the generated alpha mask
    blended_image = blend_images(
        aligned_image_centered, result_image, alpha_mask)

    return blended_image


def get_harris_corners(im, edge_discard=20):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """

    assert edge_discard >= 20

    # find harris corners
    h = corner_harris(im, method='eps', sigma=1)
    coords = peak_local_max(h, min_distance=1)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T
    return h, coords


def dist2(x, c):
    """
    dist2 Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them. Both matrices must be of
    the same column dimension. If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns. The
    I, Jth entry is the squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """

    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, 'Data dimension does not match dimension of centers'

    return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
        np.ones((ndata, 1)) * np.sum((c**2).T, axis=0) - \
        2 * np.inner(x, c)
