import scipy as sp
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt
from skimage.feature import corner_harris, peak_local_max


def compute_H(im1_pts: np.ndarray, im2_pts: np.ndarray) -> np.ndarray:
    """
    This function computes the homography matrix H that maps points from
    im1_pts to im2_pts. 

    Parameters:
    - im1_pts: A numpy array of shape (N, 2) containing N points from the first image.
    - im2_pts: A numpy array of shape (N, 2) containing N points from the second image.

    Returns:
    - H: A 3x3 numpy array representing the homography matrix.
    """

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
    """
    This function warps the input image im using the homography matrix H.

    Parameters:
    - im: The input image to be warped, given as a numpy array.
    - H: The homography matrix used to warp the image.

    Returns:
    - warped_im: The warped image.
    - min_x: The minimum x-coordinate of the warped image bounding box.
    - min_y: The minimum y-coordinate of the warped image bounding box.
    """

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
    """
    This function warps the input points using the homography matrix H.

    Parameters:
    - points: The input points to be warped, given as an Nx2 numpy array.
    - H: The homography matrix used to warp the points.
    - min_x: The minimum x-coordinate of the warped image bounding box.
    - min_y: The minimum y-coordinate of the warped image bounding box.

    Returns:
    - warped_points: The warped points, adjusted by the minimum x and y coordinates.
    """

    num_points = points.shape[0]
    homogeneous_points = np.hstack([points, np.ones((num_points, 1))])

    warped_points = H @ homogeneous_points.T
    warped_points /= warped_points[2, :]

    warped_points[0, :] -= min_x
    warped_points[1, :] -= min_y

    return warped_points[:2, :].T


def create_blending_mask(warped_left: np.ndarray, image_right: np.ndarray) -> np.ndarray:
    """
    This function creates a blending mask for the two images to be blended.

    Parameters:
    - warped_left: The warped left image.
    - image_right: The right image to be blended with the warped left image.

    Returns:
    - alpha_mask: The alpha mask used for blending the two images.
    """

    mask_left = np.zeros(warped_left.shape[:2], dtype=np.uint8)
    mask_right = np.zeros(image_right.shape[:2], dtype=np.uint8)

    mask_left[np.sum(warped_left, axis=2) > 0] = 1
    mask_right[np.sum(image_right, axis=2) > 0] = 1

    # Compute the distance transform
    dtrans_left = distance_transform_edt(mask_left)
    dtrans_right = distance_transform_edt(mask_right)

    # Create an alpha mask
    # Avoid division by zero
    alpha_mask = dtrans_left / (dtrans_left + dtrans_right + 1e-10)
    # Ensure the values are between 0 and 1
    alpha_mask = np.clip(alpha_mask, 0, 1)

    return alpha_mask


def blend_images(warped_left: np.ndarray, image_right: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
    """
    This function blends the two images using the alpha mask.

    Parameters:
    - warped_left: The warped left image.
    - image_right: The right image to be blended with the warped left image.
    - alpha_mask: The alpha mask used for blending the two images.

    Returns:
    - blended_image: The resulting blended image.
    """

    alpha_mask = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)

    blended_image = warped_left * alpha_mask + \
        image_right * (1 - alpha_mask)
    blended_image = blended_image.astype(np.uint8)

    return blended_image


def apply_affine_transform(image: np.ndarray, transformation_matrix: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    """
    This function applies an affine transformation to an input image.

    Parameters:
    - image: The input image to be transformed.
    - transformation_matrix: The 2x3 affine transformation matrix.
    - output_shape: The shape of the output image.

    Returns:
    - output_image: The transformed image.
    """

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
    """
    This function aligns and blends two images using the provided points.

    Parameters:
    - im1: The first input image.
    - im2: The second input image.
    - warped_points: The points from the first image to be warped.
    - target_points: The target points to which the warped points should be aligned.
    - visualise_mask: A boolean flag to visualize the alpha mask.

    Returns:
    - blended_image: The resulting blended image.
    """

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


def get_harris_corners(im: np.ndarray, edge_discard: int = 20) -> Tuple[np.ndarray, np.ndarray]:
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


def dist2(x: np.ndarray, c: np.ndarray) -> np.ndarray:
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


def adaptive_non_maximal_suppression(corners: np.ndarray, h: np.ndarray, num_points: int = 500, c_robust: float = 0.9) -> np.ndarray:
    """
    This function performs adaptive non-maximal suppression on the Harris corner points.

    Parameters:
    - corners: The corner points to be suppressed.
    - h: The Harris corner strength values.
    - num_points: The number of points to be selected.
    - c_robust: The robustness constant for the adaptive non-maximal suppression.

    Returns:
    - sorted_corners: The sorted corner points after adaptive non-maximal suppression.
    """

    # Extract Harris strengths for each corner point
    scores = h[corners[0], corners[1]]

    # Calculate pairwise distances between all corners using dist2()
    dists = dist2(corners.T, corners.T)  # corners.T is of shape (n, 2)

    # Broadcast comparison: f(x_i) < c_robust * f(x_j)
    larger_mask = scores[:, np.newaxis] < (c_robust * scores[np.newaxis, :])

    # Mask the distances where the comparison holds and set the rest to infinity
    masked_dists = np.where(larger_mask, dists, np.inf)

    # Calculate the minimum radius for each point
    radii = np.min(masked_dists, axis=1)

    # Sort points by their radii in descending order
    sorted_indices = np.argsort(-radii)

    # Sort the original corners based on radii
    sorted_corners = corners[:, sorted_indices]

    return sorted_corners[:, :num_points]


def extract_feature_descriptors(im: np.ndarray, coords: np.ndarray, patch_size: int = 8, window_size: int = 40, spacing: int = 5) -> np.ndarray:
    """
    This function extracts feature descriptors from the input image around the specified coordinates.

    Parameters:
    - im: The input image from which to extract descriptors.
    - coords: The coordinates around which to extract descriptors.
    - patch_size: The size of the patch to be extracted.
    - window_size: The size of the window around the point to be considered.
    - spacing: The spacing between pixels in the window.

    Returns:
    - descriptors: The extracted feature descriptors.
    """

    # Calculate half the window size for boundary checks
    half_window = window_size // 2

    # Check if the point is within the valid bounds of the image
    def is_within_bounds(y: int, x: int) -> bool:
        """
        This function checks if the point (y, x) is within the valid bounds of the image.

        Parameters:
        - y: The y-coordinate of the point.
        - x: The x-coordinate of the point.

        Returns:
        - A boolean indicating whether the point is within the bounds of the image.
        """

        return (half_window <= y < im.shape[0] - half_window and
                half_window <= x < im.shape[1] - half_window)

    # Extract and normalize an 8x8 patch from a 40x40 window around the point (y, x) for a given channel
    def extract_patch(y: int, x: int, channel: int) -> np.ndarray:
        """
        This function extracts and normalizes an 8x8 patch from a 40x40 window around the 
        point (y, x) for a given channel.

        Parameters:
        - y: The y-coordinate of the point.
        - x: The x-coordinate of the point.
        - channel: The channel index for the image.

        Returns:
        - patch: The extracted and normalized 8x8 patch.
        """

        # Extract the 40x40 window for the current channel
        window = im[y-half_window:y + half_window,
                    x - half_window:x + half_window, channel]

        # Apply Gaussian blur to reduce aliasing
        window = gaussian_filter(window, sigma=1)

        # Sample an 8x8 patch from the window with specified spacing
        patch = window[::spacing, ::spacing][:patch_size, :patch_size]

        # Normalize the patch (bias/gain normalization)
        patch_mean = np.mean(patch)
        patch_std = np.std(patch)

        return (patch - patch_mean) / patch_std if patch_std > 0 else patch

    # Generate descriptors for each point in coords
    descriptors = [
        # Concatenate descriptors from all three channels to form a single vector
        np.concatenate([extract_patch(y, x, channel).flatten()
                       for channel in range(3)])
        for y, x in coords.T if is_within_bounds(y, x)
    ]

    return np.array(descriptors)


def visualize_descriptors(descriptors: np.ndarray, num_descriptors: int = 5, patch_size: int = 8) -> None:
    """
    This function visualizes the extracted feature descriptors.

    Parameters:
    - descriptors: The extracted feature descriptors.
    - num_descriptors: The number of descriptors to visualize.
    - patch_size: The size of the patch to be visualized.

    Returns:
    - None
    """

    fig, axes = plt.subplots(1, num_descriptors, figsize=(15, 5))

    for i, ax in enumerate(axes[:num_descriptors]):
        descriptor = descriptors[i]

        # Reshape the descriptor into three 8x8 patches for R, G, B channels
        patches = [descriptor[j*patch_size*patch_size:(j+1)*patch_size*patch_size].reshape(
            (patch_size, patch_size)) for j in range(3)]

        # Combine the patches into a single 8x8x3 RGB image
        patch_rgb = np.stack(patches, axis=-1)

        # Normalize the RGB image to [0, 1] for visualization
        patch_rgb = (patch_rgb - patch_rgb.min()) / \
            (patch_rgb.max() - patch_rgb.min())

        # Display the RGB image
        ax.imshow(patch_rgb)
        ax.axis('off')
        ax.set_title(f'Descriptor {i+1}')

    plt.show()


def match_features(descriptors1: np.ndarray, descriptors2: np.ndarray, ratio_threshold: float = 0.8) -> List[Tuple[int, int]]:
    """
    This function matches feature descriptors between two images using the ratio test.

    Parameters:
    - descriptors1: The feature descriptors from the first image.
    - descriptors2: The feature descriptors from the second image.
    - ratio_threshold: The threshold for the Lowe's ratio test.

    Returns:
    - matches: A list of matched feature indices between the two images.
    """

    matches = []
    matched_in_second_image = set()  # Track points already matched in the second image

    for i, desc1 in enumerate(descriptors1):
        # Calculate distances to all descriptors in the second image
        distances = np.linalg.norm(descriptors2 - desc1, axis=1)

        # Get the two nearest neighbors
        nearest_neighbor_idx, second_nearest_neighbor_idx = np.argsort(distances)[
            :2]

        # Apply Lowe's ratio test
        if distances[nearest_neighbor_idx] < ratio_threshold * distances[second_nearest_neighbor_idx]:
            # Ensure one-to-one matching
            if nearest_neighbor_idx not in matched_in_second_image:
                matches.append((i, nearest_neighbor_idx))
                matched_in_second_image.add(nearest_neighbor_idx)

    return matches


def warp_points_ransac(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    This function warps the input points using the homography matrix H.

    Parameters:
    - points: The input points to be warped, given as an Nx2 numpy array.
    - H: The homography matrix used to warp the points.

    Returns:
    - warped_points: The warped points.
    """

    num_points = points.shape[0]
    homogeneous_points = np.hstack([points, np.ones((num_points, 1))])
    warped_points = H @ homogeneous_points.T
    warped_points /= warped_points[2, :]

    return warped_points[:2, :].T


def ransac(coords_1: np.ndarray, coords_2: np.ndarray, matches: list, num_iterations: int = 1000, threshold: int = 1) -> Tuple[np.ndarray, List[int], np.ndarray, np.ndarray]:
    """
    This function performs RANSAC to estimate the homography matrix between two sets of points.

    Parameters:
    - coords_1: The coordinates of the first set of points.
    - coords_2: The coordinates of the second set of points.
    - matches: The matched indices between the two sets of points.
    - num_iterations: The number of RANSAC iterations.
    - threshold: The distance threshold for inliers.

    Returns:
    - final_H: The final homography matrix.
    - best_inliers: The indices of the best set of inliers.
    - final_pts1: The final set of points from the first image.
    - final_pts2: The final set of points from the second image.
    """

    pts1 = []
    pts2 = []

    for idx, (i, j) in enumerate(matches):
        y1, x1 = coords_1[:, i]
        y2, x2 = coords_2[:, j]
        pts1.append([x1, y1])
        pts2.append([x2, y2])

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    max_inliers = 0
    best_inliers = []

    for _ in range(num_iterations):
        # Randomly select 4 points
        random_indices = np.random.choice(len(matches), 4, replace=False)

        selected_pts1 = pts1[random_indices]
        selected_pts2 = pts2[random_indices]

        # Compute the homography using the selected points
        H_temp = compute_H(selected_pts1, selected_pts2)

        inliers = []
        for idx in range(len(pts1)):
            pt1 = pts1[idx]
            pt2 = pts2[idx]

            warped_pt1 = warp_points_ransac(np.array([pt1]), H_temp)[0]
            dist = np.linalg.norm(warped_pt1 - pt2)

            if dist < threshold:
                inliers.append(idx)

        # Update the best set of inliers
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_inliers = inliers

    if max_inliers < 4:
        raise ValueError("Not enough inliers to compute a homography.")

    final_pts1 = pts1[best_inliers]
    final_pts2 = pts2[best_inliers]

    # Compute the final homography using all the inliers
    final_H = compute_H(final_pts1, final_pts2)

    return final_H, best_inliers, final_pts1, final_pts2
