from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Chessboard size constants
SQUARE_SIZE = 0.035
NUM_CORNERS_X = 7
NUM_CORNERS_Y = 3

def generate_chessboard_3D_world_coordinates(
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes 3D points of chessboard cross-junctions in the world 
    coordinate frame. The order of the points in the output arrays 
    should be left-to-right, top-to-bottom.

    Returns:
        X_world: (NUM_CORNERS_X * NUM_CORNERS_Y,) array of world 
                 points along the x-axis.
        Y_world: (NUM_CORNERS_X * NUM_CORNERS_Y,) array of world 
                 points along the y-axis.
        Z_world: (NUM_CORNERS_X * NUM_CORNERS_Y,) array of world 
                 points along the z-axis.
    """
    ##### YOUR CODE STARTS HERE #####
    # Hint: Use the SQUARE_SIZE, NUM_CORNERS_X, and NUM_CORNERS_Y
    # Hint: Use np.meshgrid to generate the X, Y coordinates
    x = np.arange(NUM_CORNERS_X) * SQUARE_SIZE
    y = np.arange(NUM_CORNERS_Y - 1, -1, -1) * SQUARE_SIZE
    X_grid, Y_grid = np.meshgrid(x, y)
    
    X_world = X_grid.flatten()
    Y_world = Y_grid.flatten()
    Z_world = np.zeros_like(X_world)
    ###### YOUR CODE ENDS HERE ######

    # Sanity check the order is correct
    if (X_world[0] > X_world[-1] or Y_world[0] < Y_world[-1]):
        raise ValueError("Check that order of the coordinates is correct")
    
    return X_world, Y_world, Z_world

def generate_chessboard_2D_pixel_coordinates(img_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 2D pixel coordinates chessboard cross-junctions. The order 
    of the points in the output arrays should be left-to-right, 
    top-to-bottom.

    Args:
        img_path: path to image.

    Returns:
        u_meas: (NUM_CORNERS_X * NUM_CORNERS_Y,) array of pixel 
                coordinates along the x-axis.
        v_meas: (NUM_CORNERS_X * NUM_CORNERS_Y,) array of pixel 
                coordinates along the y-axis.
    """
    img = cv2.imread(img_path, 0)

    ##### YOUR CODE STARTS HERE #####
    # NOTE: Ensure that points are ordered left-to-right, bottom-to-top
    found, corners = cv2.findChessboardCorners( img, (NUM_CORNERS_X, NUM_CORNERS_Y), None)
    if not found:
        raise ValueError(f"Chessboard corners not found in image {img_path}")
    print("Found:", found)
    print("Corner shape:", corners.shape if found else None)
      # corners shape: (N, 1, 2)
    corners = corners.reshape(-1, 2)
    u_meas = corners[:, 0]
    v_meas = corners[:, 1]
    ###### YOUR CODE ENDS HERE ######
    
    return u_meas, v_meas

def compute_homography(
    u: np.ndarray,
    v: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """
    Computes homography matrix, which is required to transform 3D world 
    coordinates to 2D pixel coordinates.

    Args:
        u: (N,) pixel x-coordinates.
        v: (N,) pixel y-coordinates.
        X: (N,) 3D world x-coordinates.
        Y: (N,) 3D world y-coordinates.
    
    Returns:
        Homography matrix, M, with shape (3, 3).
    """
    ##### YOUR CODE STARTS HERE #####
    # Hint: use np.linalg.svd
    # Hint: Use <array>.reshape(3, 3) to give M the proper dimensions.
    N = len(u)
    P = np.zeros((2 * N, 9))
    
    for i in range(N):
        ui, vi = u[i], v[i]
        Xi, Yi = X[i], Y[i]
        P[2 * i] = np.array([-Xi, -Yi, -1, 0, 0, 0, ui * Xi, ui * Yi, ui])
        P[2 * i + 1] = np.array([0, 0, 0, -Xi, -Yi, -1, vi * Xi, vi * Yi, vi])
    
    _, _, Vh = np.linalg.svd(P)
    H = Vh[-1].reshape(3, 3)
    H = H / H[2, 2]  # Chuẩn hóa phần tử cuối về 1
    return H
    ###### YOUR CODE ENDS HERE ######
    
def compute_intrinsics(homographies: List[np.ndarray]) -> np.ndarray:
    """
    Computes camera intrinsics from a list of homographies.
    
    Args:
        homographies: List of (3, 3) homography matrices, one per calibration image.
    
    Returns:
        Camera intrinsic matrix, K, with shape (3, 3).
    """
    def v(M, i, j):
        return np.array(
            [
                M[0, i] * M[0, j],
                M[0, i] * M[1, j] + M[1, i] * M[0, j],
                M[1, i] * M[1, j],
                M[2, i] * M[0, j] + M[0, i] * M[2, j],
                M[2, i] * M[1, j] + M[1, i] * M[2, j],
                M[2, i] * M[2, j],
            ]
        )

    V = np.array(
        [v(M, 0, 1) for M in homographies] \
            + [v(M, 0, 0) - v(M, 1, 1) for M in homographies]
    )

    ##### YOUR CODE STARTS HERE #####
    # 1. Phân tích SVD trên ma trận V sẵn có để lấy vector b (ứng với giá trị kỳ dị nhỏ nhất)
    _, _, Vh = np.linalg.svd(V)
    b = Vh[-1]  # b = [B11, B12, B22, B13, B23, B33]
    
    B11, B12, B22, B13, B23, B33 = b
    
      # 2. Tính các tham số nội dạng đóng theo phương pháp Zhang (Section 6.5.2)
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lam = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(max(0, lam / B11))
    beta = np.sqrt(max(0, lam * B11 / (B11 * B22 - B12**2)))
    gamma = -B12 * (alpha**2) * beta / lam
    u0 = gamma * v0 / beta - B13 * (alpha**2) / lam
    
      # 3. Đóng gói ma trận K (3, 3)
    K = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
    
    return K
      ###### YOUR CODE ENDS HERE ######
    ###### YOUR CODE ENDS HERE ######
    
def compute_extrinsics(K: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes camera extrinsics from a homography and intrinsics.

    Args:
        K: (3, 3) camera intrinsic matrix.
        H: (3, 3) homography matrix.
    
    Returns:
        R: (3, 3) camera extrinsic rotation matrix.
        t: (3, 1) camera extrinsic translation vector.
    """
    ##### YOUR CODE STARTS HERE #####
    # Hint: use np.linalg.inv() to compute KinvM = inv(K) @ M
    # Hint: use np.linalg.norm() to compute the normalization constant for r1, r2, and t
    # Hint: use np.cross() to compute r3
    # Hint: use np.column_stack() to combine r1, r2, and r3 into the matrix Q
    Kinv = np.linalg.inv(K)
    KinvH = Kinv @ H

    # Lấy các cột h1, h2, h3 đã được nhân với Kinv
    h1 = KinvH[:, 0]
    h2 = KinvH[:, 1]
    h3 = KinvH[:, 2]

    # Hằng số chuẩn hóa lambda
    lam = 1.0 / np.linalg.norm(h1)

    r1 = lam * h1
    r2 = lam * h2
    r3 = np.cross(r1, r2)
    t = (lam * h3).reshape(3, 1)

    # Ghép r1, r2, r3 thành ma trận Q và dùng SVD để ép Q thành ma trận xoay trực giao chuẩn R
    Q = np.column_stack((r1, r2, r3))
    U, _, Vt = np.linalg.svd(Q)
    R = U @ Vt
    
    ###### YOUR CODE ENDS HERE ######

    return R, t

def transform_world_to_pixel(
    K: np.ndarray, 
    R: np.ndarray, 
    t: np.ndarray, 
    P: np.ndarray
) -> np.ndarray:
    """
    Transforms 3D point in world coordinates to homogenous point in pixel 
    coordinates.

    Note: Recall the convention, where R, t are the rotation and 
    translation of the world frame with respect to the camera frame (not 
    vice versa!)

    Args:
        K: (3, 3) camera intrinsic matrix.
        R: (3, 3) camera extrinsic rotation matrix.
        t: (3, 1) camera extrinsic translation vector.
        P: (3,) 3D point in world coordinates.

    Returns:
        p: (3,) homogenous point in pixel coordinates.
    """
    ##### YOUR CODE STARTS HERE #####
    # P là điểm (3,) -> đưa về dạng đồng nhất (4, 1)
    P_homo = np.array([P[0], P[1], P[2], 1.0]).reshape(4, 1)

    # Tạo ma trận ngoại Pose [R | t] (3x4)
    Extrinsics = np.hstack((R, t))

    # Tọa độ điểm ảnh đồng nhất p = K @ [R | t] @ P_world (3x1)
    p = K @ Extrinsics @ P_homo

    return p.flatten()
    ###### YOUR CODE ENDS HERE ######