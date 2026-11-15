import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# =====================================================================
# STEP 0: KHỞI TẠO CÁC THAM SỐ NỘI CAMERA (INTRINSICS K)
# =====================================================================
# Thay thế các tham số này bằng thông số K thực tế từ kết quả Calib camera của bạn!
fx = 386.376220703125 # Tiêu cự theo chiều ngang (pixel)
fy = 385.9943542480469 # Tiêu cự theo chiều dọc (pixel)
cx = 325.576416015625 # Tâm quang học u0 (pixel)
cy = 241.9173126220703 # Tâm quang học v0 (pixel)


# Ma trận Intrinsic K (3x3)
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

# Hệ số tỉ lệ độ sâu (Depth Scale):
# Thông thường ảnh depth 16-bit lưu đơn vị mm -> chia 1000.0 để ra Mét.
DEPTH_FACTOR = 0.001 #m/unit


# =====================================================================
# STEP 1: FEATURE DETECTION & DESCRIPTION (ORB)
# =====================================================================
def extract_features(img_color):
  """Feature extraction Keypoints và Descriptors by ORB."""
  orb = cv2.ORB_create(nfeatures=5000)
  keypoints, descriptors = orb.detectAndCompute(img_color, None)
  return keypoints, descriptors


# =====================================================================
# STEP 2: FEATURE MATCHING (BRUTE-FORCE WITH CROSS-CHECK)
# =====================================================================
def match_features(desc1, desc2):
  """Ghép cặp đặc trưng giữa 2 ảnh dùng khoảng cách Hamming."""
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
  matches = bf.match(desc1, desc2)

  # # Sắp xếp các cặp khớp theo thứ tự khoảng cách tăng dần (khớp tốt nhất lên đầu)
  # matches = sorted(matches, key=lambda x: x.distance)
  # return matches

  knn_matches = bf.knnMatch(desc1, desc2, k=2)

  good_matches = []

  for m, n in knn_matches:

      if m.distance < 0.75 * n.distance:
          good_matches.append(m)

  return good_matches


# =====================================================================
# HÀM PHỤ TRỢ: CHIẾU NGƯỢC 2D -> 3D DỰA VÀO ẢNH DEPTH (UNPROJECT 2D TO 3D)
# =====================================================================
def pixel_to_3d(u, v, depth_img, K):
  """Chuyển tọa độ pixel (u, v) và Depth d thành tọa độ 3D (X, Y, Z) trong hệ camera.

  Công thức: Z = d / depth_factor X = (u - cx) * Z / fx Y = (v - cy) * Z / fy
  """
  h, w = depth_img.shape
  u_int, v_int = int(round(u)), int(round(v))

  if u_int < 0 or u_int >= w or v_int < 0 or v_int >= h:
    return None

  z_raw = depth_img[v_int, u_int]
  if z_raw == 0 or np.isnan(z_raw):
    return None  # Không có dữ liệu chiều sâu tại pixel này

  z = z_raw * DEPTH_FACTOR
  x = (u - K[0, 2]) * z / K[0, 0]
  y = (v - K[1, 2]) * z / K[1, 1]

  return np.array([x, y, z], dtype=np.float64)


# =====================================================================
# STEP 3 & 4: GEOMETRIC VERIFICATION & POSE ESTIMATION (PnP RANSAC)
# =====================================================================
def estimate_pose_rgbd(kp1, kp2, matches, depth1, K):
  """Ước lượng  R, t từ Frame 1 sang Frame 2 bằng Perspective-n-Point

  (PnP) kết hợp RANSAC.
  """
  pts_3d = []  # Tọa độ 3D các điểm thuộc Frame 1
  pts_2d = []  # Tọa độ 2D tương ứng của điểm đó trên Frame 2

  valid_matches = []

  for m in matches:
    # Lấy tọa độ pixel trên Frame 1 và Frame 2
    u1, v1 = kp1[m.queryIdx].pt
    u2, v2 = kp2[m.trainIdx].pt

    # Chiếu điểm 2D trên Frame 1 thành 3D nhờ ảnh Depth 1
    p3d = pixel_to_3d(u1, v1, depth1, K)

    if p3d is not None:
      pts_3d.append(p3d)
      pts_2d.append([u2, v2])
      valid_matches.append(m)

  pts_3d = np.array(pts_3d, dtype=np.float64)
  pts_2d = np.array(pts_2d, dtype=np.float64)

  if len(pts_3d) < 6:
    print("Không đủ điểm 3D hợp lệ để giải PnP!")
    return None, None, None, None

  # Giải bài toán PnP với RANSAC để tìm R, t và loại bỏ Outliers
  success, rvec, tvec, inliers = cv2.solvePnPRansac(
      objectPoints=pts_3d,
      imagePoints=pts_2d,
      cameraMatrix=K,
      distCoeffs=None,
      flags=cv2.SOLVEPNP_AP3P,
      reprojectionError=3.0,
      confidence=0.99,
  )

  if not success or inliers is None:
    print(" PnP RANSAC thất bại!")
    return None, None, None, None

  # Chuyển rvec (Rodrigues vector 3x1) sang Ma trận xoay R (3x3)
  R, _ = cv2.Rodrigues(rvec)
  t = tvec

  return R, t, valid_matches, inliers


# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================
if __name__ == "__main__":
  # 1. Đọc ảnh Màu (RGB) và ảnh Chiều sâu (Depth) của 2 khung hình
  # Thay đổi tên file theo đúng ảnh bạn đang có
  # Define paths.
  DATA_PATH = os.path.join(os.getcwd(), 'images')  # Hoặc 'chessboard' tùy tên thư mục chứa ảnh của bạn
  img0 = os.path.join(DATA_PATH, 'color_0.png')
  img1 = os.path.join(DATA_PATH, 'color_1.png')
  depth_img0 = os.path.join(DATA_PATH, 'depth_0.png')
  depth_img1 = os.path.join(DATA_PATH, 'depth_1.png')
  
  color1 = cv2.imread(img0)
  depth1 = cv2.imread(depth_img0, cv2.IMREAD_UNCHANGED)

  color2 = cv2.imread(img1)
  depth2 = cv2.imread(depth_img1, cv2.IMREAD_UNCHANGED)

  # --- STEP 1: Feature Extraction ---
  kp1, desc1 = extract_features(color1)
  kp2, desc2 = extract_features(color2)
  print(f"Frame 1: Tìm thấy {len(kp1)} keypoints")
  print(f"Frame 2: Tìm thấy {len(kp2)} keypoints")
  img_keypoints1 = cv2.drawKeypoints(color1, kp1, None, color=(0, 255, 0))
  img_keypoints2 = cv2.drawKeypoints(color2, kp2, None, color=(0, 255, 0))
  cv2.imshow("ORB Keypoints", np.hstack((img_keypoints1, img_keypoints2)))
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  # --- STEP 2: Feature Matching ---
  matches = match_features(desc1, desc2)
  print(f"Số cặp khớp thô: {len(matches)}")

  img_raw = cv2.drawMatches(
    color1,
    kp1,
    color2,
    kp2,
    matches[:300],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
  )
  plt.figure(figsize=(16, 7))
  plt.imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
  plt.title("Raw ORB Matches")
  plt.axis("off")
  plt.show()
  
  # --- STEP 3 & 4: Pose Estimation qua PnP RANSAC ---
  R, t, valid_matches, inliers = estimate_pose_rgbd(
      kp1, kp2, matches, depth1, K
  )

  if R is not None:
    print("\n TÍNH TOÁN THÀNH CÔNG TƯ THẾ CAMERA (POSE):")
    print("Ma trận xoay R (Rotation Matrix):")
    print(R)
    print("\nVector tịnh tiến t (Translation Vector in meters):")
    print(t.ravel())
    print(
        f"\nSố lượng điểm Inliers khớp chính xác (khử nhiễu bởi RANSAC):"
        f" {len(inliers)} / {len(valid_matches)}"
    )

    # --- HÌNH ẢNH HÓA KẾT QUẢ KHỚP ĐIỂM (INLIERS MATCHES) ---
    inlier_matches = [valid_matches[i[0]] for i in inliers]
    img_matches = cv2.drawMatches(
        color1,
        kp1,
        color2,
        kp2,
        inlier_matches[:200],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    plt.figure(figsize=(15, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title("50 Inlier Matches (Geometric Verification Passed)")
    plt.axis("off")
    plt.show()