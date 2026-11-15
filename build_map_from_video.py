"""
Dựng bản đồ 3D từ chuỗi ảnh RGB-D đã quay (record_rgbd.py).

Chạy:
  python build_map_from_video.py recording_01
  python build_map_from_video.py recording_01 --step 2 --icp --no-show

Ý tưởng (mỗi frame):
  1. ORB + ghép với KEYFRAME hiện tại
  2. depth của keyframe -> điểm 3D -> PnP RANSAC -> pose frame hiện tại so với keyframe
  3. nếu camera đã di chuyển đủ xa -> frame này thành KEYFRAME MỚI
  4. cloud của mỗi keyframe được đưa về hệ world (= hệ camera frame đầu) và gộp vào bản đồ

Cài đặt:  pip install open3d opencv-python numpy
"""
import argparse
import copy
import glob
import json
import os

import cv2
import numpy as np
import open3d as o3d

# =====================================================================
# THAM SỐ
# =====================================================================
N_FEATURES = 3000
MIN_Z, MAX_Z = 0.3, 4.0  # m, lọc depth
VOXEL_SIZE = 0.02  # m, độ mịn của bản đồ

KF_TRANS = 0.08  # m   : tạo keyframe mới khi dịch quá ngưỡng này...
KF_ROT = 8.0  # độ  : ...hoặc xoay quá ngưỡng này
MIN_INLIERS = 30  # ít hơn thì coi là tracking thất bại ở frame đó
WEAK_INLIERS = 100  # ít hơn (nhưng vẫn hợp lệ) thì tạo keyframe mới để "làm mới"
MAX_LOST = 15  # số frame thất bại liên tiếp tối đa trước khi dừng


# =====================================================================
# TRACKER (không phụ thuộc open3d)
# =====================================================================
class RGBDTracker:
  """Theo dõi pose camera, bám theo keyframe. pose = T_world_camera (4x4)."""

  def __init__(self, K, depth_scale):
    self.K = K
    self.depth_scale = depth_scale
    self.orb = cv2.ORB_create(nfeatures=N_FEATURES)
    self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    self.kf = None  # dict: kp, des, depth, pose
    self.lost = 0

  def detect(self, color):
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    return self.orb.detectAndCompute(gray, None)

  def match(self, des1, des2):
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
      return []
    good = []
    for pair in self.bf.knnMatch(des1, des2, k=2):
      if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
        good.append(pair[0])
    return good

  def relative_pose(self, kp_cur, matches):
    """PnP: điểm 3D của keyframe -> ảnh hiện tại. Trả về (T_cur_kf, n_inliers) hoặc None."""
    kf = self.kf
    h, w = kf["depth"].shape
    uv_kf = np.array([kf["kp"][m.queryIdx].pt for m in matches])
    uv_cur = np.array([kp_cur[m.trainIdx].pt for m in matches], dtype=np.float64)

    ui = np.clip(np.round(uv_kf[:, 0]).astype(int), 0, w - 1)
    vi = np.clip(np.round(uv_kf[:, 1]).astype(int), 0, h - 1)
    z = kf["depth"][vi, ui].astype(np.float64) * self.depth_scale
    ok = (z > MIN_Z) & (z < MAX_Z)
    if ok.sum() < 6:
      return None
    z, uv_kf, uv_cur = z[ok], uv_kf[ok], uv_cur[ok]

    K = self.K
    x = (uv_kf[:, 0] - K[0, 2]) * z / K[0, 0]
    y = (uv_kf[:, 1] - K[1, 2]) * z / K[1, 1]
    pts3d = np.stack([x, y, z], axis=1)

    success, rvec, tvec, inl = cv2.solvePnPRansac(
        pts3d, uv_cur, K, None, flags=cv2.SOLVEPNP_AP3P,
        reprojectionError=3.0, confidence=0.99)
    if not success or inl is None or len(inl) < MIN_INLIERS:
      return None
    idx = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(pts3d[idx], uv_cur[idx], K, None, rvec, tvec)

    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T, len(idx)

  def set_keyframe(self, kp, des, depth, pose):
    self.kf = {"kp": kp, "des": des, "depth": depth, "pose": pose}

  def track(self, color, depth):
    """
    Trả về (pose, is_new_keyframe, n_inliers). pose = None nếu tracking thất bại.
    """
    kp, des = self.detect(color)

    if self.kf is None:  # frame đầu tiên = keyframe đầu tiên, world = hệ camera này
      self.set_keyframe(kp, des, depth, np.eye(4))
      return np.eye(4), True, 0

    matches = self.match(self.kf["des"], des)
    res = self.relative_pose(kp, matches) if len(matches) >= 6 else None
    if res is None:
      self.lost += 1
      return None, False, 0
    self.lost = 0

    T_cur_kf, n_inl = res
    T_kf_cur = np.linalg.inv(T_cur_kf)
    pose = self.kf["pose"] @ T_kf_cur  # T_world_cur

    trans = np.linalg.norm(T_kf_cur[:3, 3])
    cos = np.clip((np.trace(T_kf_cur[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    rot = np.degrees(np.arccos(cos))

    new_kf = trans > KF_TRANS or rot > KF_ROT or n_inl < WEAK_INLIERS
    if new_kf:
      self.set_keyframe(kp, des, depth, pose)
    return pose, new_kf, n_inl


# =====================================================================
# POINT CLOUD / MAP (dùng open3d)
# =====================================================================
def make_cloud(color, depth, K, depth_scale):
  """RGB-D -> point cloud màu trong hệ camera của frame đó (vector hóa)."""
  h, w = depth.shape
  v, u = np.indices((h, w))
  z = depth.astype(np.float64) * depth_scale
  mask = (z > MIN_Z) & (z < MAX_Z)
  x = (u - K[0, 2]) * z / K[0, 0]
  y = (v - K[1, 2]) * z / K[1, 1]
  pts = np.stack([x, y, z], axis=-1)[mask]
  cols = color[..., ::-1][mask].astype(np.float64) / 255.0

  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(pts)
  pcd.colors = o3d.utility.Vector3dVector(cols)
  return pcd.voxel_down_sample(VOXEL_SIZE)


def icp_refine(src_cam, tgt_world, pose_init):
  """Tinh chỉnh pose của cloud mới so với cloud keyframe trước (point-to-plane)."""
  search = o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30)
  src = copy.deepcopy(src_cam)
  src.estimate_normals(search)
  if not tgt_world.has_normals():
    tgt_world.estimate_normals(search)
  est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
  crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
  init_eval = o3d.pipelines.registration.evaluate_registration(src, tgt_world, 0.04, pose_init)
  res = o3d.pipelines.registration.registration_icp(src, tgt_world, 0.04, pose_init, est, crit)
  if res.fitness >= init_eval.fitness:  # chỉ nhận nếu không làm tệ đi
    return res.transformation
  return pose_init


def mat_to_quat(R):
  """Ma trận xoay -> quaternion (qx, qy, qz, qw)."""
  tr = np.trace(R)
  if tr > 0:
    s = np.sqrt(tr + 1.0) * 2
    return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
  if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
    s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
    return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
  if R[1, 1] > R[2, 2]:
    s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
    return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
  s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
  return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s)


def show_map(global_map, kf_positions):
  """Camera OpenCV có y xuống / z tới -> lật để hiển thị cho dễ nhìn."""
  flip = np.diag([1.0, -1.0, -1.0, 1.0])
  geoms = [copy.deepcopy(global_map).transform(flip)]
  if len(kf_positions) >= 2:
    pts = (np.array(kf_positions) @ flip[:3, :3].T)
    lines = [[i, i + 1] for i in range(len(pts) - 1)]
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts), o3d.utility.Vector2iVector(lines))
    ls.paint_uniform_color([1, 0, 0])  # quỹ đạo camera = đường đỏ
    geoms.append(ls)
  geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))
  o3d.visualization.draw_geometries(geoms, window_name="Global map + quỹ đạo (đỏ)")


# =====================================================================
# MAIN
# =====================================================================
def main(rec_dir, step, use_icp, show):
  info = json.load(open(os.path.join(rec_dir, "intrinsics.json")))
  K = np.array([[info["fx"], 0, info["cx"]], [0, info["fy"], info["cy"]], [0, 0, 1]], dtype=np.float64)
  depth_scale = info["depth_scale"]

  color_files = sorted(glob.glob(os.path.join(rec_dir, "color", "*.png")))
  depth_files = sorted(glob.glob(os.path.join(rec_dir, "depth", "*.png")))
  if not color_files or len(color_files) != len(depth_files):
    raise FileNotFoundError("Thiếu ảnh hoặc số ảnh color/depth không bằng nhau.")
  print(f"{len(color_files)} frame, dùng mỗi {step} frame. K =\n{np.round(K, 1)}")

  tracker = RGBDTracker(K, depth_scale)
  global_map = o3d.geometry.PointCloud()
  prev_kf_cloud_world = None
  kf_positions, all_poses = [], []
  n_kf, n_lost = 0, 0

  for n, i in enumerate(range(0, len(color_files), step)):
    color = cv2.imread(color_files[i])
    depth = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
    if color is None or depth is None or depth.dtype != np.uint16:
      print(f"Bỏ qua frame {i}: ảnh lỗi hoặc depth không phải uint16")
      continue

    pose, new_kf, n_inl = tracker.track(color, depth)

    if pose is None:
      n_lost += 1
      if tracker.lost > MAX_LOST:
        print(f"Mất tracking quá {MAX_LOST} frame liên tiếp ở frame {i}. Dừng và lưu bản đồ hiện có.")
        break
      continue

    all_poses.append((i, pose))

    if new_kf:
      cloud_cam = make_cloud(color, depth, K, depth_scale)
      if use_icp and prev_kf_cloud_world is not None:
        pose = icp_refine(cloud_cam, prev_kf_cloud_world, pose)
        tracker.kf["pose"] = pose  # các frame sau bám theo pose đã refine
      cloud_world = copy.deepcopy(cloud_cam).transform(pose)  # X_world = pose @ X_cam
      global_map = (global_map + cloud_world).voxel_down_sample(VOXEL_SIZE)
      prev_kf_cloud_world = cloud_world
      kf_positions.append(pose[:3, 3].copy())
      n_kf += 1

    if n % 50 == 0:
      print(f"frame {i}/{len(color_files)}  inliers={n_inl}  keyframes={n_kf}  "
            f"lost={n_lost}  map={len(global_map.points)} điểm")

  # ---- kết thúc ----
  global_map, _ = global_map.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
  o3d.io.write_point_cloud(os.path.join(rec_dir, "global_map.ply"), global_map)

  with open(os.path.join(rec_dir, "trajectory.txt"), "w") as f:  # định dạng TUM -> dùng được với evo
    for i, p in all_poses:
      q = mat_to_quat(p[:3, :3])
      f.write(f"{i / 30.0:.4f} {p[0, 3]:.5f} {p[1, 3]:.5f} {p[2, 3]:.5f} "
              f"{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}\n")

  path_len = sum(np.linalg.norm(np.array(kf_positions[k + 1]) - kf_positions[k])
                 for k in range(len(kf_positions) - 1)) if len(kf_positions) > 1 else 0.0
  print("\n================ KẾT QUẢ ================")
  print(f"Keyframe: {n_kf} | frame thất bại: {n_lost} | điểm trong map: {len(global_map.points)}")
  print(f"Độ dài quãng đường camera: {path_len:.2f} m")
  if len(kf_positions) > 1:
    gap = np.linalg.norm(np.array(kf_positions[-1]) - kf_positions[0])
    print(f"Khoảng cách điểm đầu - điểm cuối: {gap:.2f} m")
    print("  (nếu bạn quay về đúng chỗ xuất phát thì số này nên ~0; phần chênh lệch chính là DRIFT)")
  print(f"Đã lưu: {rec_dir}/global_map.ply và trajectory.txt")

  if show:
    show_map(global_map, kf_positions)


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("rec_dir", help="thư mục quay bằng record_rgbd.py")
  ap.add_argument("--step", type=int, default=1, help="dùng mỗi N frame (2-3 để chạy nhanh hơn)")
  ap.add_argument("--icp", action="store_true", help="refine pose keyframe bằng ICP")
  ap.add_argument("--no-show", action="store_true", help="không mở cửa sổ 3D")
  a = ap.parse_args()
  main(a.rec_dir, a.step, a.icp, not a.no_show)
