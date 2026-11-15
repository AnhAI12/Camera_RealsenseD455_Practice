"""
Quay "video" RGB-D từ RealSense D455 và lưu thành thư mục ảnh.

Chạy:
  python record_rgbd.py recording_01

Phím:  SPACE = bắt đầu / dừng ghi,   q = thoát

Kết quả:
  recording_01/color/000000.png ...   (ảnh màu 8-bit)
  recording_01/depth/000000.png ...   (ảnh depth 16-bit, đơn vị thô, depth_scale mỗi đơn vị = mét)
  recording_01/intrinsics.json        (fx, fy, cx, cy, depth_scale của stream màu)
  recording_01/timestamps.txt         (index  thời gian giây)

Cài đặt:  pip install pyrealsense2 opencv-python numpy
"""
import argparse
import json
import os
import queue
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs

WIDTH, HEIGHT, FPS = 640, 480, 30


def writer_thread(q, color_dir, depth_dir):
  """Ghi file ở luồng riêng để vòng lặp chính không bị chậm (tránh rớt frame)."""
  params = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # nén nhẹ = ghi nhanh
  while True:
    item = q.get()
    if item is None:
      break
    i, color, depth = item
    cv2.imwrite(os.path.join(color_dir, f"{i:06d}.png"), color, params)
    cv2.imwrite(os.path.join(depth_dir, f"{i:06d}.png"), depth, params)  # uint16 giữ nguyên


def main(out_dir):
  color_dir = os.path.join(out_dir, "color")
  depth_dir = os.path.join(out_dir, "depth")
  os.makedirs(color_dir, exist_ok=True)
  os.makedirs(depth_dir, exist_ok=True)

  # ---- cấu hình camera: màu + depth cùng độ phân giải ----
  pipe, cfg = rs.pipeline(), rs.config()
  cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
  cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
  prof = pipe.start(cfg)

  # depth_scale: 1 đơn vị trong ảnh depth = bao nhiêu mét (D455 thường là 0.001)
  depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
  # intrinsics của stream MÀU (vì depth sẽ được align vào ảnh màu)
  intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
  info = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
          "depth_scale": depth_scale, "width": intr.width, "height": intr.height}
  with open(os.path.join(out_dir, "intrinsics.json"), "w") as f:
    json.dump(info, f, indent=2)
  print("Intrinsics:", info)

  # align: chiếu từng pixel depth lên đúng pixel của ảnh màu
  align = rs.align(rs.stream.color)

  q = queue.Queue(maxsize=300)
  th = threading.Thread(target=writer_thread, args=(q, color_dir, depth_dir), daemon=True)
  th.start()

  recording, n_saved, n_dropped = False, 0, 0
  stamps = []
  t0 = time.time()
  try:
    while True:
      frames = align.process(pipe.wait_for_frames())
      cf, df = frames.get_color_frame(), frames.get_depth_frame()
      if not cf or not df:
        continue
      color = np.asanyarray(cf.get_data())
      depth = np.asanyarray(df.get_data())  # uint16

      if recording:
        try:
          q.put_nowait((n_saved, color.copy(), depth.copy()))
          stamps.append((n_saved, time.time() - t0))
          n_saved += 1
        except queue.Full:
          n_dropped += 1

      # ---- hiển thị ----
      preview = color.copy()
      status = f"REC {n_saved} frames (queue {q.qsize()})" if recording else "SPACE = bat dau ghi"
      cv2.putText(preview, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                  (0, 0, 255) if recording else (0, 255, 0), 2)
      # depth chỉ để NHÌN: đỏ/xanh theo khoảng cách; vùng đen = không đo được (hố)
      depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.05), cv2.COLORMAP_JET)
      depth_vis[depth == 0] = 0
      cv2.imshow("color", preview)
      cv2.imshow("depth (aligned)", depth_vis)

      k = cv2.waitKey(1) & 0xFF
      if k == ord("q"):
        break
      if k == 32:
        recording = not recording
        print("Bat dau ghi..." if recording else "Dung ghi.")
  finally:
    pipe.stop()
    cv2.destroyAllWindows()
    q.put(None)
    th.join()  # đợi ghi nốt các file còn trong hàng đợi

  with open(os.path.join(out_dir, "timestamps.txt"), "w") as f:
    for i, t in stamps:
      f.write(f"{i} {t:.4f}\n")
  print(f"Xong: {n_saved} frame, rớt {n_dropped} frame. Thư mục: {out_dir}")
  if n_dropped:
    print("Có frame bị rớt vì ghi file không kịp -> quay ngắn hơn hoặc dùng ổ SSD.")


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("out_dir", nargs="?", default="recording_01")
  main(ap.parse_args().out_dir)
