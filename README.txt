#Build map pointcloud from RGB-D camera
#Record video RGBD
python record_rgbd.py recording_01 
# Build map: Feature extraction - Compute R,t - ICP refine R,t - save and project Pointcloud
pip install open3d
python build_map_from_video.py recording_01
python build_map_from_video.py recording_01 --step 2 --icp   # run with ICP refine