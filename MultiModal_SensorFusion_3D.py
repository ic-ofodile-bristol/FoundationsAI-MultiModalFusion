import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from kitti_utils import *
import pymap3d as pm

DATA_PATH = r'2011_10_03_drive_0047_sync' 
#DATA_PATH = r'2011_09_26_drive_0005_sync'

# get RGB camera data
left_image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_02/data/*.png')))
right_image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_03/data/*.png')))

# get LiDAR data
bin_paths = sorted(glob(os.path.join(DATA_PATH, 'velodyne_points/data/*.bin')))

# get GPS/IMU data
oxts_paths = sorted(glob(os.path.join(DATA_PATH, r'oxts/data**/*.txt')))

print(f"Number of left images: {len(left_image_paths)}")
print(f"Number of right images: {len(right_image_paths)}")
print(f"Number of LiDAR point clouds: {len(bin_paths)}")
print(f"Number of GPS/IMU frames: {len(oxts_paths)}")

import os
from datetime import datetime

def calculate_fps(image_folder, timestamp_file):
    timestamps = []
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    with open(timestamp_file, 'r') as file:
        for line in file:
            timestamp_str = line.strip()
            # Extract microseconds (last three digits) separately
            timestamp_without_micro = timestamp_str[:-3]
            microseconds = int(timestamp_str[-3:])
            # Parse timestamp up to microseconds
            timestamp = datetime.strptime(timestamp_without_micro, '%Y-%m-%d %H:%M:%S.%f')
            # Add microseconds to the parsed timestamp
            timestamp = timestamp.replace(microsecond=microseconds)
            timestamps.append(timestamp)

    # Calculate average time difference between consecutive frames
    time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
    avg_time_diff = sum(time_diffs) / len(time_diffs)

    # Calculate fps
    fps = int(1 / avg_time_diff)
    return fps

# Specify the folder containing the images
image_folder = '2011_10_03_drive_0047_sync/image_02/data'

# Specify the path to the timestamp file
timestamp_file = '2011_10_03_drive_0047_sync/image_02/timestamps.txt'

# Calculate fps
fps = calculate_fps(image_folder, timestamp_file)
print("Estimated FPS:", fps)

def create_video_from_images(image_folder, output_video_path, fps=9):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    # Sort images based on their filenames
    images.sort()

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

# Specify the folder containing the images and the output video path
image_folder = '2011_10_03_drive_0047_sync/image_02/data'
output_video_path = 'kitti_video.mp4'

# Specify the frame rate (fps) of the output video
fps = 9

#create_video_from_images(image_folder, output_video_path, fps)

with open('2011_10_03_drive_0047_sync/calib_cam_to_cam.txt','r') as f:
    calib = f.readlines()

# get projection matrices (rectified left camera --> left camera (u,v,z))
P_rect2_cam2 = np.array([float(x) for x in calib[25].strip().split(' ')[1:]]).reshape((3,4))

# get rectified rotation matrices (left camera --> rectified left camera)
R_ref0_rect2 = np.array([float(x) for x in calib[24].strip().split(' ')[1:]]).reshape((3, 3,))

# add (0,0,0) translation and convert to homogeneous coordinates
R_ref0_rect2 = np.insert(R_ref0_rect2, 3, values=[0,0,0], axis=0)
R_ref0_rect2 = np.insert(R_ref0_rect2, 3, values=[0,0,0,1], axis=1)

# get rigid transformation from Camera 0 (ref) to Camera 2
R_2 = np.array([float(x) for x in calib[21].strip().split(' ')[1:]]).reshape((3,3))
t_2 = np.array([float(x) for x in calib[22].strip().split(' ')[1:]]).reshape((3,1))

# get cam0 to cam2 rigid body transformation in homogeneous coordinates
T_ref0_ref2 = np.insert(np.hstack((R_2, t_2)), 3, values=[0,0,0,1], axis=0)

T_velo_ref0 = get_rigid_transformation(r'2011_10_03_drive_0047_sync/calib_velo_to_cam.txt')
T_imu_velo = get_rigid_transformation(r'2011_10_03_drive_0047_sync/calib_imu_to_velo.txt')

# transform from velo (LiDAR) to left color camera (shape 3x4)
T_velo_cam2 = P_rect2_cam2 @ R_ref0_rect2 @ T_ref0_ref2 @ T_velo_ref0

# homogeneous transform from left color camera to velo (LiDAR) (shape: 4x4)
T_cam2_velo = np.linalg.inv(np.insert(T_velo_cam2, 3, values=[0,0,0,1], axis=0))

# transform from IMU to left color camera (shape 3x4)
T_imu_cam2 = T_velo_cam2 @ T_imu_velo

# homogeneous transform from left color camera to IMU (shape: 4x4)
T_cam2_imu = np.linalg.inv(np.insert(T_imu_cam2, 3, values=[0,0,0,1], axis=0))

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

# set confidence and IOU thresholds
model.conf = 0.7  # confidence threshold (0-1), default: 0.25
model.iou = 0.5  # NMS IoU threshold (0-1), default: 0.45

def get_uvz_centers(image, velo_uvz, bboxes, draw=True):
    ''' Obtains detected object centers projected to uvz camera coordinates.
        Starts by associating LiDAR uvz coordinates to detected object centers,
        once a match is found, the coordiantes are transformed to the uvz
        camera reference and added to the bboxes array.

        NOTE: The image is modified in place so there is no need to return it.

        Inputs:
          image - input image for detection
          velo_uvz - LiDAR coordinates projected to camera reference
          bboxes - xyxy bounding boxes form detections from yolov5 model output
          draw - (_Bool) draw measured depths on image
        Outputs:
          bboxes_out - modified array containing the object centers projected
                       to uvz image coordinates
        '''

    # unpack LiDAR camera coordinates
    u, v, z = velo_uvz

    # get new output
    bboxes_out = np.zeros((bboxes.shape[0], bboxes.shape[1] + 3))
    bboxes_out[:, :bboxes.shape[1]] = bboxes

    # iterate through all detected bounding boxes
    for i, bbox in enumerate(bboxes):
        pt1 = torch.round(bbox[0:2]).to(torch.int).numpy()
        pt2 = torch.round(bbox[2:4]).to(torch.int).numpy()

        # get center location of the object on the image
        obj_x_center = (pt1[1] + pt2[1]) / 2
        obj_y_center = (pt1[0] + pt2[0]) / 2

        # now get the closest LiDAR points to the center
        center_delta = np.abs(np.array((v, u))
                              - np.array([[obj_x_center, obj_y_center]]).T)

        # choose coordinate pair with the smallest L2 norm
        min_loc = np.argmin(np.linalg.norm(center_delta, axis=0))

        # get LiDAR location in image/camera space
        velo_depth = z[min_loc]; # LiDAR depth in camera space
        uvz_location = np.array([u[min_loc], v[min_loc], velo_depth])

        # add velo projections (u, v, z) to bboxes_out
        bboxes_out[i, -3:] = uvz_location

        # draw depth on image at center of each bounding box
        # This is depth as perceived by the camera
        if draw:
            object_center = (np.round(obj_y_center).astype(int),
                             np.round(obj_x_center).astype(int))
            cv2.putText(image,
                        '{0:.2f} m'.format(velo_depth),
                        object_center, # top left
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, # font scale
                        (255, 0, 0), 2, cv2.LINE_AA)

    return bboxes_out


def get_detection_coordinates(image, bin_path, draw_boxes=True, draw_depth=True):
    ''' Obtains detections for the input image, along with the coordinates of
        the detected object centers. The coordinate obtained are:
            - Camera with depth --> uvz
            - LiDAR/velo --> xyz
            - GPS/IMU --> xyz
        Inputs:
            image - rgb image to run detection on
            bin_path - path to LiDAR bin file
        Output:
            bboxes - array of detected bounding boxes, confidences, classes,
            velo_uv - LiDAR points porjected to camera uvz coordinate frame
            coordinates - array of all object center coordinates in the frames
                          listed above
        '''
    ## 1. compute detections in the left image
    detections = model(image)

    # draw boxes on image
    if draw_boxes:
        detections.save(save_dir="runs/kitti_debug")
    # get bounding box locations (x1,y1), (x2,y2) Prob, class
    bboxes = detections.xyxy[0].cpu() # remove from GPU

    # get LiDAR points and transform them to image/camera space
    velo_uvz = project_velobin2uvz(bin_path, T_velo_cam2, image, remove_plane=True)

    # get uvz centers for detected objects
    bboxes = get_uvz_centers(image,
                             velo_uvz,
                             bboxes,
                             draw=draw_depth)

    return bboxes, velo_uvz

def busy_from_bboxes(bboxes, img_h, img_w):
    # bboxes: torch tensor [N,6] => x1,y1,x2,y2,conf,cls
    if hasattr(bboxes, "detach"):
        b = bboxes.detach().cpu().numpy()
    else:
        b = np.asarray(bboxes)

    if b.shape[0] == 0:
        return {"count": 0, "occupancy": 0.0, "mean_area": 0.0}

    areas = (b[:,2] - b[:,0]) * (b[:,3] - b[:,1])
    occupancy = areas.sum() / (img_w * img_h)
    return {
        "count": int(b.shape[0]),
        "occupancy": float(occupancy),
        "mean_area": float(areas.mean() / (img_w * img_h))
    }
   
def lidar_busy_metrics(velo_uvz, img_h, img_w, near_m=15.0):
    u, v, z = velo_uvz
    mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h) & (z > 0)
    u, v, z = u[mask], v[mask], z[mask]

    n = z.size
    if n == 0:
        return {"n": 0, "density": 0.0, "near_ratio": 0.0, "z_iqr": 0.0}

    density = n / (img_h * img_w)                 # points per pixel
    near_ratio = float(np.mean(z < near_m))       # fraction of points closer than near_m
    z_iqr = float(np.percentile(z, 75) - np.percentile(z, 25))  # depth spread

    return {"n": int(n), "density": float(density), "near_ratio": near_ratio, "z_iqr": z_iqr}

def lidar_entropy(velo_uvz, img_h, img_w, bins=(60, 20)):
    u, v, z = velo_uvz
    mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h) & (z > 0)
    u, v = u[mask], v[mask]

    H, _, _ = np.histogram2d(v, u, bins=bins, range=[[0, img_h], [0, img_w]])
    p = H.flatten()
    p = p[p > 0]
    p = p / p.sum()
    ent = -np.sum(p * np.log(p))
    return float(ent)

def imu2geodetic(x, y, z, lat0, lon0, alt0, heading0):
    # convert to RAE
    rng = np.sqrt(x**2 + y**2 + z**2)
    az = np.degrees(np.arctan2(y, x)) + np.degrees(heading0)
    el = np.degrees(np.arctan2(np.sqrt(x**2 + y**2), z)) + 90

    # convert to geodetic
    lla = pm.aer2geodetic(az, el, rng, lat0, lon0, alt0)

    # convert to numpy array
    lla = np.vstack((lla[0], lla[1], lla[2])).T

    return lla

def normalized_lidar_entropy(velo_uvz, img_h, img_w, bins=(60,20)):
    ent = lidar_entropy(velo_uvz, img_h, img_w, bins)
    max_ent = np.log(bins[0] * bins[1])
    return ent / max_ent

def lidar_density_heatmap(velo_uvz, img_h, img_w, bins=(80, 30)):
        u, v, z = velo_uvz
        mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h) & (z > 0)
        u, v = u[mask], v[mask]

        H, xedges, yedges = np.histogram2d(
            v, u, bins=bins, range=[[0, img_h], [0, img_w]]
        )
        return H

# ============================================================
if __name__ == "__main__":
    index = 10

    left_image = cv2.cvtColor(cv2.imread(left_image_paths[index]), cv2.COLOR_BGR2RGB)
    bin_path = bin_paths[index]

    cv2.imshow("left Images", cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Left image loaded")
    oxts_frame = get_oxts(oxts_paths[index])

    # get detections and object centers in uvz
    bboxes, velo_uvz = get_detection_coordinates(left_image, bin_path)

    cv2.imshow("Bounding Box Detection", cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    h, w = left_image.shape[:2]
    busy2d = busy_from_bboxes(bboxes, h, w)
    print("2D busy metrics:", busy2d)

    busy_score = busy2d["count"] + 50 * busy2d["occupancy"]
    print("Overall busy score:", busy_score)


    m = lidar_busy_metrics(velo_uvz, h, w, near_m=15.0)
    busy_urban_score = (m["density"] * 1e6) + 50*m["near_ratio"] + 0.5*m["z_iqr"]
    print(m, "busy_urban_score:", busy_urban_score)


    ent = lidar_entropy(velo_uvz, h, w)
    #busy_urban_score = busy_urban_score + 10*ent
    print("entropy:", ent)
    #print("Final busy_urban_score:", busy_urban_score)


    n_ent = normalized_lidar_entropy(velo_uvz, h, w)
    print("normalized entropy:", n_ent)
    # get transformed coordinates of object centers
    uvz = bboxes[:, -3:]

    # transform to (u,v,z)
    #velo_xyz = transform_uvz(uvz, T_cam2_velo) # we can also get LiDAR coordiantes

    imu_xyz = transform_uvz(uvz, T_cam2_imu)

    # get Lat/Lon on each detected object
    lat0 = oxts_frame[0]
    lon0 = oxts_frame[1]
    alt0 = oxts_frame[2]
    heading0 = oxts_frame[5]

    lla = imu2geodetic(imu_xyz[:, 0], imu_xyz[:, 1], imu_xyz[:, 2], lat0, lon0, alt0, heading0)

    velo_image = draw_velo_on_image(velo_uvz, np.zeros_like(left_image))
    #print("Detected object locations (Lat, Lon, Alt):")

    stacked = np.vstack((left_image, velo_image))

    cv2.imshow("Stacked Images", cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    left_image_2 = cv2.cvtColor(cv2.imread(left_image_paths[index]), cv2.COLOR_BGR2RGB)
    velo_image_2 = draw_velo_on_image(velo_uvz, left_image_2)

    cv2.imshow("left image 2", cv2.cvtColor(left_image_2, cv2.COLOR_RGB2BGR))
    #cv2.imshow("LiDAR projected", cv2.cvtColor(velo_image_2, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ============================================================
    # RESULTS & VISUALISATION SECTION (FOR REPORTING)
    # ============================================================

    RESULTS_DIR = "results_kitti"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("SUMMARY OF EXPERIMENTAL RESULTS")
    print("="*60)

    # ---- 1. Textual summary (for narrative explanation) ----
    print("\n[Semantic (AI-based) perception]")
    print(f"Number of detected objects: {busy2d['count']}")
    print(f"Image occupancy by objects: {busy2d['occupancy']:.3f}")
    print(f"Mean object area (fraction of image): {busy2d['mean_area']:.3f}")
    print(f"Overall 2D busy score: {busy_score:.2f}")

    print("\n[Geometric (LiDAR-based) scene structure]")
    print(f"Number of LiDAR points (in image FOV): {m['n']}")
    print(f"LiDAR point density (points/pixel): {m['density']:.4f}")
    print(f"Near-field ratio (<15 m): {m['near_ratio']:.2f}")
    print(f"Depth IQR (m): {m['z_iqr']:.2f}")
    print(f"LiDAR entropy (nats): {ent:.2f}")
    print(f"Normalised LiDAR entropy: {n_ent:.2f}")

    print("\nInterpretation:")
    print("- Moderate visual busyness (AI detections)")
    print("- Strong geometric clutter (LiDAR density + entropy)")
    print("- Scene classified as dense urban environment")

    # ---- 2. Save combined RGB + LiDAR image ----
    combined_rgb_lidar = np.vstack((left_image, velo_image))
    combined_path = os.path.join(RESULTS_DIR, "rgb_lidar_stacked.png")
    cv2.imwrite(combined_path, cv2.cvtColor(combined_rgb_lidar, cv2.COLOR_RGB2BGR))
    print(f"\nSaved combined RGB–LiDAR image to: {combined_path}")

    # ---- 3. LiDAR spatial density heatmap (image space) ----
    H = lidar_density_heatmap(velo_uvz, h, w)

    plt.figure(figsize=(10, 4))
    plt.imshow(H, cmap="hot", aspect="auto")
    plt.colorbar(label="LiDAR point count")
    plt.title("LiDAR Spatial Density Heatmap (Image Space)")
    plt.xlabel("Image width bins")
    plt.ylabel("Image height bins")
    heatmap_path = os.path.join(RESULTS_DIR, "lidar_density_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=200)
    plt.close()
    print(f"Saved LiDAR density heatmap to: {heatmap_path}")

    # ---- 4. Overlay heatmap on RGB (intuitive figure) ----
    H_norm = (H / (H.max() + 1e-6))
    H_resized = cv2.resize(H_norm, (w, h))
    heatmap_color = cv2.applyColorMap((255 * H_resized).astype(np.uint8), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR),
        0.6,
        heatmap_color,
        0.4,
        0
    )

    overlay_path = os.path.join(RESULTS_DIR, "rgb_lidar_heatmap_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"Saved RGB + LiDAR heatmap overlay to: {overlay_path}")

    # ---- 5. Bar chart of key metrics (report-friendly) ----
    labels = [
        "Object count",
        "2D occupancy",
        "LiDAR density",
        "Near-field ratio",
        "Normalised entropy"
    ]
    values = [
        busy2d["count"],
        busy2d["occupancy"],
        m["density"],
        m["near_ratio"],
        n_ent
    ]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.title("Key Scene Density Indicators")
    plt.ylabel("Normalised / Raw Value")
    plt.xticks(rotation=20)
    metrics_path = os.path.join(RESULTS_DIR, "scene_density_metrics.png")
    plt.tight_layout()
    plt.savefig(metrics_path, dpi=200)
    plt.close()
    print(f"Saved metrics comparison plot to: {metrics_path}")

    print("\nAll results saved in:", RESULTS_DIR)
    print("="*60)
