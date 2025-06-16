import os
import numpy as np
import cv2 as cv
import pickle
import rospy
import rosbag
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import yaml
from tf.transformations import euler_from_quaternion

BAG_FILES = [
    # '/home/gonz/Desktop/bags/sec_a_1_2025-05-23-21-44-37.bag',
    # '/home/gonz/Desktop/bags/sec_a_2_2025-05-23-21-47-14.bag',
    # '/home/gonz/Desktop/bags/sec_a_3_2025-05-23-21-48-28.bag',
    # '/home/gonz/Desktop/bags/sec_a_4_2025-05-23-21-50-01.bag',
    # '/home/gonz/Desktop/bags/sec_a_5_2025-05-23-21-51-58.bag',
    # '/home/gonz/Desktop/bags/sec_a_6_2025-05-23-21-54-03.bag'
    '/home/gonz/Desktop/bags/sec_b_1_2025-05-23-21-28-55.bag',
    '/home/gonz/Desktop/bags/sec_b_2_2025-05-23-21-30-07.bag',
    '/home/gonz/Desktop/bags/sec_b_3_2025-05-23-21-32-14.bag',
    '/home/gonz/Desktop/bags/sec_b_4_2025-05-23-21-34-43.bag',
    '/home/gonz/Desktop/bags/sec_b_5_2025-05-23-21-35-52.bag',
    '/home/gonz/Desktop/bags/sec_b_6_2025-05-23-21-37-48.bag',
    '/home/gonz/Desktop/bags/sec_b_7_2025-05-23-21-39-06.bag'
]

MAP_PATH = "/home/gonz/Desktop/THESIS/code/global-planning/fcfm_navigation_dataset/ros_map_utils/maps/cancha.png"
DATA_DIR = "/home/gonz/Desktop/THESIS/code/global-planning/gnd_dataset/local_map_files_120"

LOCAL_MAPS_DIR  = os.path.join(DATA_DIR, "dd")
LOCAL_PATHS_DIR = os.path.join(DATA_DIR, "cc")


MAP_RES = 0.1
# MAP_ORIGIN = [-34.8, -81.2]  # electrica
MAP_ORIGIN = [-57.2, -90.8]   # cancha
N_LIDAR = 3  # Number of lidar messages

odom_topic = "/panther/odometry/filtered"
scan_topic = "/repub/ouster/points"
amcl_topic = "/amcl_pose"
img_topic  = "/repub/camera/image_raw"


def world_to_map(map_origin, map_res, x, y):
    """
    Convert world coordinates to map coordinates.
    :param x: x coordinate in world
    :param y: y coordinate in world
    :return: x, y coordinates in map
    """
    x_map = (x - map_origin[0]) / map_res
    y_map = (y - map_origin[1]) / map_res
    return x_map, y_map


def get_path_lenght_interval(odometry_xy, start, lenght=15., N_wpts=15):
    xsq = (odometry_xy[start+1:, 0] - odometry_xy[start:-1, 0]) ** 2
    ysq = (odometry_xy[start+1:, 1] - odometry_xy[start:-1, 1]) ** 2
    distances = np.cumsum(np.sqrt(xsq + ysq), axis=0)

    stop_idx = np.where(distances >= lenght)[0][0] + start
    ids = np.linspace(start, stop_idx, N_wpts, dtype=np.int64) # inlcude last point

    return odometry_xy[ids]


def rotate_image(image, angle, center=None):
  if center is None:
    center = tuple(np.array(image.shape[1::-1]) / 2)
  
  rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result, rot_mat


def global_to_local(points_world, robot_pose):
    """
    Convert world-frame points to robot-centric local frame.

    :param points_world: Nx2 array of (x, y) points in meters
    :param robot_pose: tuple (x_r, y_r, theta_r)
    :return: Nx2 array of transformed points in local frame
    """
    x_r, y_r, theta_r = robot_pose

    # Translate points so robot is at the origin
    translated = points_world - np.array([x_r, y_r])

    # Rotation matrix to align robot's heading with +X
    c, s = np.cos(-theta_r), np.sin(-theta_r)
    R = np.array([[c, -s],
                  [s,  c]])

    local_points = translated @ R.T
    return local_points


def get_local_map(map, pose, map_origin, map_res, size_m=30, flip=True, color=None):
    px, py = world_to_map(map_origin, map_res, pose[0], pose[1])
    px, py = int(px), int(py)
    size_px2 = int(size_m / MAP_RES)

    if flip:
        mapc = np.flipud(map).copy()
    else:
        mapc = map.copy()

    mapc, R = rotate_image(mapc, pose[2] * 180 / np.pi, center=(px, py))

    map_slice = mapc[py-size_px2:py+size_px2, px-size_px2:px+size_px2]

    if (map_slice.shape[1] < size_px2*2) or (map_slice.shape[0] < size_px2*2):
        # fill with invalid data
        canvas = np.zeros((size_px2*2, size_px2*2, 3), dtype=np.uint8)
        canvas[:map_slice.shape[0], :map_slice.shape[1]] = map_slice
        map_slice = canvas 
    
    if color is not None:
        map_slice = cv.cvtColor(map_slice, cv.COLOR_RGB2GRAY)
        map_slice[map_slice != color] = 1
        map_slice[map_slice == color] = 0

    origin = [px, py]

    return map_slice, origin, R


def get_last_msgs(msgs, current_time, N):
    return [m for t, m in reversed(msgs) if t <= current_time][:N]


def rotate_pointcloud(pcl, theta_rad):
    # Rotation matrix around Z
    R = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad), 0],
        [np.sin(theta_rad),  np.cos(theta_rad), 0],
        [0,                  0,                 1]
    ])
    return pcl @ R.T  # Rotate each point



def voxelize_lidar(batched_pts, voxel_size=0.08, max_points=5120):
        """
        Args:
            points: (N, 4) array, where columns are [x, y, z, intensity]
            voxel_size: float
            max_points: int
        Returns:
            (max_points, 4) array with downsampled [x, y, z, intensity] values
        """
        process_lidar = []
        for points in batched_pts:
            coords = np.floor(points[:, :3] / voxel_size).astype(np.int32)
            _, inv, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)

            # Sum xyz and intensity by voxel
            # xyz_intensity = np.concatenate([points[:, :3], points[:, 3]], axis=1)
            sums = np.zeros((counts.shape[0], 4), dtype=np.float32)
            np.add.at(sums, inv, points[:, :4])

            # Divide by counts to get mean per voxel
            means = sums / counts[:, None]

            N = means.shape[0]
            if N > max_points:
                indices = np.random.choice(N, max_points, replace=False)
                means = means[indices]
            elif N < max_points:
                pad = np.zeros((max_points - N, 4), dtype=np.float32)
                means = np.concatenate((means, pad), axis=0)
            
            process_lidar.append(means)

        return np.array(process_lidar)


def process_lidar(msgs, crop_fov=200):
    
    points = []
    for pcl_msg in msgs:
        pcl = np.array(list(pc2.read_points(pcl_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)))
        
        if crop_fov != -1:
            azimuth = np.arctan2(pcl[:, 1], pcl[:, 0])  # y, x
            ranges = np.linalg.norm(pcl, axis=1)

            mask = (azimuth >= np.radians(-crop_fov / 2)) & \
                    (azimuth <= np.radians(crop_fov / 2)) & \
                    (ranges >= 1.2)
            
            pcl = pcl[mask]

        pcl[:, :3] = rotate_pointcloud(pcl[:, :3], np.pi/2) # rotate only points
        pcl = pcl[:, [1, 0, 2, 3]]

        points.append(pcl)

    return points


def label_pointclouds(pcl, semantic_map, output_size=(600, 600), resolution=0.1):
    class_colors = {
        0: [0, 255, 0],     # Background (black)
        1: [255, 0, 0],     # Class 1 (red)
        # Add more classes as needed
    }
    
    # Convert point cloud to pixel coordinates
    bev_width, bev_height = output_size
    pcl_img = np.clip(pcl / resolution + np.array([bev_width/2, bev_height/2, 0]), 0, max(bev_width, bev_height)-1).astype(np.int64)
    pcl_x = pcl_img[:, 1]  # BEV image x-coordinate (matrix column)
    pcl_y = pcl_img[:, 0]  # BEV image y-coordinate (matrix row)
    
    # Filter points within image bounds
    mask = (pcl_x >= 0) & (pcl_x < bev_width) & \
           (pcl_y >= 0) & (pcl_y < bev_height)
    pcl_x = pcl_x[mask]
    pcl_y = pcl_y[mask]
    filtered_pcl = pcl[mask]
    
    # Create colored BEV image
    bev_image = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
    
    # Get class indices for each point
    point_classes = semantic_map[pcl_y, pcl_x]
    
    for (class_idx, color) in class_colors.items():
        class_mask = point_classes == class_idx
        bev_image[pcl_y[class_mask], pcl_x[class_mask]] = color
    
    # Create colored point cloud for PCD export
    # colored_pcl = np.zeros((filtered_pcl.shape[0], 6))  # x,y,z,r,g,b
    # colored_pcl[:, :3] = filtered_pcl
    pcl_labels = np.zeros(filtered_pcl.shape[0])  
    for (class_idx, color) in class_colors.items():
        class_mask = point_classes == class_idx
        pcl_labels[class_mask] = class_idx
        pcl_labels[filtered_pcl[:, 2] > .8] = 1
    
    pcl_labels = pcl_labels.reshape(-1, 1)  # Ensure it's a column vector

    return bev_image, pcl_labels


def write_pkl(data, path, filename):

    if os.path.exists(path) is False:
        os.makedirs(path)

    with open(os.path.join(path, filename), 'wb') as file:
        pickle.dump(data, file)


def main(start_index_data=0):
    # Load messages from the bag
    global_map = cv.imread(MAP_PATH)
    global_map = cv.resize(global_map, (int(global_map.shape[1] * 0.5), int(global_map.shape[0] * 0.5)))
    
    for BAG in BAG_FILES:    
        scan_msgs = []
        amcl_msgs = []
        data_dict = {}

        with rosbag.Bag(BAG, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[scan_topic, amcl_topic]):
                timestamp = t.to_sec()
                if topic == scan_topic:
                    scan_msgs.append((timestamp, msg))
                elif topic == amcl_topic:
                    amcl_msgs.append((timestamp, msg))
        bag.close()

        indices = np.linspace(10, len(amcl_msgs) - 50, 100, dtype=np.int64)


        for save_id, i in enumerate(indices):
            local_maps = []
            poses = []
            amcl_times = []
            scans = []
            scans_dn = []
            error = False

            for k in range(N_LIDAR):
                amcl_time, amcl_msg = amcl_msgs[i-k]
                # Get the local map
                q = amcl_msg.pose.pose.orientation
                _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
                pose = [amcl_msg.pose.pose.position.x, amcl_msg.pose.pose.position.y, yaw - np.pi/2]

                local_map, origin, _ = get_local_map(global_map, pose, MAP_ORIGIN, MAP_RES, color=81)

                try:
                    scan_history = process_lidar(get_last_msgs(scan_msgs, amcl_time, 1))[0]
                    _, labels = label_pointclouds(scan_history[:, :3], local_map, output_size=(local_map.shape[1], local_map.shape[0]), resolution=MAP_RES)
                    scans.append(np.concatenate([scan_history, labels], axis=1))

                    scan_dn = voxelize_lidar([scan_history], voxel_size=0.08, max_points=5120)[0]
                    _, labels = label_pointclouds(scan_dn[:, :3], local_map, output_size=(local_map.shape[1], local_map.shape[0]), resolution=MAP_RES)
                    scans_dn.append(np.concatenate([scan_dn, labels], axis=1))

                    local_maps.append(local_map)
                    poses.append(pose)
                    amcl_times.append(amcl_time)

                except Exception as e:
                    save_id += start_index_data - 1
                    print(f"Error processing message {i}: {e}")
                    error = True
                    break
            
            if not error:
                data_dict.update({
                    "lidar": scans,
                    "lidar_dn": scans_dn,
                    "poses": poses,
                    "local_map": local_maps[0],
                    "time": amcl_times,
                })
                save_id += start_index_data
                write_pkl(data_dict, LOCAL_PATHS_DIR, f"{save_id}_0.pkl")
        
        start_index_data = save_id + 1
        print(f"Processed {BAG} - saved {save_id} files.")


if __name__ == "__main__":
    main(start_index_data=600)