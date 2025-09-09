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


DATASET_VERSION = 'v3.0'

BAG_FILES = [
        '/home/gonz/Desktop/bags/teleop2/sec_a_1_2025-09-04-21-28-30.bag' ,
        '/home/gonz/Desktop/bags/teleop2/sec_a_2_2025-09-04-21-30-40.bag' ,
        '/home/gonz/Desktop/bags/teleop2/sec_a_3_2025-09-04-21-32-49.bag' ,
        '/home/gonz/Desktop/bags/teleop2/sec_a_4_2025-09-04-21-36-30.bag' ,
        '/home/gonz/Desktop/bags/teleop2/sec_a_5_2025-09-04-21-38-38.bag' ,
        '/home/gonz/Desktop/bags/teleop2/sec_a_6_2025-09-04-21-41-31.bag' ,
        '/home/gonz/Desktop/bags/teleop2/sec_a_7_2025-09-04-21-42-56.bag' ,
        '/home/gonz/Desktop/bags/teleop2/sec_a_8_2025-09-04-21-44-42.bag' ,
        '/home/gonz/Desktop/bags/teleop2/sec_a_9_2025-09-04-21-50-25.bag' ,
        '/home/gonz/Desktop/bags/teleop2/sec_a_10_2025-09-04-21-52-48.bag', 
        '/home/gonz/Desktop/bags/teleop2/sec_a_11_2025-09-04-21-57-30.bag', 
        '/home/gonz/Desktop/bags/teleop2/sec_a_12_2025-09-04-21-59-29.bag', 
        '/home/gonz/Desktop/bags/teleop2/sec_a_13_2025-09-04-22-01-44.bag', 
        '/home/gonz/Desktop/bags/teleop2/sec_a_14_2025-09-04-22-04-55.bag',
        ]

MAP_PATH = "/home/gonz/Desktop/THESIS/code/global-planning/fcfm_navigation_dataset/ros_map_utils/maps/electrica.png"
DATA_DIR = "/home/gonz/Desktop/THESIS/code/global-planning/datasets/gnd_dataset/local_map_files_120"

LOCAL_MAPS_DIR  = os.path.join(DATA_DIR, DATASET_VERSION, "maps")
LOCAL_PATHS_DIR = os.path.join(DATA_DIR, DATASET_VERSION, "data")


MAP_RES = 0.1
MAP_ORIGIN = [-34.8, -81.2]  # electrica
# MAP_ORIGIN = [-57.2, -90.8]   # cancha

N_VEL = 20  # Number of odom messages
N_LIDAR = 3  # Number of lidar messages
N_PREV = 10
N_WAYPOINTS = 13

odom_topic = "/panther/odometry/filtered"
scan_topic = "/repub/ouster/points"
amcl_topic = "/amcl_pose"
img_topic  = "/repub/camera/image_raw"
obs_topic  = "/navae/raw_observation"


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


def get_path_length_interval(odometry_xy, start, lenght=12., N_wpts=15, reversed=False):
    if not reversed:
        xsq = (odometry_xy[start+1:, 0] - odometry_xy[start:-1, 0]) ** 2
        ysq = (odometry_xy[start+1:, 1] - odometry_xy[start:-1, 1]) ** 2
        distances = np.cumsum(np.sqrt(xsq + ysq), axis=0)
        stop_idx = np.where(distances >= lenght)[0]
        
        if len(stop_idx) == 0:
            # stop_idx = len(odometry_xy) - 1
            return None
        else:
            stop_idx = stop_idx[0] + start
        
        ids = np.linspace(start, stop_idx, N_wpts, dtype=np.int64) # inlcude last point
    
    else: # search PASt waypoints
        xsq = (odometry_xy[1:start, 0] - odometry_xy[:start-1, 0]) ** 2
        ysq = (odometry_xy[1:start, 1] - odometry_xy[:start-1, 1]) ** 2
        distances = np.cumsum(np.sqrt(xsq + ysq)[::-1], axis=0)
        stop_idx = np.where(distances >= lenght)[0]
        
        if len(stop_idx) == 0:
            stop_idx = 0
            # return None
        else:
            stop_idx = len(distances) - stop_idx[0] 

        ids = np.linspace(stop_idx, start, N_wpts, dtype=np.int64) # inlcude last point

    return odometry_xy[ids]



def get_path_time_interval(odometry_xy, times, start, duration_secs=10.0, N_wpts=15, reversed=False):

    if times is None:
        raise ValueError("You must provide the `times` array for time-based sampling.")
    
    if reversed:
        start_time = times[start]
        target_times = start_time - np.linspace(0, duration_secs, N_wpts)
        # Ensure we don't go below zero index
        valid_mask = target_times >= times[0]
    else:
        start_time = times[start]
        target_times = start_time + np.linspace(0, duration_secs, N_wpts)
        # Ensure we don't go beyond the last index
        valid_mask = target_times <= times[-1]

    target_times = target_times[valid_mask]

    # Interpolate x and y separately at the desired time points
    x_interp = np.interp(target_times, times, odometry_xy[:, 0])
    y_interp = np.interp(target_times, times, odometry_xy[:, 1])
    wpts_xy = np.stack((x_interp, y_interp), axis=-1)

    return wpts_xy, target_times



def make_paths(origin, gt_lst, gt_times, start, duration, nwpts, reversed=False):
    timedelta = (duration + 1) / (nwpts + int(reversed))
    time_path, times = get_path_time_interval(gt_lst, gt_times, start, duration_secs=duration, N_wpts=nwpts, reversed=reversed)
    time_path_local = global_to_local(time_path, origin) 
    if reversed:
        time_path_local = time_path_local[::-1] # leave as [t-n, ..., t-1, t]
    # velocities = (time_path_local[1:] - time_path_local[:-1]) / timedelta
    vx = np.gradient(time_path_local[:, 0], timedelta)
    vy = np.gradient(time_path_local[:, 1], timedelta)
    ax = np.gradient(vx, timedelta)
    ay = np.gradient(vy, timedelta)

    pathf = np.concatenate([time_path_local, vx[:, np.newaxis], vy[:, np.newaxis], ax[:, np.newaxis], ay[:, np.newaxis]], axis=1)

    # print(times[1] - times[0], timedelta, pathf.shape)
    
    return pathf




def rotate_image(image, angle, center=None):
  if center is None:
    center = tuple(np.array(image.shape[1::-1]) / 2)
  
  rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result, rot_mat



def get_local_map(map, pose, map_origin, map_res, size_m=30, flip=True, color=None):
    px, py = world_to_map(map_origin, map_res, pose[0], pose[1])
    px, py = int(px), int(py)
    size_px2 = int(size_m / MAP_RES)

    if flip:
        mapc = np.flipud(map).copy()
    else:
        mapc = map.copy()
    mapc, R = rotate_image(mapc, pose[2] * 180 / np.pi-90, center=(px, py))
    map_slice = mapc[py-size_px2:py+size_px2, px-size_px2:px+size_px2]
    # import matplotlib.pyplot as plt
    # print(px, py)
    # plt.imshow(map_slice)
    # plt.show()

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
    map_slice = np.fliplr(map_slice)
    return map_slice, origin, R


def draw_path_on_map(map, path_local_list, origin, map_res, size_m=30, color=(1, 0, 0), thickness=2):
    """
    Draw a path on the map.
    :param map: The map to draw on.
    :param path: List of (x, y) tuples representing the path.
    :param color: Color of the path.
    :param thickness: Thickness of the path.
    :return: Map with the path drawn on it.
    """

    map_cpy = map.copy()    
    if map.ndim == 2:
        map_cpy = cv.cvtColor(map_cpy, cv.COLOR_GRAY2RGB)

    for path_local in path_local_list:
        for (x, y) in path_local[:, :2]:
            pxi = int(y / map_res + int(size_m / map_res))
            pyi = int(x / map_res + int(size_m / map_res))
            map_cpy = cv.circle(map_cpy, (pxi, pyi), 2, color, -1)

    return map_cpy


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


def process_goal(msgs):
    pass


def process_vel(msgs):
    odom = [[o.twist.twist.linear.x, o.twist.twist.angular.z] for o in msgs[::2]] # use 10 out of the 20 messages sampled at 20Hz to get 10Hz 
    return odom


def process_lidar(msgs, crop_fov=200):
    
    points = []
    for pcl_msg in reversed(msgs): # msgs come in as [t, t-1, ..., t-n] and we need [t-n, ..., t-1, t]
        pcl = np.array(list(pc2.read_points(pcl_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)))
        
        if crop_fov != -1:
            azimuth = np.arctan2(pcl[:, 1], pcl[:, 0])  # y, x
            ranges = np.linalg.norm(pcl, axis=1)

            mask = (azimuth >= np.radians(-crop_fov / 2)) & \
                    (azimuth <= np.radians(crop_fov / 2)) & \
                    (ranges >= 1)
            
            pcl = pcl[mask]
        


        points.append(pcl)

    return points


def process_img(msgs, bridge):
    images = [bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") for msg in msgs]
    return images
    

def voxelize_lidar(batched_pts, voxel_size=0.08, max_points=5120):
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

def select_indices_t_apart(times, T):
    times = np.asarray(times)
    indices = [0]
    current_time = times[0]

    for i in range(1, len(times)):
        if times[i] - current_time >= T:
            indices.append(i)
            current_time = times[i]

    return indices


def show(img, title="Image"):
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def write_pkl(data, path, filename):

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, filename), 'wb') as file:
        pickle.dump(data, file)


def main(start_index_data=0):
    # Load messages from the bag
    global_map = cv.imread(MAP_PATH)
    global_map = cv.resize(global_map, (int(global_map.shape[1] * 0.5), int(global_map.shape[0] * 0.5)))
    
    for BAG_FILE in BAG_FILES:    
        data_dict = {}

        goal_msgs = []
        scan_msgs = []
        past_tr_msgs = []
        amcl_msgs = []

        with rosbag.Bag(BAG_FILE, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[obs_topic, amcl_topic, '/map']):
                timestamp = t.to_sec()
                if topic == amcl_topic:
                    amcl_msgs.append((timestamp, msg))
                elif topic == obs_topic:
                    scan_msgs.append((timestamp, msg.lidar))
                    goal_msgs.append((timestamp, msg.goal))
                    past_tr_msgs.append((timestamp, msg.past_path))
                elif topic == '/map':
                    map_info = msg.info

        bag.close()

        amcl_msgs_lst = np.array([[msg.pose.pose.position.x, msg.pose.pose.position.y] for (t, msg) in amcl_msgs])
        amcl_time_lst = np.array([t for (t, msg) in amcl_msgs])

        print(len(amcl_msgs))

        indices = select_indices_t_apart(amcl_time_lst, T=0.6)[1:]
        #np.linspace(10, len(amcl_msgs) - 50, 100, dtype=np.int64)

        for save_id, i in enumerate(indices):
            amcl_time, amcl_msg = amcl_msgs[i]
            
            # Get the local map
            q = amcl_msg.pose.pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            pose = [amcl_msg.pose.pose.position.x, amcl_msg.pose.pose.position.y, yaw]

            local_map, origin, _ = get_local_map(global_map, pose, MAP_ORIGIN, MAP_RES, color=81)

            try:
                sampled_path_local = make_paths(pose, amcl_msgs_lst, amcl_time_lst, i, duration=N_WAYPOINTS, nwpts=N_WAYPOINTS+1) # this yields 12 wpts asumming temp distance is 1 s
                
                past_tr = get_last_msgs(past_tr_msgs, amcl_time, 1)
                if len(past_tr) == 0:
                    print("No past path message for amcl time")
                    start_index_data -= 1
                    continue

                past_tr = np.array([[p.pose.position.x, p.pose.position.y] for p in past_tr[0].poses])
                past_tr = -past_tr
                past_tr = past_tr - past_tr[-1] + amcl_msgs_lst[i][:2]
                pose[-1] += np.pi/2
                previous_path_local = global_to_local(past_tr, pose)
                vx = np.gradient(previous_path_local[:, 0], .5)
                vy = np.gradient(previous_path_local[:, 1], .5)
                previous_path_local = np.concatenate((previous_path_local[:, :2], vx[:, np.newaxis], vy[:, np.newaxis]), axis=-1)

                if len(sampled_path_local) < N_WAYPOINTS:
                    print("Future path not long enough")
                    save_id += start_index_data - 1
                    break
                
                elif len(previous_path_local) < N_PREV:
                    print("Previous path not long enough")
                    start_index_data -= 1
                    continue
                sampled_path_local = sampled_path_local[1:]

                local_map_drawn = draw_path_on_map(local_map, [sampled_path_local[:, :2]], origin, MAP_RES, color=(0, 1, 1), thickness=1)
                local_map_drawn = draw_path_on_map(local_map_drawn, [previous_path_local[:, :2]], origin, MAP_RES, color=(1, 0, 1), thickness=1)

                scan_history = get_last_msgs(scan_msgs, amcl_time, 1)
                scan_dn = np.array(scan_history[0].data).reshape(3, 2560, 4)  # 3 scans, 2560 points each, (x,y,z,intensity)

                if len(scan_dn) < N_LIDAR:
                    print("not enough lidar samples")
                    start_index_data -= 1
                    continue
                
                global_goal = np.array(goal_msgs[1][-1].data)

                data_dict.update({
                    "lidar_dn": scan_dn,
                    "pose": pose,
                    "goal": global_goal,
                    "local_map": local_map,
                    "path": sampled_path_local, # this includes vel
                    "previous_path": previous_path_local, # this includes vel
                    "time": amcl_time,
                })

                # # Save the data
                save_id += start_index_data

                write_pkl(data_dict, LOCAL_PATHS_DIR, f"{save_id}_0.pkl")
                # # write_pkl(data_dict, LOCAL_PATHS_DIR, f"{save_id}_1.pkl")
                print(save_id)
                path = os.path.join(LOCAL_MAPS_DIR, f"{save_id}.png")
                if not os.path.exists(LOCAL_MAPS_DIR):
                    os.makedirs(LOCAL_MAPS_DIR)
                cv.imwrite(path, local_map_drawn*255)

            except Exception as e:
                save_id += start_index_data - 1
                print(f"Error processing message {i}: {e}")
                break

        start_index_data = save_id + 1


if __name__ == "__main__":
    main(start_index_data=0)    