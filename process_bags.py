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
    '/home/gonz/Desktop/bags/sec_a_4_2025-05-23-21-50-01.bag'
            ]

MAP_PATH = "/home/gonz/Desktop/THESIS/code/global-planning/gnd_dataset/ele2/ele2.png"

DATA_DIR = "/home/gonz/Desktop/THESIS/code/global-planning/gnd_dataset/local_map_files_120"
LOCAL_MAPS_DIR  = os.path.join(DATA_DIR, "planning")
LOCAL_PATHS_DIR = os.path.join(DATA_DIR, "paths")



MAP_RES = 0.1
MAP_ORIGIN = [-34.8, -81.2]  # Origin of the map in world coordinates
N_VEL = 20  # Number of odom messages
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
    
    if color is not None:
        map_slice = cv.cvtColor(map_slice, cv.COLOR_RGB2GRAY)
        map_slice[map_slice != color] = 1
        map_slice[map_slice == color] = 0

    origin = [px, py]

    return map_slice, origin, R


def draw_path_on_map(map, path_local, origin, map_res, size_m=30, color=(1, 0, 0), thickness=2):
    """
    Draw a path on the map.
    :param map: The map to draw on.
    :param path: List of (x, y) tuples representing the path.
    :param color: Color of the path.
    :param thickness: Thickness of the path.
    :return: Map with the path drawn on it.
    """

    map_cpy = map.copy()    
    map_cpy = cv.cvtColor(map_cpy, cv.COLOR_GRAY2RGB)

    for (x, y) in path_local:
        pxi = int(x / map_res + int(size_m / map_res))
        pyi = int(y / map_res + int(size_m / map_res))
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


def process_vel(msgs):
    odom = [[o.twist.twist.linear.x, o.twist.twist.angular.z] for o in msgs[::2]] # use 10 out of the 20 messages sampled at 20Hz to get 10Hz 
    return odom


def process_lidar(msgs):
    
    points = []
    for pcl_msg in msgs:
        pcl = np.array(list(pc2.read_points(pcl_msg, field_names=("x", "y", "z"), skip_nans=True)))
        pcl = rotate_pointcloud(pcl, np.pi/2)
        pcl = pcl[:, [1, 0, 2]]

        points.append(pcl)

    return np.array(points)


def process_img(msgs, bridge):
    images = [bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") for msg in msgs]
    return images
    

def show(img, title="Image"):
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def write_pkl(data, path, filename):

    with open(os.path.join(path, filename), 'wb') as file:
        pickle.dump(data, file)


def main(start_index_data=0):
    # Load messages from the bag
    global_map = cv.imread(MAP_PATH)
    global_map = cv.resize(global_map, (int(global_map.shape[1] * 0.5), int(global_map.shape[0] * 0.5)))
    
    for BAG in BAG_FILES:    
        odom_msgs = []
        scan_msgs = []
        amcl_msgs = []
        img_msgs  = []
        data_dict = {}

        with rosbag.Bag(BAG, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=["map", odom_topic, scan_topic, amcl_topic, img_topic]):
                timestamp = t.to_sec()
                if topic == odom_topic:
                    odom_msgs.append((timestamp, msg))
                elif topic == scan_topic:
                    scan_msgs.append((timestamp, msg))
                elif topic == amcl_topic:
                    amcl_msgs.append((timestamp, msg))
                elif topic == img_topic:
                    img_msgs.append((timestamp, msg))
        bag.close()

        amcl_msgs_lst = np.array([[msg.pose.pose.position.x, msg.pose.pose.position.y] for (t, msg) in amcl_msgs])

        print(len(amcl_msgs))

        indices = np.linspace(10, len(amcl_msgs) - 100, 120, dtype=int)

        bridge = CvBridge()

        for save_id, i in enumerate(indices):
            amcl_time, amcl_msg = amcl_msgs[i]
            
            # Get the local map
            q = amcl_msg.pose.pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            pose = [amcl_msg.pose.pose.position.x, amcl_msg.pose.pose.position.y, yaw - np.pi/2]

            local_map, origin, _ = get_local_map(global_map, pose, MAP_ORIGIN, MAP_RES, color=81)

            # show(local_map, title=f"Local Map at {i}")

            try:
                sampled_path = get_path_lenght_interval(amcl_msgs_lst, i, lenght=15., N_wpts=15)

                sampled_path_local = global_to_local(sampled_path, pose)

                local_map_drawn = draw_path_on_map(local_map, sampled_path_local, origin, MAP_RES, color=(0, 1, 1), thickness=1)

                odom_history = process_vel(get_last_msgs(odom_msgs, amcl_time, N_VEL))
                scan_history = process_lidar(get_last_msgs(scan_msgs, amcl_time, N_LIDAR))
                img_history  = process_img(get_last_msgs(img_msgs, amcl_time, 1), bridge)

                sampled_path_local = sampled_path_local[:, ::-1] # flip x and y idk y tf

                data_dict.update({
                    "vel": odom_history,
                    "lidar": scan_history,
                    "camera": img_history,
                    "pose": pose,
                    "local_map": local_map,
                    "path": sampled_path_local,
                    "time": amcl_time,
                })


                # Save the data
                save_id += start_index_data
                write_pkl(data_dict, LOCAL_PATHS_DIR, f"{save_id}_0.pkl")
                write_pkl(data_dict, LOCAL_PATHS_DIR, f"{save_id}_1.pkl")

                path = os.path.join(LOCAL_MAPS_DIR, f"{save_id}.png")
                cv.imwrite(path, local_map_drawn*255)

            except Exception as e:
                print(f"Error processing message {i}: {e}")
                break


if __name__ == "__main__":
    main(start_index_data=1090)