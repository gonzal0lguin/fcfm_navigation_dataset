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

from process_bags import global_to_local, draw_path_on_map
from expand_paths_data import filter_unique_trajectories

VERSION = 'v2.3'
# INPUT_VERSION = 'v2.1'
GT_PATHS_FILE = f"/home/tesistas/Desktop/GONZALO/datasets/gnd_dataset/local_map_files_120/{VERSION}/djikstra.pkl"
DATA_FILES = f'/home/tesistas/Desktop/GONZALO/datasets/gnd_dataset/local_map_files_120/{VERSION}/data/'
IMGS_FILES   = f'/home/tesistas/Desktop/GONZALO/datasets/gnd_dataset/local_map_files_120/{VERSION}/maps/'
DATA_PKL   = f'/home/tesistas/Desktop/GONZALO/datasets/gnd_dataset/local_map_files_120/fcfm_{VERSION}.pkl'


def _process_lidar(batched_pts, voxel_size=0.08, max_points=5120):
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


def process_paths(paths, path_real, pose):

    paths = np.array([global_to_local(path, pose) for path in paths])
    # paths.append(path_real)
    
    vx, vy, ax, ay = [], [], [], []
    for path in paths:
        vx.append(np.gradient(path[:, 0], 1))
        vy.append(np.gradient(path[:, 1], 1))
        ax.append(np.gradient(vx[-1], 1))
        ay.append(np.gradient(vy[-1], 1))

    if len(vx) >= 1:    
        vx = np.array(vx)[:, 1:-2]
        vy = np.array(vy)[:, 1:-2]
        ax = np.array(ax)[:, 1:-2]
        ay = np.array(ay)[:, 1:-2]

        pathf = np.concatenate([paths[:, 1:-2], vx[:, :, np.newaxis], vy[:, :, np.newaxis], ax[:, :, np.newaxis], ay[:, :, np.newaxis]], axis=-1) 
        pathf = np.concatenate([path_real[np.newaxis, :-1, :], pathf], axis=0)   
    else:
        pathf = path_real[np.newaxis]

    return pathf[:, :12, :]


def expand_data_pkl(init, end):
    file_path = os.path.join(DATA_PKL)
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        print(f"File: {file_path} does not exits. Creating...")
        data = dict()
        data['ids'] = [] 
        data['root'] = (f'{VERSION}/maps', f'{VERSION}/data')

    for i in range(init, end+1):
        data['ids'].append((f'{i}_0.pkl', f'{i}.png'))

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def fix_lidar(data):
    lidar = []
    for pts in data['lidar']:
        azimuth = np.arctan2(pts[:, 1], pts[:, 0])  # y, x
        ranges = np.linalg.norm(pts, axis=1)

        mask = (azimuth >= np.radians(-100)) & \
                (azimuth <= np.radians(100)) & \
                (ranges >= 1)

        lidar.append(pts[mask])
    
    return lidar


def fix_lidar2(data):
    return _process_lidar(data["lidar"])


def write_gt_trajectories_posterior(start, end):
    for i in range(start, end+1):
        file_path = os.path.join(DATA_FILES, f"{i}_0.pkl")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        gt_paths = data['all_paths']
        # gt_paths = [gt_path[:, ::-1] for gt_path in gt_paths]

        img = cv.imread(os.path.join(IMGS_FILES, f"{i}.png"))#, cv.IMREAD_GRAYSCALE)

        img = draw_path_on_map(img, gt_paths, 0, 0.1, color=(0, 255, 255), thickness=1)

        cv.imwrite(os.path.join(IMGS_FILES, f"{i}.png"), img)



if __name__ == "__main__":
    with open(GT_PATHS_FILE, 'rb') as file:
        gt_paths = pickle.load(file)

    for filename, all_paths_data in gt_paths.items():

        file_path = os.path.join(DATA_FILES, filename + '.pkl')

        # Load the existing data
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        if data['path'].shape[0] != 12:
            all_paths = process_paths(all_paths_data, path_real=data['path'], pose=data['pose'])
            all_paths = filter_unique_trajectories(all_paths)[0]
            # Insert the new 'all_paths' entry
            data['all_paths'] = all_paths

        else:
            data['all_paths'] = data['path'][np.newaxis][:12, :]

        # data['imu'] = []
        # data['scan'] = np.zeros((1, 3))
        # data['lidar_array'] = []
        # data['lidar_dn'] = _process_lidar(data['lidar']) #fix_lidar2(data)
        

        # Save back the updated data
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Done processing file {filename}")



    # print("Done")

    # expand data
    # expand_data_pkl(0, 1453)

    # write the gt paths into the images.
    # write_gt_trajectories_posterior(0, 1452)

