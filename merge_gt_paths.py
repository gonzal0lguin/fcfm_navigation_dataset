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


GT_PATHS_FILE = '/home/tesistas/Desktop/GONZALO/gnd_dataset/local_map_files_120/bb/djisktra_paths.pkl'
DATA_FILES = '/home/tesistas/Desktop/GONZALO/gnd_dataset/local_map_files_120/bb/'
IMGS_FILES = '/home/tesistas/Desktop/GONZALO/gnd_dataset/local_map_files_120/aa/'
DATA_PKL = '/home/tesistas/Desktop/GONZALO/gnd_dataset/local_map_files_120/data_fcfm_only.pkl'


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

    paths = [global_to_local(path, pose)[:, ::-1] for path in paths]
    paths.append(path_real)

    return paths


def expand_data_pkl(init, end):
    file_path = os.path.join(DATA_PKL)
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        print(f"File: {file_path} does not exits. Creating...")
        data = dict()
        data['ids'] = [] 
        data['root'] = ('/home/jing/Documents/gn/database/datasets/local_map_files_120/paths', '/home/jing/Documents/gn/database/datasets/local_map_files_120/planning')

    for i in range(init, end+1):
        data['ids'].append((f'{i}_0.pkl', f'{i}.png'))
        data['ids'].append((f'{i}_1.pkl', f'{i}.png'))

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
        gt_paths = [gt_path[:, ::-1] for gt_path in gt_paths]

        img = cv.imread(os.path.join(IMGS_FILES, f"{i}.png"), cv.IMREAD_GRAYSCALE)

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

        all_paths = process_paths(all_paths_data, path_real=data['path'], pose=data['pose'])

        # Insert the new 'all_paths' entry
        data['all_paths'] = all_paths
        data['imu'] = []
        data['scan'] = np.zeros((1, 3))
        data['lidar_array'] = []
        data['lidar_dn'] = _process_lidar(data['lidar']) #fix_lidar2(data)
        

        # Save back the updated data
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Done processing file {filename}")

        file_path = os.path.join(DATA_FILES, filename.replace('_0', '_1') + '.pkl')

        # Load the existing data
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        all_paths = process_paths(all_paths_data, path_real=data['path'], pose=data['pose'])

        # Insert the new 'all_paths' entry
        data['all_paths'] = all_paths
        data['imu'] = []
        data['scan'] = np.zeros((2, 3))
        data['lidar_array'] = []
        data['lidar_dn'] = _process_lidar(data['lidar']) #fix_lidar2(data)


        # Save back the updated data
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Done processing file {filename.replace('_0', '_1')}")


    print("Done")

    # expand data
    # expand_data_pkl(0, 1114)

    # write the gt paths into the images.
    write_gt_trajectories_posterior(0, 1115)

