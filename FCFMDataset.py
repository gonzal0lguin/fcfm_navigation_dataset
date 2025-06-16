import os
import numpy as np
from torch.utils.data import Dataset
import pickle
from random import shuffle

class FCFMDataset(Dataset):
    def __init__(self, root, num_point=2048*3, transform=None, dn_cloud=False):
        """
        Args:
            data_paths: list of npy files. Each file is expected to have shape (N, 5) -> [x, y, z, intensity, label]
            num_point: number of points per sample (e.g., 4096)
            transform: optional data augmentation
        """
        self.data_root = root
        self.num_point = num_point
        self.transform = transform

        self.cloud_name = 'lidar' if not dn_cloud else 'lidar_dn'

        self.data_paths = os.listdir(root)  # List of (N, 5) arrays

        # shuffle the data paths for randomness
        shuffle(self.data_paths)
        

    def _read_pkl(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
        

    def __len__(self):
        return len(self.data_paths)


    def __getitem__(self, idx):

        data = self._read_pkl(os.path.join(self.data_root, self.data_paths[idx]))

        cloud = np.stack(data[self.cloud_name]).reshape(-1, 5)
        xyz = cloud[:, :3]
        intensity = cloud[:, 3]
        labels = cloud[:, 4].astype(np.int64)

        print(cloud.shape)

        N = cloud.shape[0]
        if N >= self.num_point:
            idxs = np.random.choice(N, self.num_point, replace=False)
        else:
            idxs = np.random.choice(N, self.num_point, replace=True)

        selected_xyz = xyz[idxs]
        selected_intensity = intensity[idxs]
        selected_labels = labels[idxs]

        print(selected_xyz.shape, selected_intensity.shape, selected_labels.shape)

        # Normalize XYZ (optional)
        selected_xyz = selected_xyz - selected_xyz.mean(0, keepdims=True)

        # Combine features: [x, y, z, intensity]
        points = np.concatenate([selected_xyz, selected_intensity.reshape(-1, 1)], axis=1)  # shape (num_point, 4)

        if self.transform:
            points, selected_labels = self.transform(points, selected_labels)

        return points.astype(np.float32), selected_labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    a = FCFMDataset(root='/home/gonz/Desktop/THESIS/code/global-planning/gnd_dataset/local_map_files_120/cc/', 
                    num_point=2048, 
                    dn_cloud=True)
    
    print(len(a))
    print(a[0])

    plt.hist(a[0][0][:, 3])
    plt.show()