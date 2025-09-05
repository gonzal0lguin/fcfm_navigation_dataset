import os
import numpy as np
import pickle


VERSION = 'v2.0'
VERSION2 = 'v2.1'
DATA_FILES = f'/home/tesistas/Desktop/GONZALO/datasets/gnd_dataset/local_map_files_120/{VERSION}/data/'
OUTFILES   = f'/home/tesistas/Desktop/GONZALO/datasets/gnd_dataset/local_map_files_120/{VERSION2}/data/'
DATA_PKL   = f'/home/tesistas/Desktop/GONZALO/datasets/gnd_dataset/local_map_files_120/fcfm_{VERSION2}.pkl'

GENERATE_AS_NEW = False

def expand_data_pkl(ids):
    file_path = os.path.join(DATA_PKL)
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        print(f"File: {file_path} does not exits. Creating...")
        data = dict()
        data['ids'] = ids
        data['root'] = (f'{VERSION2}/maps', f'{VERSION2}/data')

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def compute_ade_fde(traj1, traj2):
    # traj1, traj2: Nx2
    ade = np.mean(np.linalg.norm(traj1 - traj2, axis=1))
    fde = np.linalg.norm(traj1[-1] - traj2[-1])
    return ade, fde

def filter_unique_trajectories(trajectories, ade_th=0.5, fde_th=0.5):
    K = trajectories.shape[0]
    keep = []
    for i in range(K):
        traj_i = trajectories[i]
        is_similar = False
        for j in keep:
            ade, fde = compute_ade_fde(traj_i, trajectories[j])
            if ade < ade_th and fde < fde_th:
                is_similar = True
                break
        if not is_similar:
            keep.append(i)
    
    filtered_trajectories = trajectories[keep]
    return filtered_trajectories, keep


def generate_data(files, target_dir):
    ids = []
    for j, file in enumerate(sorted(os.listdir(files))):
        with open(os.path.join(files, file), 'rb') as f:
            data = pickle.load(f)
        
        gt_paths = np.array(data['all_paths'])
        kept_trajs, _ = filter_unique_trajectories(gt_paths)
        # data.pop('lidar')
        # data.pop('camera') # this is to make the dataset smaller

        os.makedirs(target_dir, exist_ok=True)
        if GENERATE_AS_NEW:
            for i, traj in enumerate(kept_trajs):
                data['path'] = traj
                subfile = file[:-5]+f'{i}.pkl'
                # with open(os.path.join(target_dir, subfile), 'wb') as f:
                #     pickle.dump(data, f)
                
                ids.append((subfile, f'{subfile.split("_")[0]}.png'))

                print(f'Saved file: {subfile}')
        else:
            data['all_paths'] = kept_trajs
            with open(os.path.join(target_dir, file), 'wb') as f:
                pickle.dump(data, f)
            print('Updated file ', file)
            ids.append((file, f'{file.split("_")[0]}.png'))

    expand_data_pkl(ids)

    print("done!")



if __name__=="__main__":
    generate_data(DATA_FILES, OUTFILES)