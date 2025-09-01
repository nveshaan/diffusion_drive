import numpy as np
import h5py
from sklearn.cluster import k_means
import matplotlib.pyplot as plt

def global_to_ego_2d(global_points, ego_position, ego_yaw_deg):
    """
    Convert global 2D/3D points to ego-centric coordinates.
    Ego frame: +x = forward, +y = right, -y = left.
    Ensures that points in front always have x > 0, regardless of global direction.

    Args:
        global_points: (N, 3) or (N, 2)
        ego_position: (3,) or (2,) — vehicle position in global coordinates
        ego_yaw_rad: float — heading in radians

    Returns:
        (N, 2) — ego-frame (x, y): x > 0 = ahead, y > 0 = right
    """
    global_xy = global_points[:, :2]
    ego_xy = np.array(ego_position[:2])
    ego_yaw_rad = np.deg2rad(ego_yaw_deg)

    # Translate: center the frame at ego
    delta = global_xy - ego_xy

    # Rotation: global → ego with Y flipped (so +Y is right)
    cos_yaw = np.cos(ego_yaw_rad)
    sin_yaw = np.sin(ego_yaw_rad)

    rotation_matrix = np.array([ 
        [ cos_yaw,  sin_yaw],
        [ sin_yaw, -cos_yaw],
    ])

    return np.matmul(delta, rotation_matrix.T)  # shape: (N, 2)

file_path = '/Volumes/New Volume/marathon.hdf5'
obs_horizon = 8
obs_stride = 5
skip = 0

index_map = []
total_frames = 500
max_index = total_frames - (obs_horizon - 1) * obs_stride

for i in range(skip, max_index):
    index_map.append([i + j * obs_stride for j in range(obs_horizon)])

trajectories = []
with h5py.File(file_path, 'r') as f:
    for run_key in f['runs']:
        run = f[f'runs/{run_key}']
        for index in index_map:
            global_trajectory = np.array([run['vehicles']['ego']['location'][idx] for idx in index])
            trajectory = global_to_ego_2d(global_trajectory, ego_position=global_trajectory[0], ego_yaw_deg=run['vehicles']['ego']['rotation'][index[0]][1])
            trajectories.append(np.array(trajectory).reshape(-1))  # Flatten to 1D for k-means

anchors = k_means(np.array(trajectories), 20)
anchors = np.reshape(np.array(anchors[0]), (-1, 8, 2))

np.save('utils/kmeans_carla_traj_20.npy', anchors)

navsim_trajectories = np.load('utils/kmeans_navsim_traj_20.npy')

plt.figure(figsize=(10, 10))
for i, navsim_trajectory in enumerate(navsim_trajectories):
    plt.plot(navsim_trajectory[:, 0], navsim_trajectory[:, 1], color='blue', alpha=0.3, label='NavSim Anchors' if i == 0 else "_nolegend_")
for j, anchor in enumerate(anchors):
    plt.plot(anchor[:, 0], anchor[:, 1], color='red', alpha=0.3, label='CARLA Anchors' if j == 0 else "_nolegend_")
plt.xlabel('Ego X')
plt.ylabel('Ego Y')
plt.title('NavSim Trajectories and K-Means Anchors')
plt.legend(loc='upper right')
plt.grid()
plt.axis('equal')
plt.show()