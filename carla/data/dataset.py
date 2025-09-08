import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from math import pi
import cv2

from navsim.agents.transfuser.transfuser_config import TransfuserConfig

class SampleData(Dataset):
    def __init__(self, file_path, num_poses, stride, gap, skip=150):
        super().__init__()
        self.file_path = file_path
        self.file = None
        self.skip = skip
        self.num_poses = num_poses
        self.stride = stride
        self.feature_keys = ['image', 'laser', 'velocity', 'acceleration', 'command']
        self.target_keys = ['location']
        self.gap = gap
        self.index_map = []
        with h5py.File(self.file_path, 'r') as f:
            for run_key in f['runs']:
                run = f[f'runs/{run_key}']
                total_frames = len(run[f'vehicles/ego/control'])
                max_index = total_frames - (self.gap +
                    (self.num_poses - 1) * self.stride + 1)

                for i in range(self.skip, max_index):
                    self.index_map.append((run_key, i))

    def __len__(self) -> int:
        return len(self.index_map)
    
    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')
        run_key, start_idx = self.index_map[idx]
        try:
            result = self._load_sample(run_key, start_idx)
            return result
        except Exception as e:
            print(f"Error loading sample {idx} (run_key={run_key}, start_idx={start_idx}): {e}")
            raise

    def _load_sample(self, run_key, start_idx):
        run = self.file[f'runs/{run_key}']
        features, targets = {}, {}

        for key in self.feature_keys:
            data = run['vehicles/ego/'+key]
            idxs = [start_idx]
            if key == 'image':
                features.update({'camera_feature': torch.stack([torch.flip(torch.tensor(data[i], dtype=torch.float32)[:, :, :3], dims=[-1]).permute(2, 0, 1)
                                                    for i in idxs]).squeeze()})
            elif key == 'laser':
                features.update({'lidar_feature': torch.stack([self._get_lidar_feature(data[i]) for i in idxs]).squeeze()})
            elif key == 'velocity':
                velocity = np.array(data[idxs]).squeeze().norm()
            elif key == 'acceleration':
                acceleration = np.array(data[idxs]).squeeze().norm()
            elif key == 'command':
                command = np.array(data[idxs]).squeeze()

        features.update({'status_feature': torch.tensor([command, command, command, command, velocity, velocity, acceleration, acceleration], dtype=torch.float32).unsqueeze(0)})

        pred_start = start_idx + self.gap
        for key in self.target_keys:
            ego_data = run['vehicles/ego/'+key]
            idxs = [pred_start + i * self.stride for i in range(self.num_poses)]
            if key == 'location':
                targets.update({'trajectory': torch.tensor(self._global_to_ego_2d(np.array(ego_data[idxs]), np.array(ego_data[start_idx]), np.array(run['vehicles/ego/rotation'][start_idx][1])), dtype=torch.float32).squeeze()})
            else:
                vehicles = [v for v in run['vehicles'] if v != 'ego']
                vehicles_location = []
                vehicles_yaw = []
                vehicles_extent = []
                if len(vehicles) > 0:
                    for vehicle in vehicles:
                        vehicle_data = run['vehicles/'+vehicle+'/'+key]
                        vehicles_location.append(torch.tensor(self._global_to_ego_2d(np.array(vehicle_data[start_idx]), np.array(ego_data[start_idx]), np.array(run['vehicles/ego/rotation'][start_idx][1])), dtype=torch.float32).unsqueeze(0))
                        vehicles_yaw.append(torch.tensor(run['vehicles/'+vehicle+'/rotation'][start_idx][1], dtype=torch.float32).unsqueeze(0))
                        vehicles_extent.append(torch.tensor(run['vehicles/'+vehicle+'/extent'][start_idx], dtype=torch.float32).unsqueeze(0))

                walkers = [w for w in run['walkers']]
                walkers_location = []
                walkers_yaw = []
                walkers_extent = []
                if len(walkers) > 0:
                    for walker in walkers:
                        walker_data = run['walkers/'+walker+'/'+key]
                        walkers_location.append(torch.tensor(self._global_to_ego_2d(np.array(walker_data[start_idx]), np.array(ego_data[start_idx]), np.array(run['vehicles/ego/rotation'][start_idx][1])), dtype=torch.float32).unsqueeze(0))
                        walkers_yaw.append(torch.tensor(run['walkers/'+walker+'/rotation'][start_idx][1], dtype=torch.float32).unsqueeze(0))
                        walkers_extent.append(torch.tensor(run['walkers/'+walker+'/extent'][start_idx], dtype=torch.float32).unsqueeze(0))

                distances = np.linalg.norm(np.array(ego_data[start_idx][:2]) - np.array([loc.numpy()[:2] for loc in vehicles_location + walkers_location]), axis=1) if (len(vehicles_location) + len(walkers_location)) > 0 else np.array([])
                argsort = np.argsort(distances)[:30]
                agent_location = np.concat(vehicles_location + walkers_location, dim=0) if (len(vehicles_location) + len(walkers_location)) > 0 else torch.zeros((0, 3), dtype=torch.float32)
                agent_yaw = np.concat(vehicles_yaw + walkers_yaw, dim=0) if (len(vehicles_yaw) + len(walkers_yaw)) > 0 else torch.zeros((0,), dtype=torch.float32)
                agent_extent = np.concat(vehicles_extent + walkers_extent, dim=0) if (len(vehicles_extent) + len(walkers_extent)) > 0 else torch.zeros((0, 3), dtype=torch.float32)
                if len(argsort) > 0:
                    agent_location = agent_location[argsort]
                    agent_yaw = agent_yaw[argsort]
                    agent_extent = agent_extent[argsort]
                    agent_labels = torch.ones((30,), dtype=torch.int64)[argsort].unsqueeze(0)
                if len(argsort) < 30:
                    padding = 30 - len(argsort)
                    agent_location = torch.cat([agent_location, torch.zeros((padding, 3), dtype=torch.float32)], dim=0)
                    agent_yaw = torch.cat([agent_yaw, torch.zeros((padding,), dtype=torch.float32)], dim=0)
                    agent_extent = torch.cat([agent_extent, torch.zeros((padding, 3), dtype=torch.float32)], dim=0)
                    agent_labels = torch.cat([torch.ones((len(argsort),), dtype=torch.int64), torch.zeros((padding,), dtype=torch.int64)], dim=0).unsqueeze(0)

                agent_states = torch.cat([agent_location[:0].unsqueeze(0), agent_location[:1].unsqueeze(0), agent_yaw.unsqueeze(1), agent_extent*2[:, 0].unsqueeze(1), agent_extent*2[:, 1].unsqueeze(1)], dim=1)
                targets.update({'agent_states': agent_states})
                targets.update({'agent_labels': agent_labels})

                bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
                entity_mask = self._compute_box_mask(annotations, laye
                def _xy_in_lidar(x: float, y: float) -> bool:
                    return (self.lidar_min_x <= x <= self.lidar_max_x) and (self.lidar_min_y <= y <= self.lidar_max_y)r
                   bev_semantic_map[entity_mask] = label

        return (features, targets)

    def _global_to_ego_2d(self, global_points, ego_position, ego_yaw_rad):
        global_xy = global_points[:, :2]
        ego_xy = np.array(ego_position[:2])

        delta = global_xy - ego_xy

        cos_yaw = np.cos(ego_yaw_rad)
        sin_yaw = np.sin(ego_yaw_rad)

        rotation_ np.array([ 
            [ cos_yaw,  sin_yaw],
            [ sin_yaw, -cos_yaw],
        ])

        return np.matmul(delta, rotation_matrix.T)
    
    def _get_lidar_feature(self, point_cloud) -> torch.Tensor:
        """
        Compute LiDAR feature as 2D histogram, according to Transfuser
        :param agent_input: input dataclass
        :return: LiDAR histogram as torch tensors
        """

        # only consider (x,y,z) & swap axes for (N,3) numpy array
        lidar_pc = point_cloud[:, [1, 0, 2]]

        # NOTE: Code from
        # https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py#L873
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(
                self._config.lidar_min_x,
                self._config.lidar_max_x,
                (self._config.lidar_max_x - self._config.lidar_min_x) * int(self._config.pixels_per_meter) + 1,
            )
            ybins = np.linspace(
                self._config.lidar_min_y,
                self._config.lidar_max_y,
                (self._config.lidar_max_y - self._config.lidar_min_y) * int(self._config.pixels_per_meter) + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self._config.hist_max_per_pixel] = self._config.hist_max_per_pixel
            overhead_splat = hist / self._config.hist_max_per_pixel
            return overhead_splat

        # Remove points above the vehicle
        lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.max_height_lidar]
        below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
        above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
        above_features = splat_points(above)
        if self._config.use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)

        return torch.tensor(features)
    
    def _compute_box_mask(self, annotations, layers):
        """
        Compute binary of bounding boxes in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                # box_value = (x, y, z, length, width, height, yaw) TODO: add intenum
                x, y, heading = box_value[0], box_value[1], box_value[-1]
                box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
                agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
                exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0
    
    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """

        # NOTE: remove half in backward direction
        pixel_center = np.array([[0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)

    def __del__(self):
        if self.file:
            try:
                self.file.close()
            except Exception:
                pass