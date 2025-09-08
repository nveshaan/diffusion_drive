import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from math import pi
import cv2

class SampleData(Dataset):
    def __init__(self, file_path, gap=0, num_poses=8, stride=5, skip=150):
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

        self.max_height_lidar: float = 100.0
        self.pixels_per_meter: float = 4.0
        self.hist_max_per_pixel: int = 5

        self.lidar_min_x: float = -32
        self.lidar_max_x: float = 32
        self.lidar_min_y: float = -32
        self.lidar_max_y: float = 32

        self.lidar_split_height: float = 0.2
        self.use_ground_plane: bool = False

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
                v = np.array(data[idxs]).squeeze()
                # Use 2D speed if available; fallback to full norm
                velocity = float(np.linalg.norm(v[:2] if v.shape[-1] >= 2 else v))
            elif key == 'acceleration':
                a = np.array(data[idxs]).squeeze()
                acceleration = float(np.linalg.norm(a[:2] if a.shape[-1] >= 2 else a))
            elif key == 'command':
                command = np.array(data[idxs]).squeeze()

        features.update({'status_feature': torch.tensor([command, command, command, command, velocity, velocity, acceleration, acceleration], dtype=torch.float32).unsqueeze(0)})

        pred_start = start_idx + self.gap
        # Build targets
        # 1) Trajectory (ego frame)
        ego_loc_ds = run['vehicles/ego/location']
        idxs = [pred_start + i * self.stride for i in range(self.num_poses)]
        ego_traj_global = np.array(ego_loc_ds[idxs])
        ego_start_global = np.array(ego_loc_ds[start_idx])
        ego_yaw_deg = float(run['vehicles/ego/rotation'][start_idx][1])
        traj_xy = self._global_to_ego_2d(ego_traj_global, ego_start_global, ego_yaw_deg)
        targets.update({'trajectory': torch.tensor(traj_xy, dtype=torch.float32)})

        # 2) Agents: filter with _xy_in_lidar, select nearest 30, pad
        def _xy_in_lidar(x: float, y: float) -> bool:
            return (self.lidar_min_x <= x <= self.lidar_max_x) and (self.lidar_min_y <= y <= self.lidar_max_y)

        agent_pos_list = []
        agent_yaw_list = []
        agent_len_list = []
        agent_wid_list = []
        agent_label_list = []

        # Collect vehicles (exclude ego)
        vehicles = [v for v in run['vehicles'] if v != 'ego']
        for vehicle in vehicles:
            v_loc = np.array(run[f'vehicles/{vehicle}/location'][start_idx])
            rel_xy = self._global_to_ego_2d(np.array([v_loc]), ego_start_global, ego_yaw_deg)[0]
            if not _xy_in_lidar(float(rel_xy[0]), float(rel_xy[1])):
                continue
            v_yaw_deg = float(run[f'vehicles/{vehicle}/rotation'][start_idx][1])
            # relative yaw in radians, wrapped to [-pi, pi]
            yaw_rel = np.deg2rad(v_yaw_deg - ego_yaw_deg)
            yaw_rel = (yaw_rel + np.pi) % (2 * np.pi) - np.pi
            extent = np.array(run[f'vehicles/{vehicle}/extent'][start_idx])  # (ex, ey, ez) half-sizes
            agent_pos_list.append(rel_xy.astype(np.float32))
            agent_yaw_list.append(np.float32(yaw_rel))
            agent_len_list.append(np.float32(extent[0] * 2.0))
            agent_wid_list.append(np.float32(extent[1] * 2.0))
            agent_label_list.append(1)  # vehicle

        # Collect walkers (if present)
        walkers = [w for w in run['walkers']]
        for walker in walkers:
            w_loc = np.array(run[f'walkers/{walker}/location'][start_idx])
            rel_xy = self._global_to_ego_2d(np.array([w_loc]), ego_start_global, ego_yaw_deg)[0]
            if not _xy_in_lidar(float(rel_xy[0]), float(rel_xy[1])):
                continue
            w_yaw_deg = float(run[f'walkers/{walker}/rotation'][start_idx][1])
            yaw_rel = np.deg2rad(w_yaw_deg - ego_yaw_deg)
            yaw_rel = (yaw_rel + np.pi) % (2 * np.pi) - np.pi
            extent = np.array(run[f'walkers/{walker}/extent'][start_idx])
            agent_pos_list.append(rel_xy.astype(np.float32))
            agent_yaw_list.append(np.float32(yaw_rel))
            agent_len_list.append(np.float32(extent[0] * 2.0))
            agent_wid_list.append(np.float32(extent[1] * 2.0))
            agent_label_list.append(2)  # pedestrian

        if len(agent_pos_list) > 0:
            pos = torch.tensor(np.stack(agent_pos_list, axis=0), dtype=torch.float32)          # (N,2)
            yaw = torch.tensor(np.array(agent_yaw_list), dtype=torch.float32)                  # (N,)
            length = torch.tensor(np.array(agent_len_list), dtype=torch.float32)               # (N,)
            width = torch.tensor(np.array(agent_wid_list), dtype=torch.float32)                # (N,)
            label = torch.tensor(np.array(agent_label_list), dtype=torch.int64)               # (N,)
            # nearest by ego-frame distance
            distances = torch.linalg.norm(pos, dim=1)
            order = torch.argsort(distances)[:30]
            pos = pos[order]
            yaw = yaw[order]
            length = length[order]
            width = width[order]
            label = label[order]
            count = pos.shape[0]
        else:
            pos = torch.zeros((0, 2), dtype=torch.float32)
            yaw = torch.zeros((0,), dtype=torch.float32)
            length = torch.zeros((0,), dtype=torch.float32)
            width = torch.zeros((0,), dtype=torch.float32)
            label = torch.zeros((0,), dtype=torch.int64)
            count = 0

        # pad to 30
        pad = max(0, 30 - count)
        if pad > 0:
            pos = torch.cat([pos, torch.zeros((pad, 2), dtype=torch.float32)], dim=0)
            yaw = torch.cat([yaw, torch.zeros((pad,), dtype=torch.float32)], dim=0)
            length = torch.cat([length, torch.zeros((pad,), dtype=torch.float32)], dim=0)
            width = torch.cat([width, torch.zeros((pad,), dtype=torch.float32)], dim=0)
            label = torch.cat([label, torch.zeros((pad,), dtype=torch.int64)], dim=0)
        agent_states = torch.cat([pos, yaw.unsqueeze(1), length.unsqueeze(1), width.unsqueeze(1), label.unsqueeze(1)], dim=1)  # (30,5)
        agent_labels = torch.zeros((30,), dtype=torch.bool)
        if count > 0:
            agent_labels[:count] = True

        # 3) BEV semantic map of agents (binary), from agent_states
        bev_semantic_map = self._compute_box_mask(agent_states.numpy())

        targets.update({'bev_semantic_map': torch.tensor(bev_semantic_map.astype(np.float32)).unsqueeze(0)})
        targets.update({'agent_states': agent_states[:, :4]})  # (x,y,yaw,len,wid), ignore label
        targets.update({'agent_labels': agent_labels})

        return (features, targets)

    def _global_to_ego_2d(self, global_points, ego_position, ego_yaw_rad):
        global_xy = global_points[:, :2]
        ego_xy = np.array(ego_position[:2])

        delta = global_xy - ego_xy

        cos_yaw = np.cos(ego_yaw_rad)
        sin_yaw = np.sin(ego_yaw_rad)

        rotation_matrix = np.array([
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
                self.lidar_min_x,
                self.lidar_max_x,
                (self.lidar_max_x - self.lidar_min_x) * int(self.pixels_per_meter) + 1,
            )
            ybins = np.linspace(
                self.lidar_min_y,
                self.lidar_max_y,
                (self.lidar_max_y - self.lidar_min_y) * int(self.pixels_per_meter) + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self.hist_max_per_pixel] = self.hist_max_per_pixel
            overhead_splat = hist / self.hist_max_per_pixel
            return overhead_splat

        # Remove points above the vehicle
        lidar_pc = lidar_pc[lidar_pc[..., 2] < self.max_height_lidar]
        below = lidar_pc[lidar_pc[..., 2] <= self.lidar_split_height]
        above = lidar_pc[lidar_pc[..., 2] > self.lidar_split_height]
        above_features = splat_points(above)
        if self.use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)

        return torch.tensor(features)
    
    def _compute_box_mask(self, agent_states):
        """
        Compute BEV semantic map with different integer values for different classes
        :param agent_states: (N,6) numpy array of (x,y,yaw,len,wid,label)
        :return: BEV semantic map as numpy array
        """
        bev_semantic_map = np.zeros(self.bev_semantic_frame, dtype=np.int64)
        box_polygon_mask = np.zeros(self.bev_semantic_frame[::-1], dtype=np.uint8)
        for agent_state in agent_states:
            x, y, heading = agent_state[0], agent_state[1], agent_state[2]
            box_length, box_width, label = agent_state[3], agent_state[4], agent_state[5]
            # calculate corners of the oriented box
            cos_h = np.cos(heading)
            sin_h = np.sin(heading)
            dx1 = (box_length / 2) * cos_h
            dy1 = (box_length / 2) * sin_h
            dx2 = (box_width / 2) * sin_h
            dy2 = (box_width / 2) * cos_h
            agent_box = [[x - dx1 - dx2, y - dy1 + dy2],
                         [x - dx1 + dx2, y - dy1 - dy2],
                         [x + dx1 + dx2, y + dy1 - dy2],
                         [x + dx1 - dx2, y + dy1 + dy2]]
            exterior = np.array(agent_box).reshape((-1, 1, 2))
            exterior = self._coords_to_pixel(exterior)
            cv2.fillPoly(box_polygon_mask, [exterior], color=255)
            # OpenCV has origin on top-left corner
            box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
            entity_mask = box_polygon_mask > 0
            bev_semantic_map[entity_mask] = label  # set to class label
        return bev_semantic_map
    
    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """

        # NOTE: remove half in backward direction
        pixel_center = np.array([[0, self.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)

    def __del__(self):
        if self.file:
            try:
                self.file.close()
            except Exception:
                pass