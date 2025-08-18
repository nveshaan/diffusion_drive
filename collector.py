import time
import carla
import random
import numpy as np
import h5py
import queue
import yaml
import shutil
import argparse
import os
import subprocess
import sys
import signal
from tqdm import tqdm

FPS = 10

COMMAND_MAP = {
    "VOID": 0,
    "LEFT": 1,
    "RIGHT": 2,
    "STRAIGHT": 3,
    "LANEFOLLOW": 4,
    "CHANGELANELEFT": 5,
    "CHANGELANERIGHT": 6
}

class CarlaSyncMode:
    """
    Source: https://github.com/dotchen/LearningByCheating/blob/release-0.9.6/bird_view/models/resnet.py
    """
    def __init__(self, world, *sensors, fps=20):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / fps
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

def parse_args():
    parser = argparse.ArgumentParser(description="CARLA Data Collection Script")
    parser.add_argument('--duration', type=int, default=50, help='Duration of each run in seconds')
    parser.add_argument('--runs', type=int, default=100, help='Number of runs to perform')
    parser.add_argument('--map', type=str, default='Town10', help='CARLA map name')
    parser.add_argument('--vehicle', type=str, default='vehicle.nissan.patrol', help='Vehicle blueprint ID')
    parser.add_argument('--output', type=str, default='/Marsupium/marathon.hdf5', help='Final HDF5 output file path')
    parser.add_argument('--temp', type=str, default='/Marsupium/sprint.hdf5', help='Temporary HDF5 path for intermediate storage')
    parser.add_argument('--no-progress', action='store_true', help='Disable tqdm progress bars for supervised runs')
    return parser.parse_args()

def save_data_hdf5(file, run, actor, ego, data):
    try:
        with h5py.File(file, 'a') as f:
            run_group = f.require_group(f"runs/{run}")
            vehicle_group = run_group.require_group(f"{actor}/{ego}")
            if ego == 'ego':
                dataset_names = ["image", "laser", "velocity", "acceleration", "location", "rotation", "control", "command"]
            elif actor == 'vehicles':
                dataset_names = ["velocity", "acceleration", "location", "rotation", "extent"]
            else:
                dataset_names = ["velocity", "acceleration", "location", "rotation"]

            for ds_name, d in zip(dataset_names, data):
                d = np.array(d)
                if ds_name in vehicle_group:
                    ds = vehicle_group[ds_name]
                    if ds.shape[1:] != d.shape:
                        print(f"[ERROR] Shape mismatch for '{ds_name}': existing {ds.shape[1:]}, incoming {d.shape}. Skipping.")
                        continue
                    ds.resize((ds.shape[0] + 1,) + d.shape)
                    ds[-1] = d
                else:
                    maxshape = (None,) + d.shape
                    vehicle_group.create_dataset(ds_name, data=d[None], maxshape=maxshape, chunks=True)
            f.flush()

    except Exception as e:
        print(f"[CRITICAL] Failed to save HDF5 data for run {run}: {e}")
        print(ds_name)
        raise

def spawn_sensors(world, blueprint_library, config, vehicles):
    def spawn_sensor(world, blueprint_library, config, attach_to=None):
        sensor_bp = blueprint_library.find(config['type'])
        for key, value in config.get('attributes', {}).items():
            sensor_bp.set_attribute(key, str(value))

        tf = config['transform']
        transform = carla.Transform(
            carla.Location(x=tf['x'], y=tf['y'], z=tf['z']),
            carla.Rotation(pitch=tf['pitch'], yaw=tf['yaw'], roll=tf['roll'])
        )
        return world.spawn_actor(sensor_bp, transform, attach_to=attach_to)
    
    sensors = []
    for vehicle in vehicles:
        sensors.extend([
            spawn_sensor(world, blueprint_library, config['camera'], attach_to=vehicle),
            spawn_sensor(world, blueprint_library, config['lidar'], attach_to=vehicle)
        ])
    return sensors

def spawn_vehicle_or_walker(world, blueprint_library, type):
    spawn_point = random.choice(world.get_map().get_spawn_points())
    return world.spawn_actor(type, spawn_point)

def collect_data(world, tm, blueprint_library, run_no, args, sensor_config):
    ego = spawn_vehicle_or_walker(world, blueprint_library, blueprint_library.find(args.vehicle))
    if ego is None:
        raise RuntimeError("Failed to spawn vehicle.")

    sensors = spawn_sensors(world, blueprint_library, sensor_config, [ego])
    if any(s is None for s in sensors):
        raise RuntimeError("Failed to spawn one or more sensors.")
    
    actors = world.get_actors()
    vehicles = actors.filter('*vehicle*')
    walkers = actors.filter('*walker*')
    
    try:
        with CarlaSyncMode(world, *sensors, fps=FPS) as sync_mode:
            sync_mode.tick(timeout=2.0)
            ego.set_autopilot(True)
            loop = range(FPS * args.duration)
            if not args.no_progress:
                loop = tqdm(loop, desc=f"Run {run_no}")

            for _ in loop:
                snapshot = sync_mode.tick(timeout=2.0)
                image = np.array(snapshot[1].raw_data, dtype=np.uint8).reshape((snapshot[1].height, snapshot[1].width, 4))

                laser = np.frombuffer(snapshot[2].raw_data, dtype=np.float32).reshape((-1, 4))
                expected_points = int(sensor_config['lidar']['attributes']['points_per_second']) // FPS
                if laser.shape[0] < expected_points:
                    laser = np.pad(laser, ((0, expected_points - laser.shape[0]), (0, 0)), mode='constant')

                velocity = ego.get_velocity()
                acceleration = ego.get_acceleration()
                location, rotation = ego.get_transform().location, ego.get_transform().rotation
                control = ego.get_control()
                command, waypoint = tm.get_next_action(ego)
                command = COMMAND_MAP.get(str(command).upper(), -1)

                data = [
                    image, laser,
                    [velocity.x, velocity.y, velocity.z],
                    [acceleration.x, acceleration.y, acceleration.z],
                    [location.x, location.y, location.z],
                    [rotation.pitch, rotation.yaw, rotation.roll],
                    [control.throttle, control.steer, control.brake, control.reverse],
                    [command],
                ]
                save_data_hdf5(args.temp, run_no, 'vehicles', 'ego', data)

                i = 0
                for vehicle in vehicles:
                    velocity = vehicle.get_velocity()
                    acceleration = vehicle.get_acceleration()
                    location, rotation = vehicle.get_transform().location, vehicle.get_transform().rotation
                    extent = vehicle.bounding_box.extent

                    data = [[velocity.x, velocity.y, velocity.z],
                            [acceleration.x, acceleration.y, acceleration.z],
                            [location.x, location.y, location.z],
                            [rotation.pitch, rotation.yaw, rotation.roll],
                            [extent.x, extent.y, extent.z]]
                    
                    save_data_hdf5(args.temp, run_no, 'vehicles', i, data)
                    i += 1

                i = 0
                for walker in walkers:
                    velocity = walker.get_velocity()
                    acceleration = walker.get_acceleration()
                    location, rotation = walker.get_transform().location, walker.get_transform().rotation

                    data = [[velocity.x, velocity.y, velocity.z],
                            [acceleration.x, acceleration.y, acceleration.z],
                            [location.x, location.y, location.z],
                            [rotation.pitch, rotation.yaw, rotation.roll],]
                    
                    save_data_hdf5(args.temp, run_no, 'walkers', i, data)
                    i += 1               

    finally:
        for actor in sensors:
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                print("[WARN] Could not destry sensor")
        for actor in actors:
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                print("[WARN] Could not destroy traffic")
        try:
            if ego is not None and ego.is_alive:
                ego.destroy()
        except RuntimeError:
            print("[WARN] Could not destroy ego")
        time.sleep(1.0)

def main():
    args = parse_args()

    with open("failed_runs.log", "w") as log:
        log.write(f"\n\n---{time.time()}---\n")

    if os.path.exists(args.temp):
        print("Removing existing temporary HDF5 file...")
        os.remove(args.temp)

    with open('sensor_config.yaml', 'r') as file:
        sensor_config = yaml.safe_load(file)

    try:
        client = carla.Client('192.168.212.250', 2000)
        client.set_timeout(10.0)
        # world = client.load_world(args.map)
        world = client.get_world()
        tm = client.get_trafficmanager()
        blueprint_library = world.get_blueprint_library()

        for run_no in range(1, args.runs + 1):
            traffic_proc = None
            try:
                # Start traffic generator as a subprocess for this run
                gen_script = 'python generate_traffic.py --asynch'
                try:
                    traffic_proc = subprocess.Popen(gen_script, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"[INFO] Started traffic generator (pid={traffic_proc.pid}) for run {run_no}")
                except Exception as e:
                    print(f"[WARN] Could not start traffic generator: {e}")
                    traffic_proc = None

                collect_data(world, tm, blueprint_library, run_no, args, sensor_config)

                # Update main output with this run's temp file
                if os.path.exists(args.output):
                    try:
                        os.remove(args.output)
                    except Exception as e:
                        print(f"[WARN] Failed to remove existing output {args.output}: {e}")
                try:
                    shutil.copy(args.temp, args.output)
                    print(f"[INFO] Marathon updated: {args.output}")
                except Exception as e:
                    print(f"[ERROR] Failed to copy temp to output: {e}")

            except Exception as e:
                print(f"Run {run_no} failed: {e}")
                with open("failed_runs.log", "a") as log:
                    log.write(f"Run {run_no} failed: {e}\n")

            finally:
                # Ensure traffic generator subprocess is stopped after each run
                if traffic_proc is not None:
                    try:
                        # try a graceful interrupt first
                        traffic_proc.send_signal(signal.SIGINT)
                        traffic_proc.wait(timeout=10)
                        print(f"[INFO] Traffic generator (pid={traffic_proc.pid}) stopped gracefully")
                    except Exception:
                        try:
                            traffic_proc.terminate()
                            traffic_proc.wait(timeout=10)
                            print(f"[INFO] Traffic generator (pid={traffic_proc.pid}) terminated")
                        except Exception:
                            try:
                                traffic_proc.kill()
                                print(f"[INFO] Traffic generator (pid={traffic_proc.pid}) killed")
                            except Exception as e:
                                print(f"[WARN] Failed to kill traffic generator: {e}")
    finally:
        print("Shutdown complete.")

if __name__ == '__main__':
    main()