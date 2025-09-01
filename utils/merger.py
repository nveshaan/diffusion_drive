import os
import h5py
import argparse
from tqdm import tqdm

def merge_hdf5_files(sprint_dir, marathon_file):
    """
    Merges all sprint HDF5 files in a directory into a single marathon HDF5 file.
    """
    sprint_files = sorted([f for f in os.listdir(sprint_dir) if f.startswith('sprint_') and f.endswith('.hdf5')])

    if not sprint_files:
        print(f"No sprint files found in {sprint_dir}")
        return

    print(f"Found {len(sprint_files)} sprint files to merge.")

    if not os.path.exists(marathon_file):
        with h5py.File(marathon_file, 'w') as f:
            f.create_group('runs')

    with h5py.File(marathon_file, 'a') as agg:
        current_count = len(agg.get('runs', {}))
        
        for sprint_file in tqdm(sprint_files, desc="Merging Sprints"):
            sprint_path = os.path.join(sprint_dir, sprint_file)
            try:
                with h5py.File(sprint_path, 'r') as src:
                    if 'runs' not in src:
                        print(f"No 'runs' group in {sprint_path}")
                        continue

                    # Filter out keys that are not directories, and then sort numerically
                    run_keys = [k for k in src['runs'].keys() if isinstance(src['runs'][k], h5py.Group)]
                    
                    # Attempt to sort numerically, but handle non-integer keys gracefully
                    def sort_key(k):
                        try:
                            return int(k.split('_')[-1])
                        except (ValueError, IndexError):
                            return k
                    
                    sorted_keys = sorted(run_keys, key=sort_key)
                    
                    for run_key in sorted_keys:
                        new_run_id = f"{current_count}"
                        src.copy(f'runs/{run_key}', agg['runs'], name=new_run_id)
                        # print(f"Copied run {run_key} from {sprint_file} to {new_run_id}")
                        current_count += 1

            except Exception as e:
                print(f"Could not process file {sprint_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge sprint HDF5 files into a single marathon file.")
    parser.add_argument('--sprint_dir', type=str, required=True, help='Directory containing sprint HDF5 files.')
    parser.add_argument('--marathon_file', type=str, default='marathon.hdf5', help='Output marathon HDF5 file.')
    args = parser.parse_args()

    merge_hdf5_files(args.sprint_dir, args.marathon_file)
