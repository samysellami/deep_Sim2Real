from pickletools import read_uint1
import numpy as np
from deep_calibration import script_dir


def save_read_data(file_name=None, io='w', data=None):
    read_data = None
    with open(f"{script_dir}/calibration/saved_data/{file_name}.npy", f'{io}b') as f:
        if io == 'w':
            f.truncate(0)
            np.save(f, data, allow_pickle=True)
        else:
            read_data = np.load(f, allow_pickle=True).item()
        f.close()
    return read_data


def get_length_dict(dict):
    count = 0
    for value in dict.values():
        count += len(value)
    return count
