import numpy as np


def merge_exec_data(exec_datas):
    all_exec_data = {k: [] for k in exec_datas[0]}

    for exec_data in exec_datas:
        for k, v in exec_data.items():
            all_exec_data[k].append(v)

    for k in ['T_exec', 'costs']:
        all_exec_data[k] = np.array(all_exec_data[k])

    return all_exec_data
