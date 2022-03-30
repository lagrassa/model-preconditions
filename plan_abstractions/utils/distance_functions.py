import numpy as np

def xy_and_water_out(state1, state2):
    pos_diff = np.linalg.norm(state1[:2]-state2[:2])
    water_out_difference = abs(state1[-1]-state2[-1])
    return pos_diff + 0.2*water_out_difference


def xy_and_water_out_and_water_in(state1, state2):
    pos_diff = np.linalg.norm(state1[:2]-state2[:2])
    water_out_difference = abs(state1[-1]-state2[-1])
    water_in_difference = abs(state1[-2]-state2[-2])
    return pos_diff + (water_out_difference + water_in_difference)

