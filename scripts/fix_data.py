import os
import numpy as np
from pillar_state import State
import pickle
write = False

drop_timestamps = os.listdir("/home/lagrassa/git/plan-abstractions/data/iterative_realpickplace/iterative_training/skills/LiftAndDrop/default__n_iter_1__seed_0")
pick_timestamps = os.listdir("/home/lagrassa/git/plan-abstractions/data/iterative_realpickplace/iterative_training/skills/Pick/default__n_iter_1__seed_0")
for pick_timestamp in pick_timestamps:
    assert pick_timestamp in drop_timestamps
    place_shard_root = f"/home/lagrassa/git/plan-abstractions/data/iterative_realpickplace/iterative_training/skills/LiftAndDrop/default__n_iter_1__seed_0/{pick_timestamp}/data"
    pick_shard_root = f"/home/lagrassa/git/plan-abstractions/data/iterative_realpickplace/iterative_training/skills/Pick/default__n_iter_1__seed_0/{pick_timestamp}/data"
    for place_shard in os.listdir(place_shard_root):
        if "tmp" in place_shard:
            continue
        #change init_state to init_state for pick. 
        with open(f"{pick_shard_root}/{place_shard}", "rb") as pick_shard_file:
            pick_data = pickle.load(pick_shard_file)
            initial_state_str = pick_data[0]["initial_states"][0]
            initial_state = State.create_from_serialized_string(initial_state_str)
            pick_init_states = []
            rod_poses_at_pick = []
            rod_poses_at_place = []
            for rod_idx in [0,1]:
                
                rod_poses_at_pick.append(initial_state.get_values_as_vec([f"frame:rod{rod_idx}:pose/position",f"frame:rod{rod_idx}:pose/quaternion"]))

            place_shard_file = open(f"{place_shard_root}/tmp_{place_shard}", "rb")
            place_data = pickle.load(place_shard_file)
            place_initial_state_str = place_data[0]["initial_states"][0]
            place_initial_state = State.create_from_serialized_string(place_initial_state_str)

            for rod_idx in [0,1]:
                rod_poses_at_place.append(place_initial_state.get_values_as_vec([f"frame:rod{rod_idx}:pose/position",f"frame:rod{rod_idx}:pose/quaternion"]))
                place_initial_state.update_property(f"frame:rod{rod_idx}:pose/position", rod_poses_at_pick[rod_idx][:3])
                place_initial_state.update_property(f"frame:rod{rod_idx}:pose/quaternion", rod_poses_at_pick[rod_idx][3:])
            place_data[0]["initial_states"][0] = place_initial_state.get_serialized_string()
            place_shard_file.close()
            if write:
                place_shard_file_to_write_to = open(f"{place_shard_root}/tmp_{place_shard}", "wb")
                pickle.dump(place_data, place_shard_file_to_write_to)
                place_shard_file_to_write_to.close()


                
            for rod_pose_pick, rod_pose_place in zip(rod_poses_at_pick, rod_poses_at_place):
                print("At pick", np.array(rod_pose_pick[:3]).round(2))
                print("At place", np.array(rod_pose_place[:3]).round(2))
                print(np.linalg.norm(np.array(rod_pose_pick[:2])-np.array(rod_pose_place[:2])))
            print("Next file....")


