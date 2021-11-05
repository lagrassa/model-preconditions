import argparse
import os
import numpy as np


def main(args):
    data_path = args.data_path
    for filename in os.listdir(data_path):
        data = np.load(os.path.join(data_path, filename), allow_pickle=True).item()
        print("#############################################")
        print(f"Filename : {filename}")
        print(f" Skills: {data['skills']}")
        print(f" Model: {data['model']}")
        print(f" Test L_g : {data['test']['lg']}")
        print(f" Test MAE : {round(100*data['test']['mean_error'],1)}")
        print(f" Deviation mean : {round(100*data['deviation_mean'],1)}")
        print(f" Deviation std : {round(100*data['deviation_std'],1)}")
        print("#############################################")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", '-d', type=str)
    args = parser.parse_args()
    main(args)
