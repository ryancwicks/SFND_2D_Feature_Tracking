import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    """
    Load in the keypoint csv files and plot histograms for each type.
    """
    input_dir = Path("../build/")
    input_files = list(input_dir.rglob("*_keypoints.csv"))

    data = {}
    figs, ax = plt.subplots (3, 3)
    for file in input_files:
        detector_type = file.name.split("_")[0]
        raw_data = np.loadtxt(str(file))
        data[detector_type] = raw_data

    count = 0
    for detector_type, raw_data in data.items():
        row = int(count / 3)
        col = count % 3
        ax[row, col].hist(raw_data, bins=100, label=detector_type)
        #ax[row, col].set_title (detector_type)
        #ax[row, col].axis([0, 115, 0, 7500])
        count += 1
        ax[row, col].legend()

    for a in ax.flat:
        a.set(xlabel='Pixel', ylabel='Count')
        #a.label_outer()
        
    
    plt.show()
    #plt.savefig("keypoints_distribution.png")


if __name__ == "__main__":
    main()
    
