
import numpy as np
from tqdm import tqdm
import json as js
import os, pathos
import multiprocessing

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.path.join(root,'data')
dataset_root = os.path.join(root,'dataset')


def CHIP_seq_hg38(args):
    if "paternal" in args["dataset"]:
        cell_type = "testis"
    elif "maternal" in args["dataset"]:
        cell_type = "ovary"

    root_feature = os.path.join(dataset_root,
                                args["dataset"],
                                cell_type+"_CHIP_seq")


    print("=> Loading {}".format(os.path.join(root_feature, 'X.npy')))
    X   = np.load(os.path.join(root_feature, 'X.npy'))
    print("=> Loading {}".format(os.path.join(root_feature, 'Y.npy')))
    Y     = np.load(os.path.join(root_feature, 'Y.npy'))

    return X, Y
