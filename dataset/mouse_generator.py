import json as js
import pathos,sys,os
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,root)
from dataset.transforms import *
data_root = os.path.join(root,'data')
dataset_root = os.path.join(root,'dataset')
Base = ["A", "C", "G","T"]


def Mouse_Cell_2016_generator(args):
    root_feature = os.path.join(dataset_root, args["dataset"],
                            str(args["max_length"]))

    hot_data_list = js.load(open(os.path.join(root_feature,"hot_data_list.json")))
    cold_data_list = js.load(open(os.path.join(root_feature,"cold_data_list.json")))

    # Label the training data
    print("We have {} hot data, {} cold data ".format(len(hot_data_list), len(cold_data_list)))
    print('=>generating Mouse Genetics Hot data')
    cores = pathos.multiprocessing.cpu_count()
    pool = Pool(processes=int(cores))
    X, Y, L = [], [], []
    with tqdm(total=len(hot_data_list)) as t:
        for x_hot, l_hot in pool.imap(one_hot_padding,
                                      [seq["seqence"] for seq in hot_data_list],
                                      [args["max_length"] for i in range(len(hot_data_list))]):
            X.append(x_hot)
            Y.append(1)
            L.append(l_hot)
            t.update()
    print('=>generating Mouse Genetics Cold data')
    with tqdm(total=len(cold_data_list)) as t:
        for x_cold, l_cold in pool.imap(one_hot_padding,
                                        [seq["seqence"] for seq in cold_data_list],
                                        [args["max_length"] for i in range(len(cold_data_list))]):
            X.append(x_cold)
            Y.append(0)
            L.append(l_cold)
            t.update()

    # Do not close pool for other data generator
    # pool.close()
    # pool.join()

    X = np.array(X)
    Y = np.array(Y)
    L = np.array(L)

    print('saving => {}'.format(os.path.join(root_feature, 'X.npy')))
    np.save(os.path.join(root_feature, 'X.npy'), X)
    print('saving => {}'.format(os.path.join(root_feature, 'Y.npy')))
    np.save(os.path.join(root_feature, 'Y.npy'), Y)
    print('saving => {}'.format(os.path.join(root_feature, 'L.npy')))
    np.save(os.path.join(root_feature, 'L.npy'), L)

    return X, Y, L




def Split_and_Crop_Hot_Cold_Data(args,data_raw,num_hot,num_cold):
    # Length Control
    # TODO data raw from low (start) rate to high (end) rate
    data_raw = sorted(data_raw, key=lambda d: d["rate"])
    hot_data_list,cold_data_list = [],[]

    print("Sorting Hot/Cold Data")
    with tqdm(total=num_hot) as t:
        for i in range(num_hot):
            d = data_raw[ -1 - i]
            if d["length"] < 10 :
                continue
            d = crop_sequence(d, args["min_length"], args["max_length"])
            hot_data_list.append(d)
            t.update()

    with tqdm(total=num_cold) as t:
        for i in range(num_cold):
            d = data_raw[i]
            if d["length"] < 10 :
                continue
            d = crop_sequence(d, args["min_length"], args["max_length"])
            cold_data_list.append(d)
            t.update()


    return hot_data_list,cold_data_list


def crop_sequence(d, min_len, max_len):
    seq = d["seqence"]
    length = d["length"]
    mid = d["start_index"] + int(0.5 * int(d["end_index"] - d["start_index"]))

    if length > max_len:
        d["end_index"] = int(mid + 0.5 * max_len)
        d["start_index"] = int(mid - 0.5 * max_len)
        m = int(0.5 * length)
        d["seqence"] = seq[int(m - 0.5 * max_len): int(m + 0.5 * max_len)]
        d["length"] = max_len

    return d