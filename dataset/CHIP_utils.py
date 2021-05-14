from data.generate_data import *
from dataset.data_generator import *
import numpy as np
from tqdm import tqdm
import json as js
import os, pathos
import multiprocessing
from dataset.transforms import *

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.path.join(root,'data')
dataset_root = os.path.join(root,'dataset')


def CHIP_seq_hg38(args):
    if "paternal" in args["dataset"]:
        type = "paternal"
    elif "maternal" in args["dataset"]:
        type = "maternal"
    else:
        type = ""

    root_feature = os.path.join(dataset_root,
                                args["dataset"],
                                type + "CHIP_seq")
    print("=> Loading {}".format(os.path.join(root_feature, 'X.npy')))
    X   = np.load(os.path.join(root_feature, 'X.npy'))
    print("=> Loading {}".format(os.path.join(root_feature, 'Y.npy')))
    Y     = np.load(os.path.join(root_feature, 'Y.npy'))

    return X, Y

def Chromosome_CHIP_seq_hg38(args):

    root_feature = os.path.join(dataset_root,
                                args["dataset"],
                                "Chromosome_Data",
                                "CHIP_seq"
                                )

    X     = np.load(os.path.join(root_feature, 'X.npy'))
    Y     = np.load(os.path.join(root_feature, 'Y.npy'))

    return X, Y


def CHIP_seq_data_generator(args,
                            root_feature,
                            hot_data_list,
                            cold_data_list):
    X,Y = [],[]
    HOT,COLD = [],[]
    cores = pathos.multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes= int(cores))

    print('=>generating CHIP_seq Hot data')
    with tqdm(total=len(hot_data_list)) as t:
        for hot_feature in pool.imap_unordered(CHIP_seq_Feature_extraction,
                                    hot_data_list):
            X.append(hot_feature)
            HOT.append(hot_feature)
            Y.append(1)
            t.update()


    print('=>generating CHIP_seq Cold data')
    with tqdm(total=len(cold_data_list)) as t:
        for cold_feature in pool.imap_unordered(CHIP_seq_Feature_extraction,
                                    cold_data_list):
            X.append(cold_feature)
            COLD.append(cold_feature)
            Y.append(0)
            t.update()

    pool.close()
    pool.join()

    X = np.array(X)
    Y = np.array(Y)
    HOT = np.array(HOT)
    COLD= np.array(COLD)

    print(root_feature)
    if not os.path.exists(root_feature):
        os.makedirs(root_feature)
    print('saving => {}'.format(os.path.join(root_feature, 'X.npy')))
    np.save(os.path.join(root_feature, 'X.npy'), X)
    print('saving => {}'.format(os.path.join(root_feature, 'Y.npy')))
    np.save(os.path.join(root_feature, 'Y.npy'), Y)

    np.save(os.path.join(root_feature, "HOT.npy"), HOT)
    print("HOT.npy SAVED !!")
    np.save(os.path.join(root_feature, "COLD.npy"), COLD)
    print("COLD.npy SAVED !!")

    from lib.vis import Feature_Comparison
    feature_list = ["H3k4me1 Score",
                    "H3k4me1 signalValue",
                    "H3k4me1 pValue",
                    "H3k4me1 qValue",
                    "H3k4me1 peakValue",
                    "H3k4me3 Score",
                    "H3k4me3 signalValue",
                    "H3k4me3 pValue",
                    "H3k4me3 qValue",
                    "H3k4me3 peakValue",
                    "H3k27ac Score",
                    "H3k27ac signalValue",
                    "H3k27ac pValue",
                    "H3k27ac qValue",
                    "H3k27ac peakValue"
    ]
    for i in range(X.shape[1]):
        Feature_Comparison(root_feature, HOT[:, i], COLD[:, i], feature_list[i])

    return X,Y

def CHIP_seq_Feature_extraction(seq):
    chr = seq["chromosome"]
    start = seq["start_index"]
    end = seq["end_index"]
    middle = int(0.5*(start + end))
    H3K4me3_file = os.path.join(data_root, "CHIP_seq", "H3K4me3", "H3k4me3_conservative_peaks.bed")
    H3K4me1_file = os.path.join(data_root, "CHIP_seq", "H3K4me1", "H3k4me1_conservative_peaks.bed")
    H3K27ac_file = os.path.join(data_root, "CHIP_seq", "H3K27ac", "H3k27ac_conservative_peaks.bed")
    # Find the min_distance/score/signalScore/pValue/qValue feature
    # of the nearest histone
    H3K4me1_feature = Get_Value(chr, middle, H3K4me1_file)
    H3K4me3_feature = Get_Value(chr, middle, H3K4me3_file)
    H3K27ac_feature = Get_Value(chr, middle, H3K27ac_file)

    output = np.hstack((H3K4me1_feature,H3K4me3_feature,H3K27ac_feature))
    return output

def Get_Value(chr, middle, file_name):
    f = open(file_name)
    min_distance = 1000
    score, signalValue, pValue, qValue, peakValue = 0, 0, 0, 0, 0
    for line in (f.readlines()):
        line = line.split('\t', )
        chromosome          = line[0]
        start_idx           = int(line[1])
        end_idx             = int(line[2])
        score_read          = float(line[4])
        signalValue_read    = float(line[6])
        pValue_read         = float(line[7])
        qValue_read         = float(line[8])
        peakValue_read      = float(line[9])
        # direction = line[5].strip('\n')
        if chr != chromosome:
            continue
        else:
            peak = int(0.5 * (start_idx + end_idx))
            dis = np.abs(peak - middle)
            if dis < min_distance:
                min_distance = dis
                score = score_read
                signalValue = signalValue_read
                pValue = pValue_read
                qValue = qValue_read
                peakValue = peakValue_read

    # chromosome_sequence = js.load(open(os.path.join(data_root,"hg38","{}.json".format(chr))))
    # promotor = chromosome_sequence[chr][int(peak_base-500) : int(peak_base+500) ]
    # output, _ = one_hot_padding(promotor,1000)
    # return output
    return np.array([score, signalValue, pValue, qValue, peakValue])
