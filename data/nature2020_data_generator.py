import numpy as np
import sys
import os
# the root here is "iRspot-SeqModel/data"
root = os.path.dirname(__file__)
sys.path.insert(0, root)
# from data.generate_data import get_database
# from data.generate_data import Label_zScore_normalization
from tqdm import tqdm
import json as js


def get_nature2020_summary_data():
    data_base_root = os.path.join(root, 'hg38', 'database.json')
    # if not os.path.exists(data_base_root):
    #     get_database()
    data_base = js.load(open(data_base_root, 'r'))
    Out = []
    # Generate 1:paternal&maternal
    data_root = os.path.join(root, 'nature2020')
    with open(os.path.join(data_root, 'nature2020_map')) as f:
        for row in tqdm(f.readlines()):
            if '#' in row:
                continue
            else:
                row = row.split('\t')
                spot_list = crop_chromosome_data(row, data_base)
                if spot_list is not None:
                    for element in spot_list:
                        Out.append(element)
    # Normalize data and Label hot or cold
    # Out = Label_zScore_normalization(Out)
    file = os.path.join(root, 'nature_2020_full.json')
    print('saving ==>{}'.format(file))
    with open(file, 'w') as fw:
        js.dump(Out, fw)
    print('{} SAVED !!!'.format(file))


def crop_chromosome_data(line,data_base):

    chromosome = line[0]
    whole_sequence = data_base[chromosome]
    start_index = int(line[1])
    end_index   = int(line[2])


    length = 1000
    recom_rate  = float(line[3])
    # print(idx)
    # print('start_index--{}'.format(start_index))
    # print('end_index--{}'.format(end_index))
    # print('length -- {}'.format(length))
    # print('recom_rate--{}'.format(recom_rate))
    out_list = []
    start = start_index
    # end = start_index + length
    end = end_index
    # for i in range(50):
    seqence = whole_sequence[start:end]

    cell = {}
    cell["id"]          = None
    cell["chromosome"]  = chromosome
    cell["label"]       = None
    cell["seqence"]     = seqence
    cell["rate"]        = recom_rate
    cell["start_index"] = start
    cell["end_index"]   = end
    cell["length"]      = end - start

    # start += length
    # end += length
    out_list.append(cell)

    return out_list


if __name__ == '__main__':
    get_nature2020_summary_data()

