import numpy as np
import sys
import os
# the root here is "iRspot-SeqModel/data"
root = os.path.dirname(__file__)
sys.path.insert(0, root)
from tqdm import tqdm
import json as js

def get_chr_data():
    #TODO this function get the chr1-chr22 into json file
    for i in range(1,23):
        data_base = {}
        file_name = os.path.join(root, 'hg38','chr'+str(i)+ '.fa')
        with open(file_name, "r") as f:
            logo = None
            fa_Seq = ""
            for line in tqdm(f.readlines()):
                line = line.rstrip()  # 去掉行末的换行符
                if line[0] == ">":  # 判断如果是信息行，就存储在fa_Info，否则存储在fa_Seq
                    logo = line[1:]
                else:
                    fa_Seq = fa_Seq + line
            data_base[logo]= fa_Seq
        save_json_file = os.path.join(root,'hg38','chr{}'.format(i)+'.json')
        print('saving ==>{}'.format(save_json_file))
        with open(save_json_file, 'w') as fw:
            js.dump(data_base, fw)

def get_chr_into_database():
    # TODO his function get chr1.json - chr22.json into a big json file
    # TODO big file name is database.json under root hg38
    # TODO the big_data_base looks like :
    # TODO "chr1": "ATCGGCTAGCAA..."
    # TODO "chr2": "GCTAGCTAGCAA..."
    # TODO "chr3": "CCTAGCTAGAGC..."T
    # TODO "chr4": "ATCGGCTAGCAA..."
    big_data_base = {}

    for i in tqdm(range(1,23)):
        file_name = os.path.join(root, 'hg38','chr'+str(i)+ '.json')
        data = js.load(open(file_name,'r'))
        for key,value in data.items():
            big_data_base[key] = data[key]
    big_json_file = os.path.join(root, 'hg38','database.json')
    print('saving ==>{}'.format(big_json_file))
    with open(big_json_file, 'w') as fw:
        js.dump(big_data_base, fw)

def get_database():
    data_root = os.path.join(root,'hg38')
    if not os.path.exists(data_root):
        raise Exception("Please have {} in dir".format(data_root))
    # creating empty dic for samples
    print('=> Generating chr.json data')
    get_chr_data()
    print('=> Generating data_base')
    get_chr_into_database()

def get_paternal_maternal_data():
    print('=> Loading data_base')
    data_base_root = os.path.join(root, 'hg38', 'database.json')
    if not os.path.exists(data_base_root):
        get_database()
    data_base = js.load(open(data_base_root,'r'))
    Pap_Mom = []
    # Generate 1:paternal&maternal
    data_root = os.path.join(root, 'decode_data_2019_science')
    with open(os.path.join(data_root,'aau1043_DataS3')) as f:
        for row in tqdm(f.readlines()):
            if '#' in row :
                continue
            else:
                row = row.split('\t')
                if row[0] not in data_base.keys():
                    continue
                cell = get_chromosome_data(row , data_base)
                if cell is not None:
                    Pap_Mom.append(cell)
    # Normalize data and Label hot or cold
    Pap_Mom = Label_zScore_normalization(Pap_Mom)
    file = os.path.join(root,'paternal_maternal.json')
    print('saving ==>{}'.format(file))
    with open(file, 'w') as fw:
        js.dump(Pap_Mom, fw)
    print('{} SAVED !!!'.format(file))

def get_paternal_data():
    print('=> Loading data_base')
    data_base_root = os.path.join(root, 'hg38', 'database.json')
    if not os.path.exists(data_base_root):
        get_database()
    data_base = js.load(open(data_base_root, 'r'))
    Pap = []
    # Generate 1:paternal&maternal
    data_root = os.path.join(root, 'decode_data_2019_science')
    with open(os.path.join(data_root, 'aau1043_DataS1')) as f:
        for row in tqdm(f.readlines()):
            if '#' in row:
                continue
            else:
                row = row.split('\t')
                if row[0] not in data_base.keys():
                    continue
                cell = get_chromosome_data(row, data_base)
                if cell is not None:
                    Pap.append(cell)
    #Normalize data and Label hot or cold
    Pap = Label_zScore_normalization(Pap)

    file = os.path.join(root, 'paternal.json')
    print('saving ==>{}'.format(file))
    with open(file, 'w') as fw:
        js.dump(Pap, fw)
    print('{} SAVED !!!'.format(file))

def get_maternal_data():
    print('=> Loading data_base')
    data_base_root = os.path.join(root, 'hg38', 'database.json')
    if not os.path.exists(data_base_root):
        get_database()
    data_base = js.load(open(data_base_root, 'r'))
    Mom = []
    # Generate 1:paternal&maternal
    data_root = os.path.join(root, 'decode_data_2019_science')
    with open(os.path.join(data_root, 'aau1043_DataS2')) as f:
        for row in tqdm(f.readlines()):
            if '#' in row:
                continue
            else:
                row = row.split('\t')
                if row[0] not in data_base.keys():
                    continue
                cell = get_chromosome_data(row, data_base)
                if cell is not None:
                    Mom.append(cell)
    # Normalize data and Label hot or cold
    Mom = Label_zScore_normalization(Mom)
    file = os.path.join(root, 'maternal.json')
    print('saving ==>{}'.format(file))
    with open(file, 'w') as fw:
        js.dump(Mom, fw)
    print('{} SAVED !!!'.format(file))

def get_chromosome_data(line,data_base):
    cell = {}
    chromosome = line[0]
    whole_sequence = data_base[chromosome]

    start_index = int(line[1])
    end_index   = int(line[2])
    seqence = whole_sequence[start_index:end_index]
    length = len(seqence)
    recom_rate  = float(line[3])
    # print(idx)
    # print('start_index--{}'.format(start_index))
    # print('end_index--{}'.format(end_index))
    # print('length -- {}'.format(length))
    # print('recom_rate--{}'.format(recom_rate))
    cell["id"]          = None
    cell["chromosome"]  = chromosome
    cell["label"]       = None
    cell["seqence"]     = seqence
    cell["rate"]        = recom_rate
    cell["start_index"] = start_index
    cell["end_index"]   = end_index
    cell["length"]      = length
    return cell

def get_summary_data(type):
    if type == "paternal":
        get_paternal_data()
    elif type == "maternal":
        get_maternal_data()
    else:
        get_paternal_maternal_data()

def Label_zScore_normalization(data):
    # import math
    data = list(reversed(sorted(data , key=lambda d: d["rate"])))
    hot_data_id, cold_data_id = [],[]
    for i in range(len(data)):
        data[i]["id"] = i
        if int(data[i]["rate"]) > 10:
            data[i]["label"] = "hot"
            hot_data_id.append(i)
        elif int(data[i]["rate"] < 0.5):
            data[i]["label"] = "cold"
            cold_data_id.append(i)

    return data

def sample():
    args = {
        "dataset": "human_science_2019",
        "num_data": 40000,
        "max_length": 1000,
        "min_length": 500,
        "model": "CNN",
        "log_dir": "logs",
        "output_dir": "output",
        "output_prefix": "CNN",

        "input_length": 1000,
        "input_height": 4,
        "data_augmentation": 1,
        "get_distribution": 0,
        "get_line_comparison": 1,
        "equivariant": 0,

        "dp_rate": 0.1,

        "trails": 4,
        "folds": 5,
        "epochs": 100,
        "batch_size": 64,
        "lr": 0.001,

        "init_weight": 0,
        "reinforce_train": 0,
        "loss": "binary_crossentropy"
    }
    interval = 25
    num_hot = 20000
    num_cold = 20000
    # get_paternal_maternal_data()
    data_raw = js.load(open("paternal_maternal.json", 'r'))
    datas = []
    # Save data with proper length
    for d in data_raw:
        if d["length"] < args["min_length"] or d["length"] > args["max_length"]:
            continue
        else:
            datas.append(d)
    datas = sorted(datas, key=lambda d: d["rate"])
    num_dic = int((args["max_length"] - args["min_length"]) / interval)
    # dic for length control
    # Because hot data has relative short length
    # So to make sure the Neureal net don't focus on length
    # We need to make the cold data same in lenght and number as hot data
    dic = {}
    for i_n in range(num_dic):
        start_idx = int(args["min_length"] + i_n * interval)
        end_idx = int(args["min_length"] + (i_n + 1) * interval)
        dic["{}-{}".format(start_idx, end_idx)] = 0
    hot_data_list = list(reversed(datas[-num_hot:]))
    # hot_data_list_raw: [most hot -> less hot]
    cold_data_list_raw = datas[:int(0.5 * len(datas))]
    cold_data_list = []
    print("Sorting Hot Data")
    with tqdm(total=len(hot_data_list)) as t:
        for hot_candidate in hot_data_list:
            length = hot_candidate["length"]
            idx = int((length - args["min_length"] - 1) / interval)
            start_idx = int(args["min_length"] + idx * interval)
            end_idx = int(args["min_length"] + (idx + 1) * interval)
            dic["{}-{}".format(start_idx, end_idx)] += 1
            t.update()
    print("Sorting Cold Data")
    with tqdm(total=len(cold_data_list_raw)) as t:
        for cold_candidate in cold_data_list_raw:
            length = cold_candidate["length"]
            idx = int((length - args["min_length"] - 1) / interval)
            start_idx = int(args["min_length"] + idx * interval)
            end_idx = int(args["min_length"] + (idx + 1) * interval)

            if dic["{}-{}".format(start_idx, end_idx)] > 0:
                cold_data_list.append(cold_candidate)
                dic["{}-{}".format(start_idx, end_idx)] -= 1
            t.update()
    return  hot_data_list,cold_data_list
    # hot_rate = [h["rate"] for h in hot_data_list]
    # cold_rate = [c["rate"] for c in cold_data_list]
    # print("hot data rate average : {}".format(np.average(hot_rate)))
    # print("cold data rate average : {}".format(np.average(cold_rate)))
    # print("hot data num : {}".format(len(hot_rate)))
    # print("cold data num : {}".format(len(cold_rate)))
    #
    # chromosome_hot_list, chromosome_cold_list = [], []
    # for d in data_raw:
    #     if d["length"] < args["min_length"] or d["length"] > args["max_length"]:
    #         continue
    #     elif d["label"] == "hot":
    #         chromosome_hot_list.append(d)
    #     elif d["label"] == "cold":
    #         chromosome_cold_list.append(d)
    # chromosome_hot_rate = [h["rate"] for h in chromosome_hot_list]
    # chromosome_cold_rate = [h["rate"] for h in chromosome_cold_list]
    # print("\nchromosome hot data rate average : {}".format(np.average(chromosome_hot_rate)))
    # print("chromosome cold data rate average : {}".format(np.average(chromosome_cold_rate)))
    # print("chromosome hot data num : {}".format(len(chromosome_hot_rate)))
    # print("chromosome cold data num : {}".format(len(chromosome_cold_rate)))

if __name__ == '__main__':
    # get_paternal_maternal_data()

    data = js.load(open("data/maternal.json", 'r'))

    num_hot, num_cold = 0, 0
    hot_rate, cold_rate = 0, 0
    for d in tqdm(data):
        if d["rate"] >=5:
            num_hot +=1
            hot_rate += d["rate"]
        elif d["rate"]<1:
            num_cold += 1
            cold_rate += d["rate"]
    print("num hot : {} num cold {}".format(num_hot, num_cold))
    print("hot_rate : {} cold_rate {:.20f}".format(hot_rate / num_hot, cold_rate / num_cold))



