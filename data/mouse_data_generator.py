import numpy as np
import sys
import os
# the root here is "iRspot-SeqModel/data"
root = os.path.dirname(__file__)
sys.path.insert(0, root)
from tqdm import tqdm
import json as js
# from data.generate_data import Label_zScore_normalization

def get_mouse_chr_data():
    #TODO this function get the chr1-chr20 into json file
    data_base = {}
    for i in range(1,21):
        chromosome = 'chr'+str(i)
        file_name = os.path.join(root, 'hm38','chr'+str(i)+ '.fa')
        print('=> {}'.format('chr'+str(i)))
        with open(file_name, "r") as f:
            fa_Seq = ""
            for line in tqdm(f.readlines()):
                line = line.rstrip()  # 去掉行末的换行符
                if line[0] == ">":  # 判断如果是信息行，就存储在fa_Info，否则存储在fa_Seq
                    continue
                else:
                    fa_Seq = fa_Seq + line
            data_base[chromosome] = fa_Seq
        save_json_file = os.path.join(root,'hm38','chr{}'.format(i)+'.json')
        print('saving ==>{}'.format(save_json_file))
        with open(save_json_file, 'w') as fw:
            js.dump(data_base, fw)

def get_mouse_chr_into_database():
    # TODO his function get chr1.json - chr20.json into a big json file
    # TODO big file name is database.json under root hm38
    # TODO the big_data_base looks like :
    # TODO "chr1": "ATCGGCTAGCAA..."
    # TODO "chr2": "GCTAGCTAGCAA..."
    # TODO "chr3": "CCTAGCTAGAGC..."T
    # TODO "chr4": "ATCGGCTAGCAA..."
    big_data_base = {}

    for i in tqdm(range(1,21)):
        chromosome = 'chr'+str(i)
        file_name = os.path.join(root, 'hm38','chr'+str(i)+ '.json')
        data = js.load(open(file_name,'r'))
        big_data_base[chromosome] = data[chromosome]
    big_json_file = os.path.join(root, 'hm38','database.json')
    print('saving ==>{}'.format(big_json_file))
    with open(big_json_file, 'w') as fw:
        js.dump(big_data_base, fw)

def get_mouse_avg_data():
    print('=> Loading data_base')
    data_base_root = os.path.join(root, 'hm38', 'database.json')

    data_base = js.load(open(data_base_root,'r'))
    Pap_Mom = []
    # Generate 1:paternal&maternal
    data_root = os.path.join(root, 'mouse2014')
    with open(os.path.join(data_root,'avg_rate_map')) as f:
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
    file = os.path.join(root,'mouse_paternal_maternal.json')
    print('saving ==>{}'.format(file))
    with open(file, 'w') as fw:
        js.dump(Pap_Mom, fw)
    print('{} SAVED !!!'.format(file))

def get_mouse_male_data():
    print('=> Loading data_base')
    data_base_root = os.path.join(root, 'hm38', 'database.json')

    data_base = js.load(open(data_base_root,'r'))
    Pap = []
    # Generate 1:paternal&maternal
    data_root = os.path.join(root, 'mouse2014')
    with open(os.path.join(data_root,'male_rate_map')) as f:
        for row in tqdm(f.readlines()):
            if '#' in row :
                continue
            else:
                row = row.split('\t')
                if row[0] not in data_base.keys():
                    continue
                cell = get_chromosome_data(row , data_base)
                if cell is not None:
                    Pap.append(cell)
    # Normalize data and Label hot or cold
    Pap = Label_zScore_normalization(Pap)
    file = os.path.join(root,'mouse_paternal.json')
    print('saving ==>{}'.format(file))
    with open(file, 'w') as fw:
        js.dump(Pap, fw)
    print('{} SAVED !!!'.format(file))

def get_mouse_female_data():
    print('=> Loading data_base')
    data_base_root = os.path.join(root, 'hm38', 'database.json')

    data_base = js.load(open(data_base_root,'r'))
    Mom = []
    # Generate 1:paternal&maternal
    data_root = os.path.join(root, 'mouse2014')
    with open(os.path.join(data_root,'female_rate_map')) as f:
        for row in tqdm(f.readlines()):
            if '#' in row :
                continue
            else:
                row = row.split('\t')
                if row[0] not in data_base.keys():
                    continue
                cell = get_chromosome_data(row , data_base)
                if cell is not None:
                    Mom.append(cell)
    # Normalize data and Label hot or cold
    Mom = Label_zScore_normalization(Mom)
    file = os.path.join(root,'mouse_maternal.json')
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

    # if end_index - start_index < 500 :
    #     mid = start_index + int(0.5 * int(end_index - start_index))
    #     end_index = int(mid + 500)
    #     start_index = int(mid - 500)

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


if __name__ == '__main__':
    get_mouse_chr_data()
    get_mouse_chr_into_database()
    # get_mouse_male_data()
    # get_mouse_female_data()
    # database = js.load(open(os.path.join(root,'mouse_paternal_maternal.json')))
    # num_correct = 0
    # num_hot ,num_cold = 0,0
    # hot_rate, cold_rate = 0,0
    # hot_len, cold_len = 0,0
    # for d in database:
    #     if d['length']>=500 and d['length']<=1000:
    #         if d['label'] == 'hot':
    #             hot_rate +=d["rate"]
    #             num_hot +=1
    #             hot_len += d["length"]
    #         elif d['label'] == 'cold':
    #             cold_rate +=d["rate"]
    #             num_cold +=1
    #             cold_len += d["length"]
    # print('num_hot :{} num_cold:{}'.format(num_hot, num_cold))
    # print('hot_rate :{} cold_rate:{}'.format(hot_rate/num_hot,cold_rate/num_cold))
    # print('hot_len_avg :{} cold_len_avg:{}'.format(hot_len/num_hot,cold_len/num_cold))