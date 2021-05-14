import json as js
import pathos,sys,os
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,root)
from dataset.transforms import *
from dataset.mouse_generator import Split_and_Crop_Hot_Cold_Data
data_root = os.path.join(root,'data')
dataset_root = os.path.join(root,'dataset')
Base = ["A", "C", "G","T"]

def Human_Science_2019_generator(args,summary_path,root_feature):
    raw_datas = js.load(open(summary_path, 'r'))
    # TODO: dic keys--["label"]["seqence"]["rate"]["length"]
    hot_data_list, cold_data_list = Split_Hot_Cold_Data(args,
                                                        raw_datas,
                                                        num_hot=int(args["num_data"]/2),
                                                        num_cold=int(args["num_data"]/2),
                                                        interval=25)
    if not os.path.exists(root_feature):
        os.makedirs(root_feature)
    print("=> saving {}".format(os.path.join(root_feature,"hot_data_list.json")))
    with open(os.path.join(root_feature,"hot_data_list.json"), 'w') as fw:
        js.dump(hot_data_list, fw)
    print("=> saving {}".format(os.path.join(root_feature,"cold_data_list.json")))
    with open(os.path.join(root_feature,"cold_data_list.json"), 'w') as fw:
        js.dump(cold_data_list, fw)
    #Label the training data
    print("We have {} hot data, {} cold data ".format(len(hot_data_list),len(cold_data_list)))

    print('=>generating Human_Science Hot data')
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
    print('=>generating Human_Science Cold data')
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

    print(root_feature)
    print('saving => {}'.format(os.path.join(root_feature, 'X.npy')))
    np.save(os.path.join(root_feature, 'X.npy'), X)
    print('saving => {}'.format(os.path.join(root_feature, 'Y.npy')))
    np.save(os.path.join(root_feature, 'Y.npy'), Y)
    print('saving => {}'.format(os.path.join(root_feature, 'L.npy')))
    np.save(os.path.join(root_feature, 'L.npy'), L)

    return X,Y,L

def Nature_2020_generator(args,summary_path,root_feature):
    raw_datas = js.load(open(summary_path, 'r'))
    
    hot_data_list, cold_data_list = Split_and_Crop_Hot_Cold_Data(args,
                                                        raw_datas,
                                                        num_hot=int(args["num_data"] / 2),
                                                        num_cold=int(args["num_data"] / 2),
                                                        )
    if not os.path.exists(root_feature):
        os.makedirs(root_feature)
    print("=> saving {}".format(os.path.join(root_feature, "hot_data_list.json")))
    with open(os.path.join(root_feature, "hot_data_list.json"), 'w') as fw:
        js.dump(hot_data_list, fw)
    print("=> saving {}".format(os.path.join(root_feature, "cold_data_list.json")))
    with open(os.path.join(root_feature, "cold_data_list.json"), 'w') as fw:
        js.dump(cold_data_list, fw)
    # Label the training data
    print("We have {} hot data, {} cold data ".format(len(hot_data_list), len(cold_data_list)))

    print('=>generating Nature 2020 Hot data')
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
    print('=>generating Human_Science Cold data')
    with tqdm(total=len(cold_data_list)) as t:
        for x_cold, l_cold in pool.imap(one_hot_padding,
                                        [seq["seqence"] for seq in cold_data_list],
                                        [args["max_length"] for i in range(len(cold_data_list))]):
            X.append(x_cold)
            Y.append(0)
            L.append(l_cold)
            t.update()


    pool.close()
    pool.join()

    X = np.array(X)
    Y = np.array(Y)
    L = np.array(L)

    print(root_feature)
    print('saving => {}'.format(os.path.join(root_feature, 'X.npy')))
    np.save(os.path.join(root_feature, 'X.npy'), X)
    print('saving => {}'.format(os.path.join(root_feature, 'Y.npy')))
    np.save(os.path.join(root_feature, 'Y.npy'), Y)
    print('saving => {}'.format(os.path.join(root_feature, 'L.npy')))
    np.save(os.path.join(root_feature, 'L.npy'), L)

    return X, Y, L

def Chromosome_2019_generator(args,summary_path,root_feature):
    data_raw = js.load(open(summary_path, 'r'))
    datas = []
    for d in data_raw:
        if d["length"]< args["min_length"] or d["length"] > args["max_length"]:
            continue
        else:
            datas.append(d)
    hot_data_list, cold_data_list = [], []
    # Exclude Training Data
    train_hot = js.load(open(os.path.join(dataset_root,
                                                args["dataset"],
                                                str(args["max_length"]),
                                                "hot_data_list.json")))
    train_cold = js.load(open(os.path.join(dataset_root,
                                                args["dataset"],
                                                str(args["max_length"]),
                                                "cold_data_list.json")))

    # Select the proper length
    for d in tqdm(datas):
        # Exclude the 5-fold used data
        if d in train_cold or d in train_hot:
            continue
        elif d["label"] == "hot":
            hot_data_list.append(d)
        elif d["label"] == "cold":
            cold_data_list.append(d)
    print("We have {} chromosome hot data avg:{}, {} chromosome cold data avg: {} ".format(
        len(hot_data_list) ,np.average([d["rate"] for d in hot_data_list]),
        len(cold_data_list),np.average([d["rate"] for d in cold_data_list])))

    if not os.path.exists(root_feature):
        os.makedirs(root_feature)
    print("=> saving {}".format(os.path.join(root_feature, "hot_data_list.json")))
    with open(os.path.join(root_feature, "hot_data_list.json"), 'w') as fw:
        js.dump(hot_data_list, fw)
    print("=> saving {}".format(os.path.join(root_feature, "cold_data_list.json")))
    with open(os.path.join(root_feature, "cold_data_list.json"), 'w') as fw:
        js.dump(cold_data_list, fw)

    print('=>generating Chromosome_2019 Hot data')
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
    print('=>generating Chromosome_2019 Cold data')
    with tqdm(total=len(cold_data_list)) as t:
        for x_cold, l_cold in pool.imap(one_hot_padding,
                                        [seq["seqence"] for seq in cold_data_list],
                                        [args["max_length"] for i in range(len(cold_data_list))]):
            X.append(x_cold)
            Y.append(0)
            L.append(l_cold)
            t.update()

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

def Nature_Genetics_2008_generator(args,data_path, root_feature):
    if not os.path.exists(root_feature):
        os.makedirs(root_feature)
    hot_sequences = load_fasta_gz(os.path.join(data_path,"hotspots.fasta.gz"))
    cold_sequences = load_fasta_gz(os.path.join(data_path,"coldspots.fasta.gz"))

    hot_data_list = [ {"seqence": h} for h in hot_sequences]
    cold_data_list = [ {"seqence": c} for c in cold_sequences]

    print("=> saving {}".format(os.path.join(root_feature, "hot_data_list.json")))
    with open(os.path.join(root_feature, "hot_data_list.json"), 'w') as fw:
        js.dump(hot_data_list, fw)
    print("=> saving {}".format(os.path.join(root_feature, "cold_data_list.json")))
    with open(os.path.join(root_feature, "cold_data_list.json"), 'w') as fw:
        js.dump(cold_data_list, fw)

    print('=>generating recomb_data')
    cores = pathos.multiprocessing.cpu_count()
    pool = Pool(processes=int(cores))
    X, Y ,L= [], [], []
    with tqdm(total=len(hot_data_list)) as t:
        for x_hot,len_seq in pool.imap(one_hot_padding,
                                       [seq["seqence"] for seq in hot_data_list],
                                       [args["max_length"] for seq in hot_data_list]):
            X.append(x_hot)
            Y.append(1)
            L.append(len_seq)
            t.update()
    with tqdm(total=len(cold_data_list)) as t:
        for x_cold,len_seq in pool.imap(one_hot_padding,
                                        [seq["seqence"] for seq in cold_data_list],
                                       [args["max_length"] for seq in cold_data_list]):
            X.append(x_cold)
            Y.append(0)
            L.append(len_seq)
            t.update()

    X = np.array(X)  # to make the pooling symmetric
    Y = np.array(Y)
    L = np.array(L)

    print(root_feature)
    print('saving => {}'.format(os.path.join(root_feature, 'X_train.npy')))
    np.save(os.path.join(root_feature, 'X.npy'), X)
    print('saving => {}'.format(os.path.join(root_feature, 'Y_train.npy')))
    np.save(os.path.join(root_feature, 'Y.npy'), Y)
    print('saving => {}'.format(os.path.join(root_feature, 'L_train.npy')))
    np.save(os.path.join(root_feature, 'L.npy'), L)

    return X, Y, L

def _26_Population_generator(args,root_feature):
    Population_dic = {
        "AFR": ['ASW', 'ACB', 'LWK', 'GWD', 'ESN', 'YRI', 'MSL'],
        "EAS": ['CHS', 'JPT', 'CHB', 'KHV', 'CDX'],
        "SAS": ['BEB', 'STU', 'GIH', 'ITU', 'PJL'],
        "EUR": ['TSI', 'FIN', 'GBR', 'CEU', 'IBS'],
        "AMR": ['PUR', 'CLM', 'PEL', 'MXL']
    }
    population_list = Population_dic[args["dataset"]]

    hot_data_list, cold_data_list = [], []
    for population in population_list:
        pop = js.load(open(os.path.join(data_root,"26population", "{}.json".format(population))))
        h = [d for d in pop if d["Label"] == "hot"]
        c = [d for d in pop if d["Label"] == "cold"]
        hot_data_list +=h
        cold_data_list +=c

    if not os.path.exists(root_feature):
        os.makedirs(root_feature)
    print("=> saving {}".format(os.path.join(root_feature, "hot_data_list.json")))
    with open(os.path.join(root_feature, "hot_data_list.json"), 'w') as fw:
        js.dump(hot_data_list, fw)
    print("=> saving {}".format(os.path.join(root_feature, "cold_data_list.json")))
    with open(os.path.join(root_feature, "cold_data_list.json"), 'w') as fw:
        js.dump(cold_data_list, fw)
    # Label the training data
    print("We have {} hot data, {} cold data ".format(len(hot_data_list), len(cold_data_list)))

    print('=>generating Human_Science Hot data')
    cores = pathos.multiprocessing.cpu_count()
    pool = Pool(processes=int(cores))
    X, Y, L = [], [], []
    with tqdm(total=len(hot_data_list)) as t:
        for x_hot, l_hot in pool.imap(one_hot_padding,
                                      [seq["sequence"] for seq in hot_data_list],
                                      [args["max_length"] for i in range(len(hot_data_list))]):
            X.append(x_hot)
            Y.append(1)
            L.append(l_hot)
            t.update()
    print('=>generating Human_Science Cold data')
    with tqdm(total=len(cold_data_list)) as t:
        for x_cold, l_cold in pool.imap(one_hot_padding,
                                        [seq["sequence"] for seq in cold_data_list],
                                        [args["max_length"] for i in range(len(cold_data_list))]):
            X.append(x_cold)
            Y.append(0)
            L.append(l_cold)
            t.update()

    # Do not close pool for other data generator
    pool.close()
    pool.join()

    X = np.array(X)
    Y = np.array(Y)
    L = np.array(L)

    print(root_feature)
    print('saving => {}'.format(os.path.join(root_feature, 'X.npy')))
    np.save(os.path.join(root_feature, 'X.npy'), X)
    print('saving => {}'.format(os.path.join(root_feature, 'Y.npy')))
    np.save(os.path.join(root_feature, 'Y.npy'), Y)
    print('saving => {}'.format(os.path.join(root_feature, 'L.npy')))
    np.save(os.path.join(root_feature, 'L.npy'), L)

    return X, Y, L

def Yeast_2008_generator(args,root_feature):
    hot_data_list = js.load(open(os.path.join(root_feature, "hot_data_list.json")))
    cold_data_list = js.load(open(os.path.join(root_feature, "cold_data_list.json")))

    # Label the training data
    print("We have {} hot data, {} cold data ".format(len(hot_data_list), len(cold_data_list)))
    print('=>generating Yeast 2008 Hot data')
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
    print('=>generating Yeast 2008 Cold data')
    with tqdm(total=len(cold_data_list)) as t:
        for x_cold, l_cold in pool.imap(one_hot_padding,
                                        [seq["seqence"] for seq in cold_data_list],
                                        [args["max_length"] for i in range(len(cold_data_list))]):
            X.append(x_cold)
            Y.append(0)
            L.append(l_cold)
            t.update()

    # Do not close pool for other data generator
    pool.close()
    pool.join()

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

def Split_Hot_Cold_Data(args,data_raw,num_hot,num_cold,interval=25):
    # Length Control
    datas = []
    # Save data with proper length
    for d in data_raw:
        if d["length"]< args["min_length"] or d["length"] > args["max_length"]:
            continue
        else:
            datas.append(d)
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
    hot_data_list_raw,cold_data_list_raw = [],[]
    hot_data_list = []
    datas = sorted(datas, key=lambda d: d["rate"])
    for d in datas:
        if d["label"] == "hot":
            hot_data_list_raw.append(d)

    random.shuffle(hot_data_list_raw)
    for i in range(int(num_hot)):
        hot_data_list.append(hot_data_list_raw[i])

    cold_data_list_raw = datas[:int(0.5 * len(datas))]
    cold_data_list = []
    print("Sorting Hot Data")
    with tqdm(total=len(hot_data_list)) as t:
        for hot_candidate in hot_data_list:
            length = hot_candidate["length"]
            idx = int((length - args["min_length"]-1)/interval)
            start_idx = int(args["min_length"] + idx * interval)
            end_idx = int(args["min_length"] + (idx + 1) * interval)
            dic["{}-{}".format(start_idx, end_idx)] +=1
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
    return hot_data_list,cold_data_list

if __name__ == '__main__':
#     args = {
#   "dataset":        "nature_2020",
#   "num_data":       10000,
#   "max_length":     1000,
#   "min_length":     500,
#   "model":          "SeqModel_Attention",
#   "log_dir":        "logs",
#   "output_dir":     "output",
#   "output_prefix":  "SeqModel_Attention",
#
#   "input_length":1000,
#   "input_height":4,
#   "data_augmentation":1,
#   "hidden_dim":64,
#   "attention":1,
#   "equivariant":0,
#   "get_distribution":0,
#   "get_line_comparison":1,
#
#   "dp_rate":0.1,
#
#   "trails":4,
#   "folds":5,
#   "epochs":100,
#   "batch_size":64,
#   "lr":0.001,
#
#   "init_weight":0,
#   "reinforce_train":0,
#   "loss":"binary_crossentropy"
# }
    summary_path = os.path.join(data_root,"nature2020", 'nature2020.json')
    # root_feature = os.path.join(dataset_root, args["dataset"],
    #                             str(args["input_length"]) + "_" + str(args["input_height"]))
    # Nature_2020_generator(args, summary_path, root_feature)
    raw_datas = js.load(open(summary_path, 'r'))

    num_hot, num_cold = 0,0
    hot_rate, cold_rate = 0,0
    for d in tqdm(raw_datas):
        if d["rate"]>10:
            num_hot +=1
            hot_rate +=d["rate"]
        elif d["rate"] < 0.5:
            num_cold +=1
            cold_rate += d["rate"]
    print("num hot : {} num cold {}".format(num_hot,num_cold))
    print("hot_rate : {} cold_rate {}".format(hot_rate/num_hot,cold_rate/num_cold))
