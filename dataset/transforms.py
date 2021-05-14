import numpy as np
import random
from sklearn.model_selection import train_test_split
one_hot_conv = {"A": [1, 0, 0, 0], "T": [0, 0, 0, 1],
                "C": [0, 1, 0, 0], "G": [0, 0, 1, 0],
                "a": [1, 0, 0, 0], "t": [0, 0, 0, 1],
                "c": [0, 1, 0, 0], "g": [0, 0, 1, 0],
                "n": [0, 0, 0, 0], "N": [0, 0, 0, 0]}
capital ={"A": "A", "T": "T",
          "C": "C", "G": "G",
          "a": "A", "t": "T",
          "c": "C", "g": "G"}
from scipy import signal

def cropping(inpt,len_seq,random_seed):
    out = np.zeros(inpt.shape)
    end_idx = random.randint(len_seq-random_seed, len_seq)
    out[0:end_idx,:] = inpt[0:end_idx,:]
    return out

mutation_dic = {
    "1":[1,0,0,0],
    "2":[0,1,0,0],
    "3":[0,0,1,0],
    "4":[0,0,0,1]
}

def mutation(inpt,len_seq,rate):
    out = inpt.copy()
    num_mutation = int(len_seq * rate)
    mutate_spot = np.random.randint(len_seq,size=num_mutation)
    for spot in mutate_spot:
        into_ = random.randint(1,4)
        out[spot,:] = mutation_dic[str(into_)]
    return out

def one_hot_padding(sequence,out_length):
    output = np.zeros((out_length,4))
    temp = np.array([np.array(one_hot_conv[base], dtype=np.float) for base in sequence] )
    len_seq = temp.shape[0]
    if len_seq > out_length:
        output[:, :] = temp[0:out_length, :]
    else:
        output[0:len_seq,:]=temp
    return output , len_seq
import gzip as gz
def load_fasta_gz(f_name):
    sequences = []
    cur_string = ""
    s = 0
    with gz.open(f_name) as fasta_file:
        for line in fasta_file:
            line = line.decode("ascii")
            if line[0] == '>':
                s+=1
                if cur_string:
                    assert len(cur_string) ==1000
                    sequences.append(cur_string)

                cur_string = ""
            else:
                line = line.strip()
                cur_string += line

        assert len(cur_string) ==1000
        sequences.append(cur_string)


    return sequences
