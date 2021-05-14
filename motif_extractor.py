"""
Usage: motif_extractor <json_file>
"""


import scipy.signal as signal
from lib.classification_models import *
from tqdm import tqdm
from collections import OrderedDict
import deeplift.conversion.kerasapi_conversion as kc
from deeplift.layers import NonlinearMxtsMode
from deeplift.visualization.viz_sequence import *
import os,docopt
import json

root = os.path.dirname(os.path.abspath(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def one_hotidx_2_base(idx):
    if idx == 0 :
        return "A"
    elif idx == 1 :
        return "C"
    elif idx == 2 :
        return "G"
    elif idx == 3 :
        return "T"

def one_hot2_seq(one_hot_array):
    seq = ""
    _,non_zero_idx = np.nonzero(one_hot_array)
    seq = seq.join([one_hotidx_2_base(base) for base in non_zero_idx])
    return seq

default_colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
default_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_t}
def save_plot_weights(array,
                 figsize=(20,2),
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=1.0,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={},
                 enriched_factor = 0,
                 picname = None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plot_weights_given_ax(ax=ax, array=array,
        height_padding_factor=height_padding_factor,
        length_padding=length_padding,
        subticks_frequency=subticks_frequency,
        colors=colors,
        plot_funcs=plot_funcs,
        highlight=highlight)
    plt.title('Enriched Factor :{}'.format(enriched_factor), fontsize='large', fontweight = 'bold')
    plt.savefig(picname)

def get_rate(args,rate_mat_file):
    rate_folder_name = os.path.join(root, 'dataset',
                                    args["dataset"],
                                    str(args["input_length"])
                                    )
    data = json.load(open(os.path.join(rate_folder_name,"hot_data_list.json")))
    rate_mat = []

    if len(args["dataset"]) == 3:
        char_rate = "Rate"
    elif args["dataset"] == "mouse_cell_2016":
        char_rate = "BPM"
    else:
        char_rate = "rate"

    if args["dataset"] in ["nature_genetics_2008","yeast_2008" ]:

        for i in range(len(data)):
            rate_mat.append(10)
    else:
        for d in data:
            rate_mat.append(d[char_rate])


    np.save(rate_mat_file, np.array(rate_mat))
    return np.array(rate_mat)


def get_score(args,score_mat_file):
    data_folder_name = os.path.join(root, 'dataset',
                                    args["dataset"],
                                    str(args["input_length"])
                                    )


    data = np.load(os.path.join(data_folder_name,"X.npy"))
    print("=> {} loaded ".format(os.path.join(root,data_folder_name,"X.npy")))
    Y = np.load(os.path.join(data_folder_name,"Y.npy"))


    num_hot = np.count_nonzero(Y, axis=None)

    hot_data_ont_hot = np.array(data[0:num_hot])


    model_file_name = os.path.join(root, 'model',
                                   args["dataset"],
                                   args["model"],
                                   str(args["input_length"]) + "_" + str(args["input_height"]),
                                   args["output_prefix"]
                                   )

    keras_model_weights = model_file_name + ".h5"
    keras_model_yaml = model_file_name + ".yaml"
    keras_model_json = model_file_name + ".json"

    model = CNN(epochs=args["epochs"], args=args, loss="binary_crossentropy")
    model.load_weights(keras_model_weights)

    # print("=> {} loaded ".format(keras_model_json))
    # deeplift_model = kc.convert_model_from_saved_files(
    #     h5_file=keras_model_weights,
    #     json_file=keras_model_json)

    guided_backprop = kc.convert_model_from_saved_files(
        h5_file=keras_model_weights,
        json_file=keras_model_json,
        nonlinear_mxts_mode=NonlinearMxtsMode.GuidedBackprop)
    guided_backprop_scoring_func = guided_backprop.get_target_contribs_func(find_scores_layer_idx=0,
                                                                            target_layer_idx=-2)

    method_to_task_to_scores = OrderedDict()
    method_name = "guided_backprop"
    score_func = guided_backprop_scoring_func
    print("on method", method_name)
    method_to_task_to_scores[method_name] = OrderedDict()
    background = OrderedDict([('A', 0.3), ('C', 0.2), ('G', 0.2), ('T', 0.3)])

    for task_idx in [0,1]:
        scores = np.array(score_func(
            task_idx=task_idx,
            input_data_list=[hot_data_ont_hot],
            input_references_list=[
                np.array([background['A'],
                          background['C'],
                          background['G'],
                          background['T']])[None, None, :]],
            batch_size=200,
            progress_update=None))

    for task_idx in [0,1]:
        scores = np.array(score_func(
            task_idx=task_idx,
            input_data_list=[hot_data_ont_hot],
            input_references_list=[
                np.array([background['A'],
                          background['C'],
                          background['G'],
                          background['T']])[None, None, :]],
            batch_size=200,
            progress_update=None))
        assert scores.shape[2] == 4
        scores = np.sum(scores, axis=2)
        method_to_task_to_scores[method_name][task_idx] = scores


    score_mat = []
    task_idx = [(0, seq) for seq in range(num_hot)]

    for task, idx in tqdm(task_idx):

        method_name = 'guided_backprop'
        scores = method_to_task_to_scores[method_name][task]


        scores_for_idx = scores[idx]
        original_onehot = hot_data_ont_hot[idx]
        scores_for_idx = original_onehot * scores_for_idx[:, None]
        score_mat.append(scores_for_idx)
    np.save(score_mat_file, np.array(score_mat))
    return np.array(score_mat)

def Sliding_Window(scores_for_idx, k):

    top_score = 0
    # Sliding window
    motif = {
        "k_mer": np.zeros((k,4)),  # matrix
        "sequence": "",  # string
        "length": k,
        "start": 0,
        "end": 0,
        "score": 0,
        "count": 0
    }
    for i in range(0, 1000 - k):
        score = np.average(scores_for_idx[i:i + k])
        if score > top_score:
            top_score = score
            motif["k_mer"] = scores_for_idx[i:i + k]
            motif["score"] = top_score
            motif["start"] = int(i)
            motif["end"] = int(i + k)
            motif["sequence"] = one_hot2_seq(scores_for_idx[i:i + k])

    return motif


def Low_pass_filter_extractor(scores_for_idx, recomb_rate, filter_factor):
    # inpt.shape [1000,4]


    motif_list = []
    sample_non_zero_idx, sample_non_zero_ATCG = scores_for_idx.nonzero()
    sample_signal = np.zeros((1000,))

    for i in range(len(sample_non_zero_idx)):
        pos = sample_non_zero_idx[i]
        sample_signal[pos] = scores_for_idx[pos][sample_non_zero_ATCG[i]]

    b, a = signal.butter(8, filter_factor, 'lowpass')
    filtered_signal = signal.filtfilt(b, a, sample_signal)
    # inpt.shape [1000]

    peaks, _ = signal.find_peaks(filtered_signal, prominence=0.06)
    valleys, _ = signal.find_peaks(filtered_signal * (-1), width=1)
    start_idx_list = []
    end_idx_list = []
    for peak in peaks:
        reverse_peak = np.array([valleys[v] - peak for v in range(len(valleys))])
        start = valleys[np.where(reverse_peak < 0, reverse_peak, -np.inf).argmax()]
        end = valleys[np.where(reverse_peak > 0, reverse_peak, np.inf).argmin()]
        start_idx_list.append(start)
        end_idx_list.append(end)

    for i in range(len(start_idx_list)):
        motif = {
            "k_mer":scores_for_idx[start_idx_list[i]:end_idx_list[i]],  # matrix
            "sequence": "",  # string
            "recomb_rate":recomb_rate,
            "length": 0,
            "start": 0,
            "end": 0,
            "score": 0,
            "enriched_factor":0,
            "count": 0,
        }

        motif["start"] = start_idx_list[i]
        motif["end"] = end_idx_list[i]
        if motif["end"] < motif["start"]:
            print("Wrong peak, Continue")
            continue
        motif["score"] = np.average(sample_signal[motif["start"]:motif["end"]])
        motif["enriched_factor"] = motif["score"]/np.average(sample_signal)
        motif["length"] = int(end_idx_list[i] - start_idx_list[i])
        motif["sequence"] = one_hot2_seq(scores_for_idx[start_idx_list[i]:end_idx_list[i]])
        motif_list.append(motif)

    return motif_list

def count_alg(motif, sequence_list):
    count = 0
    for sequence in sequence_list:
        count += sequence.count(motif)

    return count

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    args = json.load(open(args["<json_file>"]))

    # Do counting
    target_length = 13
    result_root = os.path.join(root,
                               "motifs",
                               args["dataset"]
                               )
    if not os.path.exists(result_root):
        os.makedirs(result_root)



    # Get scores
    score_mat_file = os.path.join(result_root,"scores.npy")
    rate_mat_file = os.path.join(result_root,"recomb_rate.npy")

    if not os.path.exists(rate_mat_file):
        recomb_rates = get_rate(args,rate_mat_file)
    else:
        recomb_rates = np.load(rate_mat_file)


    if not os.path.exists(score_mat_file):
        scores = get_score(args,score_mat_file)
    else:
        scores = np.load(score_mat_file)



    data_folder_name = os.path.join(root, 'dataset',
                                    args["dataset"],
                                    str(args["input_length"])
                                    )


    labels = np.load(os.path.join(data_folder_name,"Y.npy"))

    num_data = np.count_nonzero(labels, axis=None)

    task_idx = [(0, seq) for seq in range(num_data)]
    # Extract Motifs

    # filter_list = [0.1]
    if "_2014" in args["dataset"]:
        filter_list = [0.4, 0.2, 0.1]
        num_save = 20
    else:
        num_save = 500
        filter_list = [0.4, 0.2, 0.1]

    for filter_v in filter_list:
        motif_list = []
        for task, idx in tqdm(task_idx):
            scores_for_idx = scores[idx]
            recomb_rate = recomb_rates[idx]
            # scores_for_idx.shape =  [1000,4]
            # motif_list.append(Sliding_Window(scores_for_idx, target_length ))
            m_list = Low_pass_filter_extractor(scores_for_idx,
                                               recomb_rate,
                                               filter_factor = filter_v)
            for m in m_list:
                motif_list.append(m)

        motif_list = sorted(motif_list, key=lambda d: d["enriched_factor"], reverse=True)

        # Draw motif
        num_draw = 20
        for i in range(0, num_draw):
            pic_root = os.path.join(result_root,"filter{}".format(filter_v))
            if not os.path.exists(pic_root):
                os.mkdir(pic_root)
            print(motif_list[i]["sequence"])
            start = int(motif_list[i]["start"])
            end = int(motif_list[i]["end"])
            print("Whole Sequence ranking {}:".format(i + 1))
            #     highlight_seq = {'blue': [
            #         (start, end)]
            #     }
            save_plot_weights(motif_list[i]["k_mer"],
                              subticks_frequency=1,
                              enriched_factor=motif_list[i]["enriched_factor"],
                              picname= os.path.join(pic_root,'filter{}_rank_{}.jpg'.format(filter_v,i + 1)))
            print('=> {} saved'.format(os.path.join(pic_root,'filter{}_rank_{}.jpg'.format(filter_v,i + 1))))


        saved_file = os.path.join(result_root,'filter{}_motifs.json'.format(filter_v))
        save_list = []
        for i in range(0,num_save):
            motif_list[i]["rank"] = i+1
            save_list.append(motif_list[i])
        print('saving ==>{}'.format(saved_file))
        with open(saved_file, 'w') as fw:
            js.dump(save_list, fw,cls=NumpyEncoder, indent=4)

    #     save_plot_weights(sequence_list[i]["sequence"],
    #                       subticks_frequency=100,
    #                       highlight=highlight_seq,
    #                       confidence_score = sequence_list[i]["score"],
    #
    #                       picname = os.path.join(result_root,'sequence_rank_{}.jpg'.format(i + 1)))
    #
    #