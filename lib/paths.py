import os,sys
root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, root)

def get_chromosome_img_root(args):
    chromosome_img_root = os.path.join(root,
                                      args["output_dir"],
                                      args["dataset"],
                                      "chromesome")
    if not os.path.exists(chromosome_img_root):
        os.makedirs(chromosome_img_root)
    return chromosome_img_root

def get_result_file_name(args):
    result_filename = os.path.join(root,
                                       args["output_dir"],
                                       args["dataset"],
                                       args["model"],
                                       str(args["input_length"]) + "_" + str(args["input_height"]),
                                       args["output_prefix"]+ ".json")
    if not os.path.exists(os.path.dirname(result_filename)):
        os.makedirs(os.path.dirname(result_filename))

    return result_filename

def get_model_file_name(args):
    model_file_name = (os.path.join(root,
                                    "model",
                                    args["dataset"],
                                    args["model"],
                                    str(args["input_length"]) + "_" + str(args["input_height"]),
                                    args["output_prefix"]))
    if not os.path.exists(os.path.dirname(model_file_name)):
        os.makedirs(os.path.dirname(model_file_name))
    return model_file_name


def get_eval_list_file_name(args):
    eval_list_file = os.path.join(root,
                                  args["output_dir"],
                                  args["dataset"],
                                  args["model"],
                                  str(args["input_length"]) + "_" + str(args["input_height"]),
                                  args["output_prefix"] + "_result.json")
    if not os.path.exists(os.path.dirname(eval_list_file)):
        os.makedirs(os.path.dirname(eval_list_file))
    return eval_list_file

def get_logger_file_name(args):
    logger_path = os.path.join(root,
                               args["output_dir"],
                               args["dataset"],
                               args["model"],
                               str(args["input_length"]) + "_" + str(args["input_height"]))
    if not os.path.exists(os.path.dirname(logger_path)):
        os.makedirs(os.path.dirname(logger_path))
    return logger_path

def get_final_result_root(args):
    result_root = os.path.join(root,
                               "result",
                               args["dataset"],
                               str(args["input_length"]) + "_" + str(args["input_height"]))
    if not os.path.exists(os.path.dirname(result_root)):
        os.makedirs(os.path.dirname(result_root))
    return result_root

def get_binding_sequence_file_name(args,k):
    binding_sequence_file = os.path.join(root,
                               "feature",
                               args["dataset"],
                               str(args["input_length"]) + "_" + str(args["input_height"]) + "_" + "{}-mer".format(k),
                               args["output_prefix"]+"_"+"motif.json")
    if not os.path.exists(os.path.dirname(binding_sequence_file)):
        os.makedirs(os.path.dirname(binding_sequence_file))
    return binding_sequence_file

def get_motif_frequency_file_name(args,k):
    motif_frequency_root = os.path.join(root,
                                      "feature",
                                      args["dataset"],
                                      str(args["input_length"]) + "_" + str(args["input_height"])+ "_" + "{}-mer".format(k),
                                      args["output_prefix"] + "_" + "motif_frequency.json")
    if not os.path.exists(os.path.dirname(motif_frequency_root)):
        os.makedirs(os.path.dirname(motif_frequency_root))
    return motif_frequency_root

def get_motif_fasta_file_root(args,k):
    motif_fasta_root = os.path.join(root,
                                      "feature",
                                      args["dataset"],
                                      str(args["input_length"]) + "_" + str(args["input_height"])+ "_" + "{}-mer".format(k),
                                      "fasta_{}".format(args["output_prefix"]))
    return motif_fasta_root