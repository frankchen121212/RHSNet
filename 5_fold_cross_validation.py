"""
Usage: train <json_file>
"""
# @Author  : Yu Li & Siyuan Chen
# @Software: PyCharm


import docopt
from dataset.CHIP_utils import CHIP_seq_hg38
from lib.core import *
from dataset.data_utils import _26_Population,Human_Science_2019,\
    Nature_Genetics_2008,Nature_2020,Mouse_Cell_2016
import tensorflow as tf
from keras import backend as K

root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, root)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def train(args, config):
    # Get the data
    X, Y, L = None, None, None
    if "science" in args["dataset"]:
        X, Y, L = Human_Science_2019(args)
    elif args["dataset"] == "nature_genetics_2008":
        X, Y, L = Nature_Genetics_2008(args)
    elif args["dataset"] == "nature_2020":
        X, Y, L = Nature_2020(args)
    elif args["dataset"] == "mouse_cell_2016":
        X, Y, L = Mouse_Cell_2016(args)
    elif len(args["dataset"])==3:
        X, Y, L = _26_Population(args)

    X = X [:, :args["input_length"]]

    best_F1_Score = 0
    final_result = []
    start_time = time.time()
    logger_path = get_logger_file_name(args)
    logger = log_creater(output_dir = logger_path, mode = config)
    logger.info('==>Params')
    for key in args.keys():
        logger.info('\t{}:{}\n'.format(key, args[key]))

    # Iterations
    for trail in range(args["trails"]):
        #Reset the memory
        K.clear_session()
        tf.reset_default_graph()
        trail_start_time = time.time()
        logger.info("\n => Trail {} \n".format(trail + 1))
        if not args["reinforce_train"]:
            best_F1_Score, final_result = k_fold_train(args,
                                        logger,
                                        X, Y, L,
                                        best_F1_Score,
                                        final_result)
        else:
            X_CHIPseq, Y_CHIPseq = CHIP_seq_hg38(args)
            best_F1_Score, final_result = reinforce_train(args,
                                        logger,
                                        X, Y, L,
                                        X_CHIPseq, Y_CHIPseq,
                                        best_F1_Score,
                                        final_result
                                        )

        Get_time(logger, start_time=trail_start_time, end_time=time.time(), perfix="Trail_{}".format(int(trail+1)))

    #Saving the final result for model comparison
    Save_training_result(args, final_result, logger)
    Get_time(logger,start_time=start_time,end_time = time.time(),perfix="Total")
    logger.info(time.strftime("Ending at %Y/%m/%d  %I:%M:%S"))


if __name__ == "__main__":
    args = docopt.docopt(__doc__)
    args_inpt = json.load(open(args["<json_file>"]))
    config = str(args["<json_file>"]).strip(".json")
    config = config.split("/")[-1]
    train(args_inpt, config)
