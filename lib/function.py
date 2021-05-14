from datetime import datetime
from keras.callbacks import TensorBoard
from keras.models import model_from_yaml

def get_log(args):
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    log = TensorBoard(log_dir=os.path.join(args["log_dir"],
                                           args["dataset"],
                                           args["output_prefix"],
                                           str(args["input_length"]) +"_"+ str(args["input_height"]),
                                           TIMESTAMP),  # log dir
                      histogram_freq=0,
                      write_graph=True,
                      write_grads=True,
                      write_images=True,
                      embeddings_freq=0,
                      embeddings_layer_names=None,
                      embeddings_metadata=None)
    return log


def Save_Model(model,model_name,logger):
    with open(model_name + ".json", 'w') as j_file:
        j_file.write(model.to_json())
    logger.info('saving =>{}'.format(model_name + ".json"))
    with open(model_name+".yaml", 'w') as y_file:
        y_file.write(model.to_yaml())
    logger.info('saving =>{}'.format(model_name+".yaml"))
    model.save_weights(model_name+".h5")
    logger.info('saving =>{}'.format(model_name+".h5"))

#
# import pickle
# def Save_SVM_Model(model,model_name,logger):
#     output = open(model_name + ".pkl", 'wb')
#     s = pickle.dump(model, output)
#     output.close()
#     logger.info('saving =>{}'.format(model_name + ".pkl"))

import logging
import time
import os,json
from lib.paths import *
from lib.evaluation import *

def log_creater(output_dir,mode):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = mode+'_{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir,log_name)

    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log

def Get_time(logger,start_time,end_time,perfix):

    training_time = end_time - start_time
    training_hours = int(training_time / 3600)
    training_minutes = int(training_time / 60 - training_hours*60)
    training_seconds = int(training_time % 60)
    logger.info('{} time :\t{}h {} min {} s'.format(perfix,
                                                    training_hours,
                                                    training_minutes,
                                                    training_seconds))


def Log_num_data(logger,Y_train,Y_test):
    num_cold_train, num_hot_train = 0, 0
    if Y_train is not None:
        for y_t in Y_train:
            if y_t == 0:
                num_cold_train += 1
            else:
                num_hot_train += 1
    num_cold_test, num_hot_test = 0, 0
    if Y_test is not None:
        for y_t in Y_test:
            if y_t == 0:
                num_cold_test += 1
            else:
                num_hot_test += 1
    logger.info("num_hot_train--{}  num_cold_train--{} num_hot_test--{}  num_cold_test--{}".format(
        num_hot_train,
        num_cold_train,
        num_hot_test,
        num_cold_test))

    return (num_hot_train+num_hot_test), (num_cold_train+num_cold_test)

def Save_training_result(args,final_result,logger):
    final_result_filename = get_result_file_name(args)
    print('saving => {}'.format(final_result_filename))
    with open(final_result_filename, "w") as outfile:
        json.dump(final_result, outfile)
    if "mouse" in args["dataset"]:
        logger.info("*******************Final Result**************************")
        Sn , Sp, Acc, MCC, Recall, Precision, F1 = Classification_Accuracy(final_result, logger)
    else:
        Final_Sn, Final_Sp, Final_Acc, Final_Mcc, \
        Final_Recall, Final_Precision, Final_F1, _= Classification_Accuracy_Distribution(args=args,
                                                               result_filename=final_result_filename,
                                                               logger=logger,
                                                               interval=25)

def get_feature_layer_name(model):
    for layer in model.layers:
        if "global" in layer.name:
            return layer.name

from sklearn import preprocessing
def Normalization(X):
    scaler = preprocessing.StandardScaler().fit(X)
    X_normalized = scaler.transform(X)
    return X_normalized


