import tflearn
from lib.classification_models import *
from lib.function import *
from dataset.data_utils import *
from lib.CNN_core import *
from lib.Attention_core import *
from lib.evaluation import *
from lib.vis import *
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
def k_fold_train(args,logger,X,Y,L,best_F1_Score,final_result):
    kf = KFold(n_splits=args["folds"])
    Sn_list, Sp_list, Acc_list, Mcc_list, \
    Recall_list, Precision_list, F1_list = [], [], [], [], [], [], []
    fold = 1

    #Shuffle
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation, :, :]
    Y = Y[permutation]
    L = L[permutation]

    # K_fold train&validation
    for train_index, test_index in kf.split(X):
        # Split the data
        logger.info("\n=> Fold {}".format(fold))
        result = []
        X_train_raw, X_val, X_test,\
        Y_train_raw, Y_val, Y_test,\
        _, _, L_test = Split_Train_Val_Test_Set(X, Y, L,
                                               train_index,
                                                test_index)
        # Data augmentation
        if args["data_augmentation"]:
            if "science" in args["dataset"]:
                X_train, Y_train = science_2019_data_augmentation(X_train_raw, Y_train_raw, args)
            elif args["dataset"] == "nature_genetics_2008":
                X_train, Y_train = nature_2008_data_augmentation(X_train_raw, Y_train_raw, args)
            elif args["dataset"] == "nature_2020":
                X_train, Y_train = science_2019_data_augmentation(X_train_raw, Y_train_raw, args)
            elif args["dataset"] == "mouse_cell_2016":
                X_train, Y_train = science_2019_data_augmentation(X_train_raw, Y_train_raw, args)
            elif args["dataset"] == "yeast_2008":
                X_train, Y_train = science_2019_data_augmentation(X_train_raw, Y_train_raw, args)
            else:
                X_train, Y_train = X_train_raw, Y_train_raw
        else:
            X_train, Y_train = X_train_raw, Y_train_raw

        num_hot, num_cold = Log_num_data(logger=logger, Y_train=Y_train, Y_test=Y_test)

        # Metrixs & logs
        es = EarlyStopping(monitor='val_acc', patience=4)  # TODO 4
        log = get_log(args=args)
        loss = get_loss_function(args)

        # Model
        if args["model"] == 'CNN':
            model = CNN(epochs=args["epochs"], args=args, loss=loss)
        elif args["model"] == 'Equivariant_CNN':
            model = Equivariant_CNN(epochs=args["epochs"], args=args, loss=loss)
            mm_0 = MotifMirrorGradientBleeding(0, assign_bias=True)
            mm_1 = MotifMirrorGradientBleeding(2, assign_bias=True)
        elif "SeqModel" in args["model"]:
            model = SeqModel(epochs=args["epochs"], args=args, loss=loss)
        elif "Attention" in args["model"]:
            model = Attention(epochs=args["epochs"], args=args, loss=loss)
        # print(model.summary())

        # callbacks
        if args["equivariant"]:
            callbacks = [es, mm_0, mm_1, log]
        else:
            callbacks = [es, log]

        model.summary()
        if args["init_weight"]:
            init_weight_path = args["init_weight"]
            if not os.path.exists(init_weight_path):
                raise Exception("Init Weight Path:{} Not Exists!".format(init_weight_path))
            logger.info("loading weight from {}".format(init_weight_path))
            model.load_weights(init_weight_path)

        # Train
        if args["model"] != "Random_Guess":
            model.fit(X_train, tflearn.data_utils.to_categorical(Y_train, 2),
                      validation_data=(X_val, tflearn.data_utils.to_categorical(Y_val, 2)),
                      epochs=args["epochs"],
                      batch_size=args["batch_size"],
                      callbacks=callbacks)
            predictions = model.predict_mc(X_test, n_preds=4)
        else:
            predictions = Random_Guess(X_test)

        result.append((Y_test.tolist(), predictions.tolist(), L_test.tolist()))
        final_result.append((Y_test.tolist(), predictions.tolist(), L_test.tolist()))

        # Get Evaluation Results
        Sn, Sp, Acc, MCC, Recall, Precision, F1 = Classification_Accuracy(result, logger)
        Sn_list.append(Sn)
        Sp_list.append(Sp)
        Acc_list.append(Acc)
        Mcc_list.append(MCC)
        Recall_list.append(Recall)
        Precision_list.append(Precision)
        F1_list.append(F1)
        if F1 > best_F1_Score and args["model"] != "Random_Guess":
            best_F1_Score = F1
            model_file_name = get_model_file_name(args)
            Save_Model(model=model, model_name=model_file_name, logger=logger)
        fold += 1
    # Average k_fold results
    Log_AVG_k_folds(Sn_list, Sp_list, Acc_list, Mcc_list, Recall_list, Precision_list, F1_list, logger)
    return best_F1_Score, final_result

def reinforce_train(args,logger,
                    X, Y, L,
                    X_CHIP, Y_CHIP,
                    best_F1_Score,
                    final_result):
    kf = KFold(n_splits=args["folds"])
    Sn_list, Sp_list, Acc_list, Mcc_list, \
    Recall_list, Precision_list, F1_list = [], [], [], [], [], [], []
    fold = 1

    # Shuffle
    permutation = np.random.permutation(X.shape[0])
    X           = X[permutation, :, :]
    X_CHIP      = X_CHIP[permutation,:]
    Y           = Y[permutation]
    L           = L[permutation]

    # Normalization
    X_CHIP   = Normalization(X_CHIP)
    #Data
    for train_index, test_index in kf.split(X):
        logger.info("\n=> Fold {}".format(fold))
        result = []
        X_train_raw, X_val, X_test, \
        X_CHIP_train_raw, X_CHIP_val, X_CHIP_test, \
        Y_train_raw, Y_val, Y_test, \
        _, _, L_test = Split_multi_Train_Val_Test_Set(X,Y,L,
                                    X_CHIP,
                                    train_index,test_index)
        # Data augmentation
        if args["data_augmentation"]:
            X_train, X_CHIP_train, Y_train = data_augmentation(X_train_raw,
                                                             X_CHIP_train_raw,
                                                              Y_train_raw )
        else:
            X_train, X_CHIP_train, Y_train = X_train_raw,X_CHIP_train_raw, Y_train_raw
        num_hot, num_cold = Log_num_data(logger=logger, Y_train=Y_train, Y_test=Y_test)

        # Metrixs & logs
        es = EarlyStopping(monitor='val_acc', patience=4)  # TODO 4
        log = get_log(args=args)
        loss = get_loss_function(args)

        # Model
        model = Ensemble_SeqModel(epochs=args["epochs"],
                                  args=args,
                                  loss=loss)
        model.summary()
        # callbacks

        callbacks = [es, log]

        if args["init_weight"]:
            init_weight_path = args["init_weight"]
            if not os.path.exists(init_weight_path):
                raise Exception("Init Weight Path:{} Not Exists!".format(init_weight_path))
            logger.info("loading weight from {}".format(init_weight_path))
            model.load_weights(init_weight_path)

        x_train,x_val,x_test = None,None,None
        if args["chip_seq"]:
            x_train = [X_train, X_CHIP_train]
            x_val   = [X_val,   X_CHIP_val]
            x_test  = [X_test,  X_CHIP_test]


        model.fit(x=x_train,
                  y=tflearn.data_utils.to_categorical(Y_train, 2),
                  validation_data=(x_val,
                                   tflearn.data_utils.to_categorical(Y_val, 2)),
                  epochs=args["epochs"],
                  batch_size=args["batch_size"],
                  callbacks=callbacks)
        predictions = model.predict_mc(x_test, n_preds=4)
        result.append((Y_test.tolist(), predictions.tolist(), L_test.tolist()))
        final_result.append((Y_test.tolist(), predictions.tolist(), L_test.tolist()))

        # Get Evaluation Results
        Sn, Sp, Acc, MCC, Recall, Precision, F1 = Classification_Accuracy(result, logger)
        Sn_list.append(Sn)
        Sp_list.append(Sp)
        Acc_list.append(Acc)
        Mcc_list.append(MCC)
        Recall_list.append(Recall)
        Precision_list.append(Precision)
        F1_list.append(F1)
        if F1 > best_F1_Score and args["model"] != "Random_Guess":
            best_F1_Score = F1
            model_file_name = get_model_file_name(args)
            Save_Model(model=model, model_name=model_file_name, logger=logger)

        fold += 1
        # Average k_fold results
    Log_AVG_k_folds(Sn_list, Sp_list, Acc_list, Mcc_list, Recall_list, Precision_list, F1_list, logger)
    return best_F1_Score, final_result
