# @Author  : Yu Li & Siyuan Chen
# @Software: PyCharm
import json as js
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
def Classification_Accuracy(result,logger):
    # data is a list , containing the output of 1 trail
    # len(data) = 1
    data = result
    TP, TN, FP, FN = 0, 0, 0, 0
    for i_epoch in range(len(data)):
        # This loop will only be acted once
        gt_epoch = data[i_epoch][0]
        pred_epoch = data[i_epoch][1]
        final_pred = [0 for i in range(len(pred_epoch))]

        for i_p, p in enumerate(pred_epoch):
            if p[0] > p[1]:
                final_pred[i_p] = 0
            else:
                final_pred[i_p] = 1

        for i_test in tqdm(range(len(gt_epoch))):
            gt = int(gt_epoch[i_test])
            pred = int(final_pred[i_test])
            if gt == 1 and pred == 1:
                TP += 1
            elif gt == 0 and pred == 0:
                TN += 1
            elif gt == 0 and pred == 1:
                FP += 1
            elif gt == 1 and pred == 0:
                FN += 1


    Sn, Sp, Acc, MCC , Recall, Precision, F1 = Get_Accuracy(TP, TN, FP, FN)
    if logger is not None:
        logger.info('***|\tTP\t| |\tTN\t| |\tFP\t| |\tFN\t|***')
        logger.info('***|\t{}\t| |\t{}\t| |\t{}\t| |\t{}\t|***'.format(TP,TN,FP,FN))
        logger.info('***|\tSn\t| |\tSp\t| |\tAcc\t| |\tMcc\t|***')
        logger.info('***|\t{}%\t| |\t{}%\t| |\t{}%\t| |\t{}%\t|***'.format(
            round(Sn*100,2), round(Sp*100,2),
            round(Acc*100,2), round(MCC*100,2)))
        logger.info('***|\tRec\t| |\tPrec\t| |\tF1\t|***')
        logger.info('***|\t{}%\t| |\t{}%\t| |\t{}%\t|***'.format(
            round(Recall * 100, 2), round(Precision * 100, 2),
            round(F1 * 100, 2)))
    else:
        print('***|\tTP\t| |\tTN\t| |\tFP\t| |\tFN\t|***')
        print('***|\t{}\t| |\t{}\t| |\t{}\t| |\t{}\t|***'.format(TP, TN, FP, FN))
        print('***|\tSn\t| |\tSp\t| |\tAcc\t| |\tMcc\t|***')
        print('***|\t{}%\t| |\t{}%\t| |\t{}%\t| |\t{}%\t|***'.format(
            round(Sn * 100, 2), round(Sp * 100, 2),
            round(Acc * 100, 2), round(MCC * 100, 2)))
        print('***|\tRec\t| |\tPrec\t| |\tF1\t|***')
        print('***|\t{}%\t| |\t{}%\t| |\t{}%\t|***'.format(
            round(Recall * 100, 2), round(Precision * 100, 2),
            round(F1 * 100, 2)))

    return Sn , Sp, Acc, MCC, Recall, Precision, F1

def Classification_Accuracy_Distribution(args,result_filename,logger,interval=10):
    data = js.load(open(result_filename, 'r'))
    Sn_list, Sp_list, Acc_list,Mcc_list,\
    Recall_list,Precision_list,F1_list = [],[],[],[],[],[],[]
    eval_list = []
    num_dic = int((args["max_length"] - args["min_length"])/interval)
    cor_data_list,cor_acc_list = [],[]

    # In case that the data is not variant in length
    if num_dic == 0:
        eval_list.append({
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,
            "Sn": 0,
            "Sp": 0,
            "Acc": 0,
            "Mcc": 0,
            "Recall": 0,
            "Precision": 0,
            "F1": 0
        })
    else:
        for i in range(num_dic):
            eval_list.append({
            "TP":       1,
            "TN":       1,
            "FP":       1,
            "FN":       1,
            "Sn":       1,
            "Sp":       1,
            "Acc":      1,
            "Mcc":      1,
            "Recall":   0,
            "Precision":0,
            "F1":       0
        })

    for i_epoch in range(len(data)):
        # Reset TP , TN ,FP, FN in each trail
        TP, TN, FP, FN = 0, 0, 0, 0
        gt_epoch = data[i_epoch][0]
        pred_epoch = data[i_epoch][1]
        length_epoch = data[i_epoch][2]

        final_pred = [0 for i in range(len(pred_epoch))]
        for i_p, p in enumerate(pred_epoch):
            if p[0] > p[1]:
                final_pred[i_p] = 0
            else:
                final_pred[i_p] = 1

        for i_test in range(len(gt_epoch)):

            gt = int(gt_epoch[i_test])
            pred = int(final_pred[i_test])
            length_idx = int((length_epoch[i_test]-args["min_length"]-1)/interval)

            if gt == 1 and pred == 1:
                TP += 1
                eval_list[length_idx]["TP"]+=1

            elif gt == 0 and pred == 0:
                TN += 1
                eval_list[length_idx]["TN"] += 1

            elif gt == 0 and pred == 1:
                FP += 1
                eval_list[length_idx]["FP"] += 1

            elif gt == 1 and pred == 0:
                FN += 1
                eval_list[length_idx]["FN"] += 1
        Sn, Sp, Acc, Mcc, Recall, Precision, F1 = Get_Accuracy(TP, TN, FP, FN)
        Sn_list.append(Sn)
        Sp_list.append(Sp)
        Acc_list.append(Acc)
        Mcc_list.append(Mcc)
        Recall_list.append(Recall)
        Precision_list.append(Precision)
        F1_list.append(F1)

    for i in range(len(eval_list)):
        eval_list[i]["Sn"],\
        eval_list[i]["Sp"],\
        eval_list[i]["Acc"],\
        eval_list[i]["Mcc"], \
        eval_list[i]["Recall"],\
        eval_list[i]["Precision"],\
        eval_list[i]["F1"]= Get_Accuracy(eval_list[i]["TP"],
                                          eval_list[i]["TN"],
                                          eval_list[i]["FP"],
                                          eval_list[i]["FN"])
        cor_data_list.append(int(eval_list[i]["TP"]+eval_list[i]["TN"]+eval_list[i]["FP"]+eval_list[i]["FN"]))
        cor_acc_list.append(eval_list[i]["Acc"])
    cor_data_list = np.array(cor_data_list)
    cor_acc_list= np.array(cor_acc_list)
    if len(cor_acc_list)<2 or len(cor_data_list)<2:
        correlation, p = 0,0
    else:
        correlation, p = stats.pearsonr(cor_acc_list, cor_data_list)
    #输出：r： 相关系数 [-1，1]之间，p-value: p值。
    #注：p值越小，表示相关系数越显著，一般p值在500个样本以上时有较高的可靠性。
    Sn_avg = np.mean(np.array(Sn_list))
    Sp_avg = np.mean(np.array(Sp_list))
    Acc_avg= np.mean(np.array(Acc_list))
    MCC_avg= np.mean(np.array(Mcc_list))
    Recall_avg= np.mean(np.array(Recall_list))
    Precision_avg= np.mean(np.array(Precision_list))
    F1_avg = np.mean(np.array(F1_list))


    logger.info("*******************Final Result**************************")
    logger.info('***|\tSn\t| |\tSp\t| |\tAcc\t| |\tMcc\t|***')
    logger.info('***|\t{}%\t| |\t{}%\t| |\t{}%\t| |\t{}%\t|***'.format(
        round(Sn_avg * 100, 2), round(Sp_avg * 100, 2),
        round(Acc_avg * 100, 2), round(MCC_avg * 100, 2)))
    logger.info('***|\tRec\t| |\tPrec\t| |\tF1\t|***')
    logger.info('***|\t{}%\t| |\t{}%\t| |\t{}%\t|***'.format(
        round(Recall_avg * 100, 2), round(Precision_avg * 100, 2),
        round(F1_avg * 100, 2)))
    logger.info('***|\tCor\t| |\tP\t|***')
    logger.info('***|\t{}\t| |\t{}|***'.format(
        round(correlation,2),round(p,2)))

    return np.array(Sn_list),np.array(Sp_list),np.array(Acc_list),\
           np.array(Mcc_list),np.array(Recall_list),\
           np.array(Precision_list),np.array(F1_list), \
           eval_list

def AU_ROC(logger,Y_test, Y_pred):
    TP, TN, FP, FN = 0, 0, 0, 0
    predictions = []
    for i in range(Y_pred.shape[0]):
        if Y_pred[i][0] > Y_pred[i][1]:
            pred = 0
        else:
            pred = 1
        predictions.append(Y_pred[i][1])
        gt = Y_test[i]

        if gt == 1 and pred == 1:
            TP += 1
        elif gt == 0 and pred == 0:
            TN += 1
        elif gt == 0 and pred == 1:
            FP += 1
        elif gt == 1 and pred == 0:
            FN += 1
    logger.info('***|\tTP\t| |\tTN\t| |\tFP\t| |\tFN\t|***')
    logger.info('***|\t{}\t| |\t{}\t| |\t{}\t| |\t{}\t|***'.format(TP, TN, FP, FN))
    predictions = np.array(predictions)
    au_roc = roc_auc_score(Y_test, predictions)
    logger.info("AUC Score : {}".format(au_roc))
    fpr_skl, tpr_skl, thresholds_skl = roc_curve(Y_test, predictions, drop_intermediate=False)
    skl_result = [fpr_skl, tpr_skl, au_roc]
    return au_roc, skl_result

def Get_Accuracy(TP, TN, FP, FN):
    Sn = TP / (TP + FN)
    Sp = TN / (TN + FP)
    Acc = (TP + TN) / (TP + TN + FP + FN)
    if (TP + FP)==0 or (TP + FN)==0 or (TN + FP)==0 or (TN + FN)==0:
        MCC = -1
    else:
        MCC = ((TP * TN) - (FP * FN)) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    Recall = TP / (TP + FN)

    if (TP + FP)==0:
        Precision = 0
    else:
        Precision = TP / (TP + FP)
    if Precision==0 and Recall == 0:
        F1 = 0
    else:
        F1 = 2 * (Precision * Recall)/(Precision + Recall)

    return Sn, Sp, Acc, MCC, Recall, Precision, F1

def Log_AVG_k_folds(Sn_list, Sp_list, Acc_list,Mcc_list,Recall_list,Precision_list,F1_list,logger):
    Sn_avg = np.mean(np.array(Sn_list))
    Sp_avg = np.mean(np.array(Sp_list))
    Acc_avg = np.mean(np.array(Acc_list))
    MCC_avg = np.mean(np.array(Mcc_list))
    Recall_avg = np.mean(np.array(Recall_list))
    Precision_avg = np.mean(np.array(Precision_list))
    F1_avg = np.mean(np.array(F1_list))

    logger.info('***|\tSn_avg\t| |\tSp_avg\t| |\tAcc_avg\t| |\tMCC_avg\t|***')
    logger.info('***|\t  {}%\t| |\t  {}%\t| |\t  {}%\t| |\t  {}%\t|***'.format(
        round(Sn_avg * 100, 2), round(Sp_avg * 100, 2),
        round(Acc_avg * 100, 2), round(MCC_avg * 100, 2)))
    logger.info('***|\tRecall_avg\t| |\tPrecision_avg\t| |\tF1_avg\t|***')
    logger.info('***|\t  {}%\t| |\t  {}%\t| |\t  {}%\t|***'.format(
        round(Recall_avg * 100, 2), round(Precision_avg * 100, 2),
        round(F1_avg * 100, 2)))
