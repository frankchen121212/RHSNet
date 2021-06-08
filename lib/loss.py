# @Author  : Yu Li & Siyuan Chen
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import keras.backend as K
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def get_loss_function(args):
    if args["loss"] == "binary_crossentropy":
        loss = "binary_crossentropy"
    elif args["loss"] == "focal_loss":
        loss = focal_loss(gamma=4.0, alpha=0.25)
    else:
        loss = "categorical_crossentropy"
    return loss

if __name__ == '__main__':
    Y_true = np.array([[0,1],[1,0],[1,0],[1,0]])
    Y_pred = np.array([[0.9,0.1],[0.1,0.9],[0.1,0.9],[0.1,0.9]])
    Y_true = K.variable(Y_true)
    Y_pred = K.variable(Y_pred)
    print(focal_loss(Y_true, Y_pred))

