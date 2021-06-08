# @Author  : Yu Li & Siyuan Chen
# @Software: PyCharm

import keras.backend as K
from keras.engine.topology import Layer
import numpy as np
import tflearn
from dataset.data_utils import *

class Multiheads_Attention(Layer):

    def __init__(self, multiheads, head_dim, mask_right=False, **kwargs):
        """
        # Parameter：
        #    - multiheads: Number of Attention
        #    - head_dim: dimension of Attention Score
        #    - mask_right: Mask
        """
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        super(Multiheads_Attention, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1],
                self.output_dim)  # shape=[batch_size,Q_sequence_length,self.multiheads*self.head_dim]

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),  # input_shape[0] -> Q_seq
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),  # input_shape[1] -> K_seq
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),  # input_shape[2] -> V_seq
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Multiheads_Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='add'):
        """
        # Parameters：
        #    - inputs: sequence to Mask
        #    - seq_len: shape=[batch_size,1] or [batch_size,]
        #    - mode: way to mask.
                'mul'时返回的mask位置为0,
                'add'时返回的mask位置为一个非常大的负数，在softmax下为0。由于attention的mask是在softmax之前，所以要用这种方式执行
        """
        if seq_len == None:
            return inputs
        else:
            # seq_len[:,0].shape=[batch_size,1]
            # short_sequence_length=K.shape(inputs)[1]：较短的sequence_length，如K_sequence_length，V_sequence_length
            mask = K.one_hot(indices=seq_len[:, 0], num_classes=K.shape(inputs)[
                1])  # mask.shape=[batch_size,short_sequence_length],mask=[[0,0,0,0,1,0,0,..],[0,1,0,0,0,0,0...]...]
            mask = 1 - K.cumsum(mask,
                                axis=1)  # mask.shape=[batch_size,short_sequence_length],mask=[[1,1,1,1,0,0,0,...],[1,0,0,0,0,0,0,...]...]
            # 将mask增加到和inputs一样的维度，目前仅有两维[0],[1]，需要在[2]上增加维度
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            # mask.shape=[batch_size,short_sequence_length,1,1]
            if mode == 'mul':
                # Element-wise multiply：直接做按位与操作
                # return_shape = inputs.shape
                # 返回值：[[seq_element_1,seq_element_2,...,masked_1,masked_2,...],...]，其中seq_element_i,masked_i的维度均为2维
                # masked_i的值为0
                return inputs * mask
            elif mode == 'add':
                # Element-wise add：直接做按位加操作
                # return_shape = inputs.shape
                # 返回值：[[seq_element_1,seq_element_2,...,masked_1,masked_2,...],...]，其中seq_element_i,masked_i的维度均为2维
                # masked_i的值为一个非常大的负数，在softmax下为0。由于attention的mask是在softmax之前，所以要用这种方式执行
                return inputs - (1 - mask) * 1e12

    def call(self, QKVs):
        """
        # keras.engine.base_layer
        # 1. Q',K',V' = Q .* WQ_i,K .* WK_i,V .* WV_i
        # 2. head_i = Attention(Q',K',V') = softmax((Q' .* K'.T)/sqrt(d_k)) .* V
        # 3. MultiHead(Q,K,V) = Concat(head_1,...,head_n)
        # Params
            - QKVs：[Q_seq,K_seq,V_seq] or [Q_seq,K_seq,V_seq,Q_len,V_len]
                -- Q_seq.shape = [batch_size,Q_sequence_length,Q_embedding_dim]
                -- K_seq.shape = [batch_size,K_sequence_length,K_embedding_dim]
                -- V_seq.shape = [batch_size,V_sequence_length,V_embedding_dim]
                -- Q_len.shape = [batch_size,1],如：[[7],[5],[3],...]
                -- V_len.shape = [batch_size,1],如：[[7],[5],[3],...]
        #
            -
        """
        if len(QKVs) == 3:
            Q_seq, K_seq, V_seq = QKVs
            Q_len, V_len = None, None
        elif len(QKVs) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = QKVs
        # Q_seq.shape=[batch_size,Q_sequence_length,Q_embedding_dim]
        # self.WQ.shape=[Q_embedding_dim,self.output_dim]=[Q_embedding_dim,self.multiheads*self.head_dim]
        Q_seq = K.dot(Q_seq,
                      self.WQ)  # Q_seq.shape=[batch_size,Q_sequence_length,self.output_dim]=[batch_size,Q_sequence_length,self.multiheads*self.head_dim]
        Q_seq = K.reshape(Q_seq, shape=(-1, K.shape(Q_seq)[1], self.multiheads,
                                        self.head_dim))  # Q_seq.shape=[batch_size,Q_sequence_length,self.multiheads,self.head_dim]
        Q_seq = K.permute_dimensions(Q_seq, pattern=(
        0, 2, 1, 3))  # Q_seq.shape=[batch_size,self.multiheads,Q_sequence_length,self.head_dim]

        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, shape=(-1, K.shape(K_seq)[1], self.multiheads, self.head_dim))
        K_seq = K.permute_dimensions(K_seq, pattern=(0, 2, 1, 3))

        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, shape=(-1, K.shape(V_seq)[1], self.multiheads, self.head_dim))
        V_seq = K.permute_dimensions(V_seq, pattern=(0, 2, 1, 3))

        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / K.sqrt(K.cast(self.head_dim,
                                                                   dtype='float32'))  # A.shape=[batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        A = K.permute_dimensions(A, pattern=(
        0, 3, 2, 1))  # A.shape=[batch_size,K_sequence_length,Q_sequence_length,self.multiheads]
        # Mask：
        # 1.Sequence-wise Mask(axis=1)：
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, pattern=(
        0, 3, 2, 1))  # A.shape=[batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        # 2.Position-wise Mask(axis=2)
        if self.mask_right:
            ones = K.ones_like(A[:1, :1])  # ones.shape=[1,1,Q_sequence_length,K_sequence_length],
            lower_triangular = K.tf.matrix_band_part(ones, num_lower=-1,
                                                     num_upper=0)  # lower_triangular.shape=ones.shape，
            mask = (
                               ones - lower_triangular) * 1e12  # mask.shape=ones.shape            A = A - mask  # Element-wise subtract，A.shape=[batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        A = K.softmax(A)  # A.shape=[batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        # V_seq.shape=[batch_size,V_sequence_length,V_embedding_dim]
        O_seq = K.batch_dot(A, V_seq,
                            axes=[3, 2])  # O_seq.shape=[batch_size,self.multiheads,Q_sequence_length,V_sequence_length]
        O_seq = K.permute_dimensions(O_seq, pattern=(
        0, 2, 1, 3))  # O_seq.shape=[batch_size,Q_sequence_length,self.multiheads,V_sequence_length]
        # (batch_size*V_sequence_length)/self.head_dim
        O_seq = K.reshape(O_seq, shape=(
        -1, K.shape(O_seq)[1], self.output_dim))  # O_seq.shape=[,Q_sequence_length,self.multiheads*self.head_dim]
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

from lib.function import *
from keras.models import Model
from sklearn.svm import SVC
def Reinforce_Training(args,
                       model,
                       X_train,Y_train,X_test,
                       X_Pseudo_train,X_Pseudo_test):
    layer_name = get_feature_layer_name(model)
    final_layer = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    #Train SVM
    train_out = np.squeeze(final_layer.predict(X_train))
    train_features = np.concatenate((train_out,X_Pseudo_train),axis=1)
    svm = SVC(gamma="auto", C=1)
    svm.fit(train_features, Y_train)

    #Valid
    test_out = np.squeeze(final_layer.predict(X_test))
    test_features = np.concatenate((test_out,X_Pseudo_test),axis=1)
    predictions = svm.predict(test_features)
    predictions = (tflearn.data_utils.to_categorical(predictions, 2))

    return model, svm, predictions
