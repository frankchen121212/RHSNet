import types,random
import numpy as np
from lib.CNN_core import *
from lib.Attention_core import *
from lib.loss import get_loss_function
from keras.models import Sequential,Model
from keras.layers import concatenate,Conv1D,Lambda,MaxPool1D,GlobalMaxPool1D,Dense,AveragePooling1D
from keras.layers import Input,Flatten,Bidirectional,GRU,Embedding

from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K

def predict_mc(self, X_pred, n_preds=100):
    return np.mean([self.predict(X_pred) for i in range(n_preds)], axis=0)

def CNN(epochs, args, loss):
    bigger_model = Sequential()

    input_length = args["input_length"]
    input_height = args["input_height"]

    bigger_model.add(
        Conv1D(input_shape=(input_length, input_height),  # but one channel in the one hot encoding of the genome
               filters=16,
               kernel_size=30,
               strides=1,
               padding="valid",
               activation="relu",
               kernel_regularizer=l2(0),
               name = "conv_first_layer"
               ))
    # bigger_model.add(Lambda(lambda x: K.dropout(x, level=args["dp_rate"])))
    bigger_model.add(MaxPool1D(pool_size=8))
    bigger_model.add(Conv1D(
        filters=16,
        kernel_size=4,
        strides=1,
        padding="valid",
        activation="relu",
        kernel_regularizer=l2(0),
        name = "conv_second_layer"
    ))
    # bigger_model.add(Lambda(lambda x: K.dropout(x, level=args["dp_rate"])))
    bigger_model.add(GlobalMaxPool1D())
    bigger_model.add(Dense(2, activation="softmax"))
    # Frozen Layers
    if args["init_weight"]:
        for i, layer in enumerate(bigger_model.layers):
            if "conv_first_layer" in layer.name:
                layer.trainable = True
                print("{} : {} {}".format(i, layer.name, True))
            else:
                layer.trainable = False
                print("{} : {} {}".format(i, layer.name, False))

    lrate = args["lr"]
    decay = lrate / epochs
    adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
    bigger_model.compile(loss=loss,
                         optimizer=adam,
                         metrics=["accuracy"]
                         )
    bigger_model.predict_mc = types.MethodType(predict_mc, bigger_model)
    return bigger_model

def Equivariant_CNN(epochs, args, loss):
    bigger_model = Sequential()

    input_length = args["input_length"]
    input_height = args["input_height"]

    bigger_model.add(
        Conv1D(input_shape=(input_length, input_height),  # but one channel in the one hot encoding of the genome
               filters=16,
               kernel_size=30,
               strides=1,
               padding="valid",
               activation="relu",
               kernel_regularizer=l2(0)
               ))
    bigger_model.add(MCRCDropout(args["dp_rate"]))
    bigger_model.add(MaxPool1D(pool_size=8))
    bigger_model.add(Conv1D(
        filters=16,
        kernel_size=4,
        strides=1,
        padding="valid",
        activation="relu",
        kernel_regularizer=l2(0)
    ))
    bigger_model.add(MCRCDropout(args["dp_rate"]))
    bigger_model.add(CustomSumPool())
    bigger_model.add(GlobalMaxPool1D())
    divisor = 2
    bigger_model.add(Dense(2, activation="softmax", kernel_initializer=Constant(
        np.array([[1] * (16 // divisor), [1] * (16 // divisor)])),
                           bias_initializer=Constant(np.array([1, -1]))))

    lrate = args["lr"]
    decay = lrate / epochs
    adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
    bigger_model.compile(loss=loss,
                         optimizer=adam,
                         metrics=["accuracy"]
                         )
    bigger_model.predict_mc = types.MethodType(predict_mc, bigger_model)
    return bigger_model

def SeqModel(epochs,args, loss):
    input_length = args["input_length"]
    input_height = args["input_height"]

    inpt = Input(shape=(input_length, input_height))
    # Conv1
    x = Conv1D(filters=16,
               kernel_size=30,
               strides=1,
               padding="valid",
               activation="relu",
               name = "conv_first_layer",
               kernel_regularizer=l2(0)
               )(inpt)
    x = Lambda(lambda x: K.dropout(x, level=args["dp_rate"]))(x)
    x = MaxPool1D(pool_size=8)(x)
    # Conv2
    x = Conv1D(filters=16,
               kernel_size=4,
               strides=1,
               padding="valid",
               activation="relu",
               name = "conv_second_layer",
               kernel_regularizer=l2(0)
               )(x)
    x = Lambda(lambda x: K.dropout(x, level=args["dp_rate"]))(x)
    x = MaxPool1D(pool_size=4)(x)
    # GRU
    x = Bidirectional(GRU(16, return_sequences=True))(x)

    if args["attention"]:
        x = Multiheads_Attention(multiheads=4, head_dim=4, mask_right=False)([x, x, x])

    x = Lambda(lambda x: K.dropout(x, level=args["dp_rate"]))(x)
    x = GlobalMaxPool1D()(x)

    out = Dense(2, activation="softmax")(x)
    model = Model(inpt, out)
    lrate = args["lr"]
    decay = lrate / epochs
    adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)

    model.compile(loss=loss,
                 optimizer=adam,
                 metrics=["accuracy"]
                         )
    model.predict_mc = types.MethodType(predict_mc, model)
    return model

def Ensemble_SeqModel(epochs,args, loss):

    input_length = args["input_length"]
    input_height = args["input_height"]

    seq_inpt = Input(shape=(input_length, input_height))
    # Conv1
    x = Conv1D(filters=16,
               kernel_size=30,
               strides=1,
               padding="valid",
               activation="relu",
               kernel_regularizer=l2(0),
               name = "conv_first_layer"
               )(seq_inpt)
    x = Lambda(lambda x: K.dropout(x, level=args["dp_rate"]))(x)
    x = MaxPool1D(pool_size=8)(x)
    # Conv2
    x = Conv1D(filters=16,
               kernel_size=4,
               strides=1,
               padding="valid",
               activation="relu",
               kernel_regularizer=l2(0),
               name = "conv_second_layer"
               )(x)
    x = Lambda(lambda x: K.dropout(x, level=args["dp_rate"]))(x)
    x = MaxPool1D(pool_size=4)(x)
    # GRU
    x = Bidirectional(GRU(16, return_sequences=True))(x)

    if args["attention"]:
        x = Multiheads_Attention(multiheads=4, head_dim=4, mask_right=False)([x, x, x])
    x = Lambda(lambda x: K.dropout(x, level=args["dp_rate"]))(x)
    x = GlobalMaxPool1D()(x)
    feature_seq = Dense(16, activation='relu')(x)

    # Extra Features
    inputs   = [seq_inpt]
    features = [feature_seq]

    # CHIP seq Feature
    if args["chip_seq"]:
        chip_inpt = Input(shape=(18,))
        chip_x = Dense(50, name="chip_first_layer")(chip_inpt)
        chip_x = Lambda(lambda x: K.dropout(x, level=args["dp_rate"]))(chip_x)
        feature_chip = Dense(4, name="chip_second_layer", activation="relu")(chip_x)
        inputs.append(chip_inpt)
        features.append(feature_chip)

    # Connect features
    final_features = concatenate(features)
    output = Dense(2, activation="softmax")(final_features)

    model = Model(inputs=inputs, outputs=output)
    lrate = args["lr"]
    decay = lrate / epochs
    adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=["accuracy"]
                  )
    model.predict_mc = types.MethodType(predict_mc, model)
    return model

def Attention(epochs,args, loss):
    input_length = args["input_length"]
    input_height = args["input_height"]
    #TODO
    inpt = Input(shape=(input_length, input_height))

    # Conv1
    x = Conv1D(filters=16,
               kernel_size=30,
               strides=1,
               padding="valid",
               activation="relu",
               kernel_regularizer=l2(0)
               )(inpt)
    x = Lambda(lambda x: K.dropout(x, level=args["dp_rate"]))(x)
    x = MaxPool1D(pool_size=8)(x)
    # Conv2
    x = Conv1D(filters=16,
               kernel_size=4,
               strides=1,
               padding="valid",
               activation="relu",
               kernel_regularizer=l2(0)
               )(x)
    x = Lambda(lambda x: K.dropout(x, level=args["dp_rate"]))(x)
    x = MaxPool1D(pool_size=4)(x)

    x = Multiheads_Attention(multiheads=8, head_dim=8, mask_right=False)([x, x, x])
    x = GlobalMaxPool1D()(x)
    out = Dense(2, activation="softmax")(x)

    model = Model(inpt, out)
    lrate = args["lr"]
    decay = lrate / epochs
    adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
    model.compile(loss=loss,
                 optimizer=adam,
                 metrics=["accuracy"]
                         )
    model.predict_mc = types.MethodType(predict_mc, model)
    return model

def Random_Guess(X_test):
    num_test = X_test.shape[0]
    guess_result = []
    for i in range (num_test):
        result = random.randint(0, 1)
        if result == 0:
            guess_result.append([1, 0])
        else:
            guess_result.append([0, 1])
    return np.array(guess_result)
