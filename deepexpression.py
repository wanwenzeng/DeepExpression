import os
import numpy as np
import sys
from sklearn.model_selection import StratifiedKFold,train_test_split
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn import metrics
import random
import argparse

np.random.seed(666)
parser = argparse.ArgumentParser(description='Testing hichip features')
parser.add_argument('-batchsize', dest='batchsize', type=int, default=8, help='size of one batch')
parser.add_argument('-lr', dest='lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('-c', dest='c', type=float, default=0.1, help='pesudo count added to Y')
parser.add_argument('-cl', dest='cl', type=str, default="K562", help='testing cell line')
parser.add_argument('-s', dest='s', type=str, default="human", help='testing species')
parser.add_argument('-plen', dest='plen', type=int, default=1000, help='length of promoter sequence')
parser.add_argument('-elen', dest='elen', type=int, default=200, help='length of enhancer sequence')
args = parser.parse_args()

name = str(args.c)+"_"+args.cl+"_"+str(args.batchsize)+"_"+str(args.lr)+str(args.plen)+"_"+str(args.elen)

X_seq = np.load("../data/processed/"+args.s+"/X_seq.npy")
X_exp = np.load("../data/processed/"+args.s+"/exp_"+args.cl+".npy")
fin   = open("../data/processed/"+args.s+"/rna_"+args.cl+".txt","r")
Y     = np.zeros(X_seq.shape[0])
Y_b   = np.zeros(X_seq.shape[0])
lines = fin.readlines()
for i in range(len(lines)):
	data = lines[i].strip().split()
	Y[i] = np.log10(float(data[1])+args.c)

cut = np.median(Y)
for i in range(len(Y)):
	if Y[i] > cut:
		Y_b[i] = 1
	else:
		Y_b[i] = 0

X_seq = X_seq[:,1000-args.plen:1001+args.plen]
X_exp = X_exp[:,200-args.elen:200+args.elen]

X_exp = np.reshape(X_exp,(X_exp.shape[0],X_exp.shape[1],1))
X_seq = np.reshape(X_seq,(X_seq.shape[0],X_seq.shape[1],X_seq.shape[2],1))

n_seqs = X_exp.shape[0]
indices = np.arange(n_seqs)
np.random.shuffle(indices)
X_exp = X_exp[indices]
X_seq = X_seq[indices]
y = Y[indices]

n_tr = int(n_seqs * 0.9)
n_va = int(n_seqs * 0.1)
n_te = n_seqs - n_tr - n_va
X_seq_train = X_seq[:n_tr]
X_exp_train = X_exp[:n_tr]
y_train = y[:n_tr]
X_seq_valid = X_seq[n_tr:n_tr+n_va]
X_exp_valid = X_exp[n_tr:n_tr+n_va]
y_valid = y[n_tr:n_tr+n_va]
X_seq_test = X_seq[-n_te:]
X_exp_test = X_exp[-n_te:]
y_test = y[-n_te:]

print X_seq_train.shape
print X_exp_train.shape
print y_train.shape

from keras.layers import Conv1D,Dense, Activation
from keras.models import Sequential
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Merge,Reshape,Concatenate,Flatten,BatchNormalization,MaxPooling1D,Flatten ,LSTM, Dropout, Bidirectional
from keras import optimizers
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K

def conv_block(ip, nb_filter, dropout_rate=None):
    x = Activation('relu')(ip)
    x = Convolution2D(nb_filter, 4, 4, init="he_uniform", border_mode="same", bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x

def transition_block(ip, nb_filter, dropout_rate=None):
    concat_axis = 1 if K.image_dim_ordering() == "th" else -1
    x = Convolution2D(nb_filter, 1, 1, init="he_uniform", border_mode="same", bias=False)(ip)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    return x

def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None):
    concat_axis = 1 if K.image_dim_ordering() == "th" else -1
    feature_list = [x]
    for i in range(nb_layers):
        x = conv_block(x, growth_rate, dropout_rate)
        feature_list.append(x)
        x = merge(feature_list, mode='concat', concat_axis=concat_axis)
        nb_filter += growth_rate
    return x, nb_filter

def create_dense_net(nb_classes, img_dim, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=None,verbose=True):
    model_input = Input(shape=img_dim)
    concat_axis = 1 if K.image_dim_ordering() == "th" else -1
    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"
    nb_layers = int((depth - 4) / 3)
    x = Convolution2D(128, 4, 8, init="he_uniform", border_mode="same", name="initial_conv2D", bias=False)(model_input)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate)
        x = transition_block(x, nb_filter, dropout_rate=dropout_rate)

    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate)
    x = Flatten()(x)
    x = Dense(40, activation='relu')(x)
    densenet = Model(input=model_input, output=x, name="create_dense_net")
    if verbose: print("DenseNet-%d-%d created." % (depth, growth_rate))
    return densenet
    print densenet.summary()

shape = args.elen*2
###experimental network
emodel = Sequential()
emodel.add(Dense(80, activation='relu',input_shape=(X_exp_train.shape[1],1)))
emodel.add(Dropout(0.5))
emodel.add(Flatten())
emodel.add(Dense(40, activation='relu'))

###sequential network
shape = args.plen*2+1
smodel = Sequential()
smodel = create_dense_net(1, X_seq_train.shape[1:], depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=0.5,verbose=True)
print smodel.summary()
print emodel.summary()
checkpointer = ModelCheckpoint(filepath=name+"_combine_bestmodel.h5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
model = Sequential()
model.add(Merge([emodel,smodel], mode='concat'))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='relu'))

print model.summary()
model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=args.lr))

model.fit([X_exp_train,X_seq_train], y_train,batch_size=args.batchsize, epochs=5000,shuffle=True, validation_data=([X_exp_val,X_seq_val], y_val),callbacks=[checkpointer,earlystopper], verbose=2)

model.load_weights(name+"_combine_bestmodel.h5")
predicted = model.predict([X_exp_test,X_seq_test])
print np.corrcoef(predicted[:,0],y_test)
