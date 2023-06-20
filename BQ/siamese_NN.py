# -*- coding: utf-8 -*-
import os
import tensorflow as tf

import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   
sess = tf.Session(config=config)

KTF.set_session(sess)

from keras import backend as K
from keras.layers import Embedding, Bidirectional,Input,subtract,add,Permute, Lambda,LSTM,Dense,Activation,GlobalAveragePooling1D,GlobalMaxPooling1D,multiply,concatenate,Dot,Dropout,BatchNormalization
from keras.models import Model
import data_helper
from tensorflow.python.ops.nn import softmax
from keras.utils.generic_utils import get_custom_objects
input_dim = data_helper.MAX_SEQUENCE_LENGTH
emb_dim = data_helper.EMB_DIM
model_path = './model/weights.best.hdf5'
tensorboard_path = './model/ensembling'

embedding_matrix = data_helper.load_pickle('embedding_matrix.pkl')

embedding_layer = Embedding(embedding_matrix.shape[0],
                            emb_dim,
                            weights=[embedding_matrix],
                            input_length=input_dim,
                            trainable=False)

def f1_score(y_true, y_pred):

    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    if c3 == 0:
        return 0

    precision = c1 / c2

    recall = c1 / c3

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def precision(y_true, y_pred):

    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    if c3 == 0:
        return 0

    precision = c1 / c2

    return precision

def recall(y_true, y_pred):

    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    if c3 == 0:
        return 0

    recall = c1 / c3

    return recall
def align(input_1, input_2):
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])  

    in1_aligned = add([in1_aligned,input_1])
    in2_aligned = add([in2_aligned,input_2])
    
    return in1_aligned, in2_aligned

def multi_encoding(input_1, input_2,input_3, input_4):    
    net =  Bidirectional(LSTM(300, return_sequences=True,dropout=0.5),merge_mode='sum')
    q1 = net(input_1)
    q2 = net(input_2)
    q3 = net(input_3)
    q4 = net(input_4)

    q13_aligned, q31_aligned = align(q1, q3)
    q24_aligned, q42_aligned = align(q2, q4)
    
    d1 = concatenate([q13_aligned,q31_aligned])
    d2 = concatenate([q24_aligned,q42_aligned])
    
    d1_aligned, d2_aligned = align(d1, d2)
    
    return d1_aligned, d2_aligned

def siamese_encoding(input_1, input_2):   
    net = Bidirectional(LSTM(600, return_sequences=True,dropout=0.5),merge_mode='sum')#
    p1 = net(input_1)
    p2 = net(input_2)

    q1_aligned, q2_aligned = align(p1, p2)
    
    return q1_aligned, q2_aligned

def siamese_model():
    input_shape = (input_dim,)  
    input_q1 = Input(shape=input_shape, dtype='int32') 
    input_q2 = Input(shape=input_shape, dtype='int32')
    input_q3 = Input(shape=input_shape, dtype='int32')
    input_q4 = Input(shape=input_shape, dtype='int32')
    
    q1_w = embedding_layer(input_q1) 
    q2_w = embedding_layer(input_q2) 
    q1_c = embedding_layer(input_q3) 
    q2_c = embedding_layer(input_q4) 
    
    d1_aligned, d2_aligned = multi_encoding(q1_w,q2_w,q1_c,q2_c)

    #align
    f1,f2= siamese_encoding(d1_aligned, d2_aligned)

    f1 = GlobalMaxPooling1D()(f1)  
    f2 = GlobalMaxPooling1D()(f2)  

    ab = Lambda(lambda x: K.abs(x[0] - x[1]))([f1,f2])
    ad = Lambda(lambda x: (x[0] + x[1]))([f1,f2])
    su = Lambda(lambda x: (x[0] - x[1]))([f1,f2])
    mu = Lambda(lambda x: (x[0] * x[1]))([f1,f2])
    ff = concatenate([ab,mu,f1,f2,ad,su])#

    similarity = Dropout(0.5)(ff)
    similarity = Dense(600,activation='relu')(similarity)
    similarity = Dropout(0.5)(similarity)
    similarity = Dense(600,activation='relu')(similarity)
    similarity = Dropout(0.5)(similarity)
    pred = Dense(1,activation='sigmoid')(similarity)  
     
    model = Model([input_q1, input_q2, input_q3, input_q4], [pred])

    #binary_crossentropy
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, f1_score])
    return model