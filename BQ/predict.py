# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from stats_graph import stats_graph
from keras import backend as K

import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   
sess = tf.Session(config=config)

KTF.set_session(sess)
import data_helper
from siamese_NN import siamese_model
data = data_helper.load_pickle('model_data.pkl')
test_q1 = data['test_q1']
test_q2 = data['test_q2']
test_q3 = data['test_q3']
test_q4 = data['test_q4']
test_y = data['test_label']
model = siamese_model()

model.load_weights('./model/weights.best.hdf5')
sess = K.get_session()
graph = sess.graph
stats_graph(graph)
loss, accuracy, precision, recall, f1_score = model.evaluate([test_q1, test_q2, test_q3, test_q4],test_y,verbose=1,batch_size=256)
print("Test best model =loss: %.4f, accuracy:%.4f, precision:%.4f,recall: %.4f, f1_score:%.4f" % (loss, accuracy, precision, recall, f1_score))
