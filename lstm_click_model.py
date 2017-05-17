""" LSTM-based click model
    
    The idea is to model the process of ad ranking as a sequence generating process. At each step, the model is used to predict
    the click through rate of the candidate ad accoding to its own quanlity and the ads selected at previous steps.

    LSTM model is build using the open-sourced API keras. Keras is a high-level API based on tensorflow and and theano. It is a
    good tool for rapid prototyping 
"""

from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Masking
import numpy as np
from sklearn import metrics

# feature dimention, time steps, ect
data_dim = 26
time_steps = 5
num_classes = 2
batchsize = 1000
train_epoch = 15

#####################################################################
########### auc metric: evaluate the LSTM based click mode  #########
#####################################################################
def AUC(label, pred):
    total_num = len(pred)
    click_num = 0.0
    area_num = 0.0
    list_eval = zip(label, pred)
    list_eval = sorted(list_eval, key = lambda x : x[1], reverse = True)
    for i in range(len(list_eval)):
        area_num += click_num * (1 - list_eval[i][0])
        click_num += list_eval[i][0]
    auc = 0.0
    if total_num == click_num:
        auc = 1
    elif click_num == 0:
        auc = 0
    else:
        auc = area_num / (total_num - click_num) / click_num
    return (total_num, click_num, auc)


def load_data(mode, file_prefix):
    # load training or testing data from files
    if mode == "train":
        x_train = np.load(file_prefix+'_feature.npy')
        y_train = np.load(file_prefix+'_label.npy')
        return (x_train, y_train)
    elif mode == "test":
        x_test = np.load(file_prefix+'_feature.npy')
        y_test = np.load(file_prefix+'_label.npy')
        return (x_test, y_test)

    # for run a toy model, one can generate ramdom data
    # Generate dummy training data
    #x_train = np.random.random((data_size, timesteps, data_dim))
    #y_train = np.random.randint(2, size=(data_size, timesteps, 1))

    # Generate dummy validation data
    #x_val = np.random.random((1, timesteps, data_dim))
    #y_val = np.random.randint(1, size=(1, timesteps, 1))

#####################################################################
#### get context-free click through rate from feature vector   ######
#### context-free click through rate is predicted considering  ######
#### only the user requirement and the single ad. It is used   ######
#### as a feature for LSTM based click model                   ######
#####################################################################
def get_ctr(x_test):
    ctr = []
    for i in range(0, len(x_test)):
        for j in range(0, time_steps):
            ctr.append(round(x_test[i][j][0],time_steps))

    return np.asarray(ctr)

#####################################################################
#### define the model: layers, output, loss, optimizer, metric ######
#####################################################################
def build_model():
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))
    # output at each time step
    model.add(TimeDistributed(Dense(1,activation='sigmoid')))
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    return model

#####################################################################
############################# training process: #####################
#####################################################################
def train_model(model):
    file_prefix_list = []
    #  here add code for initiate file_prefix_list
    #
    
    for j in range(0, train_epoch):
        # train the model with data from all the training file
        for i in range(0, len(file_prefix_list)):
            (x_train, y_train) = load_data("train", file_prefix_list[i])
            print "data loaded: ", len(x_train), len(y_train)
            model.fit(x_train, y_train, nb_epoch=1, batch_size=batchsize)
            del x_train, y_train

#####################################################################
######################### save the model to file ####################
#####################################################################
def save_model(model):
    model.summary()
    model.save("./lstm_model")
    model.save_weights('./lstm_model_weights')

#####################################################################
############################# test and evaluate #####################
#####################################################################
def evaluate_model(model):
    test_file = ""
    # here add code for initiate test_file
    
    # load test data
    (x_test, y_test) = load_data("test", test_file)
    # predict
    preds = model.predict(x_test, batch_size=batchsize, verbose=0)
    # extract reference ctr from the test data
    ctr_base = get_ctr(x_test) 

    # evaluate with auc
    print "New AUC:"
    print AUC(y_test, preds)
    print "Old AUC:"
    print AUC(y_test, ctr_base)    

#####################################################################
############################## main procedure #######################
#####################################################################
def main():
    model = build_model()
    print "model built!"

    train_model(model)
    print "model trained!"

    save_model(model)
    print "model saved!"
    del model    

    model = load_model("./lstm_model")
    model.load_weights('./lstm_model_weights', by_name=True)
    model.summary()
    # model evaluation
    evaluate_model(model)

if __name__ == "__main__":    
    main()
