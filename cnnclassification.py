from sklearn.metrics import classification_report
import utils as utl
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from matplotlib import pyplot as plt
import sherpa
import sherpa.algorithms.bayesian_optimization as by
from keras import optimizers
import time
from keras.utils.vis_utils import plot_model as plm

def finetunemodel(num_pens, num_wavelengths,xtrain, ytrain,bScale_data):
    [nrows_train, ncols_train] = xtrain.shape
    if bScale_data == True:
        scale_f = StandardScaler()
        xtrain = scale_f.fit_transform(xtrain)
    xtrain_np = xtrain.reshape(nrows_train, ncols_train, 1)

    ytrain_np = to_categorical(ytrain)


    parameters = [sherpa.Choice('filter_count', [8, 16, 32]),
                  sherpa.Choice('kernal_size', [3,5,7,9,11,13]),
                  sherpa.Choice('batch_size', [32, 64, 128]),
                  sherpa.Discrete('num_h_units', [2,4,6,8,10]),
                  sherpa.Choice('num_epochs', [5,10,20,30,50]),
                  sherpa.Choice('lr', [0.001, 0.003, 0.005, 0.01, 0.05])]

    alg = sherpa.algorithms.bayesian_optimization.BayesianOptimization(max_num_trials=100)

    study = sherpa.Study(parameters=parameters,
                         algorithm=alg,
                         lower_is_better=False, disable_dashboard=True, output_dir="C:/")

    input_shape = [num_wavelengths, 1]
    for trial in study:

        model = Sequential()
        model.add(Conv1D(filters=trial.parameters['filter_count'],
                         kernel_size=trial.parameters['kernal_size'],
                         activation='relu', input_shape=input_shape))

        for i in range(trial.parameters['num_h_units']):
            model.add(Conv1D(filters=trial.parameters['filter_count'],
                             kernel_size=trial.parameters['kernal_size'],
                             activation='relu', input_shape=input_shape))

        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(num_pens, activation='softmax'))

        opt = optimizers.Adam(lr=trial.parameters['lr'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        model.fit(xtrain_np, ytrain_np, epochs=trial.parameters['num_epochs'],
                  batch_size=trial.parameters['batch_size'],
                  callbacks=[study.keras_callback(trial, objective_name='acc')])
        study.finalize(trial)

    study.save("c://result_normalised.csv")
    # study.load_dashboard("c:/result.csv")


def create_model( num_pens, num_wavelengths, filter_count, kernal_sz,h_layers, lr_i ):

    # For conv1d statement:
    input_shape = [num_wavelengths,1]

    model = Sequential()
    model.add(Conv1D(filters=filter_count, kernel_size=kernal_sz, activation='relu', input_shape=input_shape))

    for i in range(h_layers):
        model.add(Conv1D(filters=filter_count, kernel_size=kernal_sz, activation='relu', input_shape=input_shape))

    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_pens, activation='softmax'))

    opt = optimizers.Adam(lr=lr_i, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.save("straw.h5")

    return model

def train_model( model_in, xtrain, ytrain, epochs, batch_size, bScale_data):
    bverbose = True
    [nrows_train, ncols_train] = xtrain.shape
    if bScale_data == True:
        scale_f = StandardScaler()
        xtrain = scale_f.fit_transform(xtrain)
    xtrain_np = xtrain.reshape(nrows_train, ncols_train, 1)

    ytrain_np = to_categorical(ytrain)

    history = model_in.fit(xtrain_np, ytrain_np, epochs=epochs, batch_size=batch_size, verbose=bverbose)
    return model_in,history

def evaluate_model(model_in,xtest, ytest,batch_size, bScale_data):
    bverbose = True
    [nrows_test, ncols_test] = xtest.shape
    if bScale_data == True:
        scale_f = StandardScaler()
        xtest = scale_f.fit_transform(xtest)
    xtest_np = xtest.reshape(nrows_test, ncols_test, 1)

    ytest_np = to_categorical(ytest)
    accuracy = model_in.evaluate(xtest_np, ytest_np, batch_size=batch_size, verbose=bverbose)
    return model_in,accuracy

def process(X_train,Y_train, X_test,Y_test, num_pens, num_wl, filter_count, kernel_size, epochs,
            batch_size, lr, hl, bstandardise = False):
    # training data proc


    model = create_model(num_pens, num_wl, filter_count, kernel_size, hl, lr )
    start = time.time()
    model, history = train_model(model, X_train, Y_train, epochs, batch_size, bstandardise)
    end = time.time()
    # print("total time for training in seconds")
    # print(end - start)

    model.save('my_model.h5')

    # summarize history for accuracy and loss
    plt.figure()
    plt.plot(history.history['acc'], "g--", label="Accuracy of training data")
    # plt.plot(history.history['val_acc'], "g", label="Accuracy of validation data")
    plt.plot(history.history['loss'], "r--", label="Loss of training data")
    # plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    #plt.close()
    plt.savefig("accvsloss29inks.jpg",dpi=600)
    #plt.show()
    #
    # %%

    # test data proc
    # [nrows_test, ncols_test] = X_test.shape
    # xtest_np = X_test.reshape(nrows_test, ncols_test, 1)
    #
    # ytest_np = to_categorical(Y_test)

    [nrows_test, ncols_test] = X_test.shape
    if bstandardise == True:
        scale_f = StandardScaler()
        X_test = scale_f.fit_transform(X_test)
    xtest_np = X_test.reshape(nrows_test, ncols_test, 1)

    ytest_np = to_categorical(Y_test)

    start = time.time()
    model, score = evaluate_model(model, X_test, Y_test, batch_size, bstandardise)
    end = time.time()
    # print("total time for testing in seconds")
    # print(end - start)
    #
    # print("\nAccuracy on test data: %0.2f" % score[1])
    # print("\nLoss on test data: %0.2f" % score[0])
    #
    # print("\n--- Confusion matrix for test data ---\n")

    y_pred_test = model.predict(xtest_np)
    # Take the class with the highest probability from the test predictions
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(ytest_np, axis=1)

    utl.show_confusion_matrix(max_y_test, max_y_pred_test, num_pens)

    # %%

    # print("\n--- Classification report for test data ---\n")
    #
    # print(classification_report(max_y_test, max_y_pred_test))
    return score
