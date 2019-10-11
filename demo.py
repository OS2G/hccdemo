from __future__ import print_function

import os  # to work with file paths

import tensorflow as tf         # to specify and run computation graphs
import numpy as np              # for numerical operations taking place outside of the TF graph
import matplotlib.pyplot as plt # to draw plots
from scipy.sparse import csc_matrix
fmnist_dir = '/work/cse479/shared/homework/01/'

def split_data(data, proportion):
    """
    Split a numpy array into two parts of `proportion` and `1 - proportion`

    Args:
        - data: numpy array, to be split along the first axis
        - proportion: a float less than 1
    """
    size = data.shape[0]
    split_idx = int(proportion * size)
    np.random.shuffle(data)
    return data[:split_idx], data[split_idx:]

def labelToArray(labelArray):
    convertedLabelArray = []
    for label in labelArray:
        element = np.zeros(10)
        element[int(label)] = 1
        convertedLabelArray.append(element)
    return convertedLabelArray


# extract the dataset and split into 80% training and 80% testing
data_train, data_test = split_data(np.load(fmnist_dir + 'fmnist_train_data.npy'), 0.8)
labels_train, labels_test = split_data(np.load(fmnist_dir + 'fmnist_train_labels.npy'), 0.8)

# convert the matrices from 0-255 to one hot encoded data
#data_train = tf.math.multiply(data_train,1/255)
#data_test = tf.math.multiply(data_test,1/255)


labels_test = labelToArray(labels_test)
labels_train = labelToArray(labels_train)


with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    # create the network
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(100,
                        input_shape=(784,),
                        kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                        activation='relu'));
    model.add(tf.keras.layers.Dense(10,
                        kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))


    model.compile(optimizer='Adam',loss='binary_crossentropy', metrics=['accuracy'])

    # define an early stopping callback for model.fit()
    # stop if there are 4 consecutive epochs with no improvement
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)


    # train & validate. Allocate 20% for validation set. 100 epochs
    history = model.fit(np.array(data_train), np.array(labels_train), batch_size = 32, epochs=100, callbacks=[early_stopping_
callback], verbose=1, shuffle=True, validation_split=0.2)
    # plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Historical loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()
    score = model.evaluate(np.array(data_test), np.array(labels_test), verbose=1)
    print('\n\nTest loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('fmnist_model.h5')




# load and evaluate the model
with tf.Session() as session:
    model = tf.keras.models.load_model('fmnist_model.h5')
    model.summary()
    loss, acc = model.evaluate(np.array(data_test), np.array(labels_test), verbose=2)