import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

#This function takes the model, hyper_parameters as well as train and validation datasets and train the provided model on the data and then
#returns the trained model as well as the training and validation history to be plotted later, it also saves the trained model
def training(model, trainX, testX, trainY, testY, hyper_params):
    lr = hyper_params['lr']
    epochs = hyper_params['epochs']
    batch_size = hyper_params['batch_size']

    aug = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    print("[INFO] compiling model...")
    model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=["accuracy"])

    print("[INFO] training head...")
    history = model.fit(
        aug.flow(trainX, trainY, batch_size=batch_size),
        validation_data=(testX, testY),
        epochs=epochs)

    model.save('classifier_model.h5')
    print('[INFO] Model Saved to Disk !')

    return model, history





#This function plots the training and testing accuracy history
def plot_accuracy_history(history):
    COLOR = 'white'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR
    plt.rcParams['axes.facecolor'] = 'black'

    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], '#76b4bd', linewidth=3, solid_capstyle="round", linestyle='-', marker="8", markersize=6)
    plt.plot(history.history['val_accuracy'], '#d11141', linewidth=3, solid_capstyle="round", linestyle='-', marker="8", markersize=6)
    plt.title('Model Accuracy', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xlabel('Epochs', fontsize=20)
    legend = plt.legend(['train', 'test'], loc='upper left')
    #plt.setp(legend.get_texts(), color='black')
    return None


#This function plots the training and testing loss history
def plot_loss_history(history):
    COLOR = 'white'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], '#76b4bd', linewidth=3, solid_capstyle="round", linestyle='-', marker="8", markersize=6)
    plt.plot(history.history['val_loss'], '#d11141', linewidth=3, solid_capstyle="round", linestyle='-', marker="8", markersize=6)
    plt.title('Model Loss', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.xlabel('Epochs', fontsize=20)
    legend = plt.legend(['train', 'test'], loc='upper left')
    #plt.setp(legend.get_texts(), color='black')
    plt.show()
    return None
