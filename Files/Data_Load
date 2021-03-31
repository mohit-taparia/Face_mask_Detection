import tensorflow as tf
import numpy as np
import os
from imutils import paths
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


#This function loads the images and pre-process them as well using their path provided as arguement and return the Test and Train set respectively
def load_images_and_labels(images_path):
    print("[INFO] Loading Images......")
    imagePaths = list(paths.list_images(images_path))

    data = []
    labels = []

    for imagePath in imagePaths:
        # get class label from folder name
        label = imagePath.split(os.path.sep)[-2]
        # our model requrie (224,224)
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        #
        image = preprocess_input(image)

        data.append(image)
        labels.append(label)


    # convert the data and labels to NumPy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)


    # perform one-hot encoding on the labels




    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    labels = tf.keras.utils.to_categorical(labels, 2)

    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                      test_size=0.20, stratify=labels, random_state=42)
    print("[INFO] Loaded")

    return trainX, testX, trainY, testY



