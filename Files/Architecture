import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Input
import tensorflow.keras.layers as layers
import cv2
import time
import numpy as np
import imutils
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import os


#This function creates a model using the base_model from MobileNetV2 architecture using pretrained weights from the "imagenet" dataset
#and then using our own head_model and finally returns the model
def fine_tune_architecture():
    baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    headModel = baseModel.output
    headModel = layers.AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = layers.Flatten(name="flatten")(headModel)
    headModel = layers.Dense(128, activation="relu")(headModel)
    headModel = layers.Dropout(0.5)(headModel)
    headModel = layers.Dense(2, activation="softmax")(headModel)

    model = tf.keras.Model(inputs=baseModel.input, outputs=headModel)

    # freeze the layers of baseModel
    baseModel.trainable = False

    return model


#This function takes an image as input and passes it through faceNet first to extract a face to predict depening on the confidence score threshold
#and then passing that face through the model to predict if a mask is observed or not and then return the final result with the face detection outline drawn
#with green or red color and the confidence score mentioned as well
def mask_detection_on_image(input_img):
    # get Models
    faceNet, maskNet = getModels()

    # relative path
    fileDir = os.path.dirname(os.path.abspath(__file__))
    parentDir = os.path.dirname(fileDir)

    # load the input image from disk, clone it, and grab the image spatial
    # dimensions
    image = cv2.imread(os.path.join(parentDir, 'examples', input_img))

    # get detections and original height , width of image
    detections, h, w = pass_img_through_model(image, faceNet)

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = maskNet.predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # return the output image
    return im_rgb
    #cv2.imshow("Output", image)
    #cv2.waitKey(0)

#This function extract an image from the input webcam stream and passes it through faceNet first to extract a face to predict depening on the confidence score threshold
#and then passing that face through the model to predict if a mask is observed or not and then return the final result with the face detection outline drawn
#with green or red color and the confidence score mentioned as well. All this is done for each frame of the webcam video until "q" is pressed to stop the webcam stream
def mask_detection_video():
    faceNet, maskNet = getModels()
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=800)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

#This function is used to load the saved model for face mask detection(our model) as well as the model for face detection(faceNet) and returns it
def getModels():
    #print("[INFO] loading face detector model...")
    fileDir = os.path.dirname(os.path.abspath(__file__))
    parentDir = os.path.dirname(fileDir)
    prototxtPath = os.path.join(parentDir, 'face_detector', 'deploy.prototxt')
    weightsPath = os.path.join(parentDir, 'face_detector', 'weights.caffemodel')

    classifier_path = os.path.join(parentDir, 'classifier_model.h5')

    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    #print("[INFO] loading face mask detector model...")
    maskNet = tf.keras.models.load_model(classifier_path)
    return faceNet, maskNet

#This function takes the input image and passes it through faceNet model to detect faces and return the detections and its height and width
def pass_img_through_model(input, faceNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    # need original size for output image
    (h, w) = input.shape[:2]

    # these are mean subtraction values (104.0, 177.0, 123.0)
    blob = cv2.dnn.blobFromImage(input, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    return detections, h, w

#This function takes an input image and detects the faces present in it and their locations as well and return them
def detect_and_predict_mask(frame, faceNet, maskNet):
    # get detections and original hight width of frame
    detections, h, w = pass_img_through_model(frame, faceNet)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

