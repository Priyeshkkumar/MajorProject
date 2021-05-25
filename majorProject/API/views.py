import base64
import ktrain
import tensorflow
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.status import (
    HTTP_200_OK,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND
)

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# If getting tensorflow module not found error:
# Visit: https://stackoverflow.com/questions/42244198/importerror-no-module-named-tensorflow


########################## TEXT TO EMOTION SECTION ###########################


# load model for Text to Emotion
modelTextToEmotion = ktrain.load_predictor(
    '/home/ubuntu/MajorProject/majorProject/TextToEmo')


########################## TEXT TO EMOTION SECTION END########################


########################## PIC TO EMOTION SECTION ###########################


# Load model for Pic To Emotion
modelPicToEmotion = Sequential()

modelPicToEmotion.add(Conv2D(32, kernel_size=(3, 3),
                             activation='relu', input_shape=(48, 48, 1)))
modelPicToEmotion.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
modelPicToEmotion.add(MaxPooling2D(pool_size=(2, 2)))
modelPicToEmotion.add(Dropout(0.25))

modelPicToEmotion.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
modelPicToEmotion.add(MaxPooling2D(pool_size=(2, 2)))
modelPicToEmotion.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
modelPicToEmotion.add(MaxPooling2D(pool_size=(2, 2)))
modelPicToEmotion.add(Dropout(0.25))

modelPicToEmotion.add(Flatten())
modelPicToEmotion.add(Dense(1024, activation='relu'))
modelPicToEmotion.add(Dropout(0.5))
modelPicToEmotion.add(Dense(7, activation='softmax'))

emotions = ['anger', 'disgust', 'fearful',
            'happy', 'neutral', 'sadness', 'surprise']

modelPicToEmotion.load_weights(
    '/home/ubuntu/MajorProject/majorProject/PicToEmo/model.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# initialize front face classifier
cascade = cv2.CascadeClassifier(
    "/home/ubuntu/MajorProject/majorProject/PicToEmo/haarcascade_frontalface_default.xml")


########################## PIC TO EMOTION SECTION END#########################

@api_view(['POST'])
def PicToEmotion(request):
    """
        Returns emotion related with a pic

        Params:
            - 'picUrl' : picUrl

        Return Format dict:
            {
                "status": Success/Failure,
                "emotion": Emotion
            }
     """

    resp = dict()
    resp['status'] = 'Success'

    try:
        data_uri = request.data['picUrl']
        header, encoded = data_uri.split(",", 1)
        data = base64.b64decode(encoded)

        with open("/home/ubuntu/MajorProject/majorProject/API/imageToSave.jpeg", "wb") as fh:
            fh.write(data)
    except:
        resp['message'] = "data_uri required"

    try:
        frame = cv2.imread(
            "/home/ubuntu/MajorProject/majorProject/API/imageToSave.jpeg")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blackwhite = cv2.equalizeHist(gray)

        rects = cascade.detectMultiScale(
            blackwhite, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        for x, y, w, h in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(
                cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = modelPicToEmotion.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
        resp['emotion'] = emotion_dict[maxindex]
        return Response(data=resp, status=HTTP_200_OK)
    except:
        resp['message'] = "Error in ML Model"
        return Response(data=resp, status=HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def TextToEmotion(request):
    """
        Returns emotion related with a message

        Params:
            - 'message' : Message

        Return Format dict:
            {
                "status": Success/Failure,
                "emotion": Emotion
            }
    """

    resp = dict()
    resp['status'] = 'Success'
    try:
        message = request.GET['message']
    except Exception as e:
        resp['message'] = "message parameter required"
        return Response(data=resp, status=HTTP_400_BAD_REQUEST)

    try:
        prediction = modelTextToEmotion.predict(message)
        resp['emotion'] = prediction
        return Response(data=resp, status=HTTP_200_OK)
    except:
        resp['message'] = "Error in ML Model"
        return Response(data=resp, status=HTTP_500_INTERNAL_SERVER_ERROR)
