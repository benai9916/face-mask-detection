# import packages
import cv2
import os
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
 
#load model # Accuracy=97.4 , validation Accuracy = 99.1 # very light model, size =5MB
model = load_model('model/model_cnn.h5') # cnn
 
# model accept below hight and width of the image
img_width, img_hight = 200, 200

 
#parameters for text
# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
# org 
org = (1, 1)
class_lable=' '      
# fontScale 
fontScale = 1 #0.5
# Blue color in BGR 
color = (255, 0, 0) 
# Line thickness of 2 px 
thickness = 2 #1
 
#read image from webcam
# color_img = cv2.imread('images/mask-5136259_1280.jpg')
     

def mask_checker_img(img):
    # Load the Cascade face Classifier
    face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")

    color_img = cv2.imread(img)

    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
         
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 6) 

    if len(faces) <= 0:
        pred = 99
        return color_img, pred

    else:
         
    #take face then predict class mask or not mask then draw recrangle and text then display image
        img_count = 0
        for (x, y, w, h) in faces:
            org = (x-10,y-10)
            img_count +=1 
            color_face = color_img[y:y+h,x:x+w] # color face
            cv2.imwrite('faces/input/%dface.jpg'%(img_count),color_face)
            img = load_img('faces/input/%dface.jpg'%(img_count), target_size=(img_width,img_hight))
                 
            img = img_to_array(img)/255
            img = np.expand_dims(img,axis=0)
            pred_prob = model.predict(img)
            #print(pred_prob[0][0].round(2))
            pred = np.argmax(pred_prob)
                     
            if pred == 1:

                print('user not wearing mask - prob = ',pred_prob[0][1])
                class_lable = "No Mask"
                color = (0, 255, 0)
                cv2.imwrite('faces/without_mask/%dface.jpg'%(img_count),color_face)
                cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
                # Using cv2.putText() method 
                cv2.putText(color_img, class_lable, org, font,  
                                           fontScale, color, thickness, cv2.LINE_AA)         
                cv2.imwrite('faces/with_mask/%dno_mask.jpg'%(img_count),color_img)

            else:
                print("User with mask - predic = ",pred_prob[0][0])
                class_lable = "Mask"
                color = (255, 0, 0)
                cv2.imwrite('faces/with_mask/%dface.jpg'%(img_count),color_face)
                cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
                # Using cv2.putText() method 
                cv2.putText(color_img, class_lable, org, font,  
                                           fontScale, color, thickness, cv2.LINE_AA) 
                cv2.imwrite('faces/with_mask/%dmask.jpg'%(img_count),color_img)

                 
    return color_img, pred
