import numpy as np
import cv2
# import torch
import pickle


##########################################

frameWidth= 640      #Camera Resolution 
frameHeight= 480
brightness= 180
threshold= 0.90    #Probability Threshold
font= cv2.FONT_HERSHEY_SIMPLEX
##############################################

#Setup The Video Camera
cap= cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(0, brightness)

#Setup The Trainnned Model
with open("model_trained.p","rb") as pickle_in:    #rb = Read Byte
    model=pickle.load(pickle_in)

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img 
def equalize(img):
    img = cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getClassName(classNO):
    if classNO == 0: return 'Speed Limit 20km/h'
    elif classNO == 2: return 'Speed Limit 50km/h'
    elif classNO == 3: return 'Speed Limit 60km/h'
    elif classNO == 4: return 'Speed Limit 70km/h'
    elif classNO == 1: return 'Speed Limit 30km/h'
    elif classNO == 5: return 'Speed Limit 80km/h'
    elif classNO == 6: return 'End Of Speed Limit 80 km/h'
    elif classNO == 7: return 'Speed Limit 100 km/h'
    elif classNO == 8: return 'Speed Limit 120 km/h'
    elif classNO == 9: return 'No Passing'
    elif classNO == 10: return 'No Passing for Vehicle Over 3.5 metric tons'
    elif classNO == 11: return 'Right of way at the next intersection'
    elif classNO == 12: return 'Priority Road'
    elif classNO == 13: return 'Yeid'
    elif classNO == 14: return 'Stop'
    elif classNO == 15: return 'No Vehicle'
    elif classNO == 16: return 'Vehicles over 3.5 metric tons prohibited'
    elif classNO == 17: return 'No Entry'
    elif classNO == 18: return 'General caution'
    elif classNO == 19: return 'Dengerous curve to the Left'
    elif classNO == 20: return 'Dengerous curve to the Right'
    elif classNO == 21: return 'Double Curve'
    elif classNO == 22: return 'Bumpy Road'
    elif classNO == 23: return 'Slippery Road'
    elif classNO == 24: return 'Road Narrow on the Right '
    elif classNO == 25: return 'Road Work'
    elif classNO == 26: return 'traffic Signals'
    elif classNO == 27: return 'Pedestrains'
    elif classNO == 28: return 'Childern Crossing'
    elif classNO == 29: return 'Bicycles Crossing'
    elif classNO == 30: return 'Beware of ice/snow'
    elif classNO == 31: return 'Wild Animals Crossing'
    elif classNO == 32: return 'End of all speed and passing Limits'
    elif classNO == 33: return 'Turn Right Ahead'
    elif classNO == 34: return 'Turn Left Ahead'
    elif classNO == 35: return 'Ahead Only'
    elif classNO == 36: return 'Go Straight or Right'
    elif classNO == 37: return 'Go Straight or Left'
    elif classNO == 38: return 'Keep Right'
    elif classNO == 39: return 'Keep Left'
    elif classNO == 40: return 'Round about Madatory'
    elif classNO == 41: return 'End of no Passing'
    elif classNO == 42: return 'End of no passing by vehicles over 3.5 metric tons'

while True:
     
     #Read Image
     success, imgOrignal = cap.read()

     #Pracess Image
     img = np.asarray(imgOrignal)
     img = cv2.resize(img, (32,32))
     img = preprocessing(img)
     cv2.imshow("Processed Image", img)
     img = img.reshape(1, 32, 32, 1)
     cv2.putText(imgOrignal, "CLASS: ", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
     cv2.putText(imgOrignal, "PROBABILITY: ", (20,75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
     
     #Predict Image
     predictions = model.predict(img)
     classIndex = np.argmax(predictions, axis=1)
     probabilityValue = np.amax(predictions)
     if probabilityValue > threshold:
         print(getClassName(classIndex))
         cv2.putText(imgOrignal,str(classIndex)+" "+str(getClassName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
         cv2.putText(imgOrignal,str(round(probabilityValue*100,2) )+"%",(180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)  
     cv2.imshow("Result", imgOrignal)

     if cv2.waitKey(1) & 0xFF == ord('q'):
         break