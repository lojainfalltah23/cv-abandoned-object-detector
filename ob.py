import numpy as myNP
import cv2
import math
import time
from tracker import *

# here we attempt to Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

#assign  the classes names from the coco.names file to the list ObjectsClasses
ObjectsClasses = []
with open("coco.names", "r") as videoFile:
    ObjectsClasses = [myLine.strip() for myLine in videoFile.readlines()]
    print(ObjectsClasses)

# retrieve layers of the network
layerNames = net.getLayerNames()

#detect the names of output layer  from the YOLO model 
outputLayers = [layerNames[k - 1] for k in net.getUnconnectedOutLayers()]
#  Tracker is Initialized
tracker = ObjectTracker()


# determine the location of first frame
pathTofirstframe = r'Frame.jpeg'

frame1 = cv2.imread(pathTofirstframe)
frame1Ggray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1Blur = cv2.GaussianBlur(frame1Ggray, (21, 21), 0)

# path of the video file
videoFilePath ='cut.mp4'
cap = cv2.VideoCapture(videoFilePath)

while (cap.isOpened()):
    
    ret, frame = cap.read()
    
    frameHeight, frameWidth, _ = frame.shape

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameBlur = cv2.GaussianBlur(frameGray, (21, 21), 0)

    # calculate the diffrence between first frame and current frame
    frameDifference = cv2.absdiff(frame1, frame)
  

    #using Canny to detect Edge 
    detectedEdges = cv2.Canny(frameDifference, 50, 200)
    cv2.imshow('CannyEdgeDet', detectedEdges)

    # define the kernel
    myKernel = myNP.ones((20, 20), myNP.uint8)
    myThresh = cv2.morphologyEx(
        detectedEdges, cv2.MORPH_CLOSE, myKernel, iterations=1)
    
    cv2.imshow('Morph_Close', myThresh)

    #using an instance of the thresh in order to detect contours    
    myCnts_list, _ = cv2.findContours(
        myThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects_detected=[]
    counter = 0
    for contour in myCnts_list:
        contour_Area = cv2.contourArea(contour)
        
        if contour_Area > 500 and contour_Area < 10000:
            counter += 1

            (x_, y_, w_, h_) = cv2.boundingRect(contour)

            objects_detected.append([x_, y_, w_, h_])

    trackedObjects_List, abandoned_objects_list = tracker.update(
        objects_detected)
    

    
    # show a rectangle and id around all tracked objects that are moving

    for newObject in trackedObjects_List:
        x_1, y1, w_1, h_1, objectId, newDist = newObject

        cv2.rectangle(frame, (x_1, y1), (x_1 + w_1, y1 + h_1), (0, 255, 0), 2)

    # show rectangle around the object with id over all abandoned objects that are not moving
    for objects_list in abandoned_objects_list:
        _, x_2, y_2, w_2, h_2, _ = objects_list

      
        label_x = x_2
        seconds = 0
        localTime = time.ctime()
        
        sizeOfLabel=cv2.getTextSize("Suspicious object detected " + localTime + ")",cv2.FONT_HERSHEY_PLAIN,1.2,2)
        labelWidth = sizeOfLabel[0][0]
        if (frameWidth-label_x) < labelWidth:
            label_x = frameWidth - labelWidth
         
        cv2.putText(frame, "Suspicious object detected (" + localTime + ")",
                    (label_x, y_2 - 10), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 255), 2)
        cv2.rectangle(frame, (x_2, y_2), (x_2 + w_2,
                      y_2 + h_2), (255, 0, 255), 2)

    cv2.imshow('main',frame)
    if cv2.waitKey(15) == ord('q'):
        break

cv2.destroyAllWindows()