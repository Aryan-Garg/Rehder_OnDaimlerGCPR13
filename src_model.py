import cv2 
from PIL import Image
import imutils
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import sys

print(f"[+]Loading essential packages...")
print(f"[+]Loaded OpenCV version {cv2.__version__} @ {time.asctime(time.localtime(time.time()))}\n")

class preProcessData:
    
    def readData(self, trainDataP):
        try:
            dirs = [f for f in listdir(trainDataP)]
            for directory in dirs:
                newPath = self.trainDataPath + directory + "\\RectGrabber\\"
                files = [f for f in listdir(newPath) if isfile(join(newPath, f))]
                self.allTrainingData[directory] = files
            #print(self.allTrainingData)
            return True
        
        except Exception as err:
            print(err)
            print("Couldn't read data. Check file paths and file health!")
            return False
        
    
    
    def showFrames(self, windowName, imglst):
        if len(windowName) != len(imglst):
            print(f"windowName list len: {len(windowName)} not equal to imglst len: {len(imglst)}")
            return False 
        
        for i in range(len(windowName)):
            cv2.imshow(f"{windowName[i]}", imglst[i])
        
        k = cv2.waitKey(0)
        if k is not None:
            cv2.destroyAllWindows()
        
        return True
        
    def showSampleImages(self, folderName = "2012-06-05_165931"):
        print("[+] Logging sample images' details\n---------------------")
        for i in range(1,10,2): 
            img_left = self.allTrainingData[folderName][i-1]
            img_right = self.allTrainingData[folderName][i]

            try:
                imgL = cv2.imread(self.trainDataPath + folderName + "\\RectGrabber\\" + img_left)
                imgR = cv2.imread(self.trainDataPath + folderName + "\\RectGrabber\\" + img_right)
                
                print(f"{(i//2) + 1}. Image names: {img_left} & {img_right}\n\t\tShape-L: {imgL.shape}     Shape-R: {imgR.shape}\n")
                
                if not self.showFrames(["Left frame", "Right frame"], [imgL, imgR]):
                    print(f"Couldn't show sample images from showSampleImages function")
            
            except Exception as err:
                print(err)
                if imgL is None:
                    print(f"[!]Couldn't load L-image: {img_left}")
                if imgR is None:
                    print(f"[!]Couldn't load R-image: {img_right}")
                continue
                
        print("---------------------\n[+] Finished viewing initial samples.")

    
    
    def frameFromLR(self):
        ### TODO: Will have to refer to the 14th paper later
        ### For now standard
        pass
    
    
    def removeNoise(self):
        ### TODO: Future optimization
        pass
    
    
    def createVideo(self, folderNum):
        try:
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            video = cv2.VideoWriter('out_video.avi', fourcc, 30.0, self.allTrainingData[folderNum][0].shape)
        
            for imgL, imgR in self.allTrainingData[folderNum][:121]:
                video.write(imgL)

            video.release()
            cv2.destroyAllWindows()
            print(f"[+]Video out_video.avi released")
            
        except Exception as err:
            print(err)
            print(f"Possible fixes:\n1. See if DIVX is compatible with your machine\n2. Use a media player that supports .avi\n\
                    3. Match the shape of frames with the video")

    
    
    def detectPedestrians(self, dirName, imageList): 
        '''
            Brief: 
                    Standard Histogram Oriented Gradients Object Detection provided by openCV. 
                    (Dalal & Triggs)
            
            Param:
                    frame -> For which pedestrian detection must be done
                    
            Returns:
                    Frame with a green bounding box around pedestrians
        '''
        
        toRet = []             # This list will become the object detected data list's part
        
        # Text formatting params:
        font                   = cv2.FONT_HERSHEY_COMPLEX_SMALL
        fontScale              = 0.7
        fontColor              = (255, 255, 255)
        lineType               = 2
        
        # Initialize standard HOG people detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        for imgName in imageList:
            img = cv2.imread( self.trainDataPath + dirName + "\\RectGrabber\\" + imgName )
            
            if img is None:
                continue
                
            img = cv2.resize(img, (600, 450))
            (rects, weights) = hog.detectMultiScale(img, winStride=(4, 4), padding=(16, 16), scale=1.2)
            for (x, y, w, h) in rects:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img,'Person', (x,y), font, fontScale,fontColor,lineType)
            
            toRet.append(img)
            if not self.showFrames(["Bounding Box"], [img]):
                print(f"Couldn't show pedestrian detected image: {img}")
        
        return toRet
           
    
    def __init__(self):
        ### Change filepath according to your machine config
        self.trainDataPath = "C:\\Users\\HP\\Desktop\\Research\\Trajectory_Markov_Research\\Implementations\Dataset_Dailmer\\Data\\TrainingData\\"
        self.allTrainingData = dict()
        if self.readData(self.trainDataPath):
            self.showSampleImages()
            self.detectedData = []
            for key in self.allTrainingData:
                self.detectedData.append(self.detectPedestrians(key, self.allTrainingData[key]))


class model:
    '''
    Brief:
        Goal Directed Pedestrian Prediction 
        (A markovian approach for pedestrian trajectory prediction)
        
    Receives:
        Pre-processed and pedestrain detected data from preProcessData class
        
    Returns:
        Predicted Trajectory of each pedestrian in scene
                
    '''
    pass
    
    
class analytics:
    '''
    Brief:
        Analyzes predicted trajectories against ground truth.
        
    Receives:
        Predicted trajectories from model class
        
    Displays:
        Accuracy and evaluation metrics. And other visual-aids for analysis of the model
                
    '''
    pass


# Create instances here :)
df = preProcessData()                       # Remove noise, use paper 14, Detect pedestrians(HOG) and then pass to model
print(df.detectedData)
#predicted_df= model(df.processedData)
#analytics(predicted_df.preds)


#if img.shape != (64, 128):
#    img = cv2.resize(img, (64, 128))
