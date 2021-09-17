import cv2 
from PIL import Image 
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import sys

### Change filepath according to your machine config
trainDataPath = "C:\\Users\\HP\\Desktop\\Research\\Trajectory_Markov_Research\\Implementations\Dataset_Dailmer\\Data\\TrainingData\\2012-04-02_115542\\RectGrabber\\"

class preProcessData:
    
    def readData(self):
        #trainDataP = "C:\\Users\\HP\\Desktop\\Research\\Trajectory_Markov_Research\\Implementations\Dataset_Dailmer\\Data\\TrainingData\\"
        #dirs = [f for f in listdir(trainDataP)]
        #print(dirs)
        #allTrainingData = []
        #for directory in dirs:
        #    newPath = trainDataP + directory + "\\RectGrabber\\"
        #    files = [f for f in listdir(newPath) if isfile(join(newPath, f))]
        #    allTrainingData.append(files)
        #self.showSampleImages()
        pass
    
    
    def showFrame(self, windowName, img):
        cv2.imshow(f"{windowName}", img)
        k = cv2.waitKey(0)
        if k is not None:
            cv2.destroyWindow(f"{windowName}")
        
        
    def showSampleImages(self):
        ### TODO: Use data from allTrainingData list in the future
        filenames = [f for f in listdir(trainDataPath) if isfile(join(trainDataPath, f))]
        lstPass = []
        
        #print("[+]Logging sample images' details\n---------------------")
        for i in range(1,241,2): # This is hardcoded!!! Change it
            img_left = filenames[i-1]
            img_right = filenames[i]

            try:
                imgL = cv2.imread(trainDataPath+img_left)
                imgR = cv2.imread(trainDataPath+img_right)
                
                # Resize images
                imLeft = cv2.resize(imgL, (64, 128) ) 
                imRight = cv2.resize(imgR, (64, 128))
                
                print(f"{(i//2) + 1}. Image names: {img_left} & {img_right}\n\t\tShape-L: {imLeft.shape}     Shape-R: {imRight.shape}\n")
                lstPass.append([imLeft, imRight])
                
                #cv2.imshow(f"Left {img_left}", imLeft)
                #cv2.imshow(f"Right {img_right}", imRight)
                #k = cv2.waitKey(0)

            
                # Destroy windows automatically on any key press
                if k is not None:
                    cv2.destroyWindow(f"Left {img_left}")
                    cv2.destroyWindow(f"Right {img_right}")
            
            except:
                if imgL is None:
                    print(f"[!]Couldn't load L-image: {img_left}")
                if imgR is None:
                    print(f"[!]Couldn't load R-image: {img_right}")
                continue
        print("---------------------\n[+] Finished viewing initial samples.")
        return lstPass
        
    
    
    def frameFromLR(self):
        ### TODO: Will have to refer to the 14th paper later
        ### For now standard
        pass
    
    
    def removeNoise(self):
        pass
    
    
    def createVideo(self):
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video = cv2.VideoWriter('out_video.avi', fourcc, 30.0, (588,320))
        
        for imgL, imgR in self.sampleImages:
            video.write(imgL)

        video.release()
        cv2.destroyAllWindows()
        print(f"[+]Video Released")

    
    
    def detectPedestrians(self, frame): 
        '''
            Brief: 
                    Standard Histogram Oriented Gradients Object Detection provided by openCV. 
                    (Dalal & Triggs)
            
            Param:
                    frame -> For which pedestrian detection must be done
                    
            Returns:
                    Frame with a green bounding box around pedestrians
        '''
        #img = cv2.imread("C:\\Users\\HP\\Desktop\\Research\\Trajectory_Markov_Research\\Implementations\\Dataset_Dailmer\\Data\\TrainingData\\2012-06-05_165931\\RectGrabber\\imgrect_000000227_c0.pgm")
        #self.showFrame("Scene", img)
        #img = cv2.resize(img, (64, 128))
        pass
    
    
    def __init__(self):
        print(f"Loaded OpenCV version {cv2.__version__} @ {time.asctime(time.localtime(time.time()))}\n")
        self.processedData = self.readData()
        

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

        
        

df = preProcessData()                       # Remove noise, use paper 14, Detect pedestrians(HOG) and then pass to model
#predicted_df= model(df.processedData)
#analytics(predicted_df.preds)