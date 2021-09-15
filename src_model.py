import cv2 
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

trainDataPath = "C:\\Users\\HP\\Desktop\\Research\\Trajectory_Markov_Research\\Implementations\Dataset_Dailmer\\Data\\TrainingData\\2012-04-02_115542\\RectGrabber\\"

class pipeline:
    def showSampleImages(self):
        print("[+]Logging sample images' details")
        for i in range(5):
            img_left = "imgrect_00000000" + str(i) + "_c0.pgm"
            img_right = "imgrect_00000000" + str(i) + "_c1.pgm"
            
            imgL = cv2.imread(trainDataPath+img_left)
            imgR = cv2.imread(trainDataPath+img_right)
            imLeft = cv2.resize(imgL, (0,0), fx=0.5, fy=0.5) 
            imRight = cv2.resize(imgR, (0,0), fx=0.5, fy=0.5) 
            
            if imgL is None or imgR is None:
                print(f"[!]Couldn't load images:\n{imgL} and {imgR}\n")
                continue
        
            print(f"Images: {img_left} and {img_right}\n(Shapes)L: {imLeft.shape} R: {imRight.shape}\n")
            cv2.imshow(f"Left {img_left}", imLeft)
            cv2.imshow(f"Right {img_right}", imRight)
            cv2.waitKey(0)
            
        print("[+] Finished viewing 5 initial samples.")
        
    def __init__(self):
        print(cv2.__version__)
        self.showSampleImages()


a = pipeline()