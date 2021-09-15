import cv2 
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Change filepath according to your machine config
trainDataPath = "C:\\Users\\HP\\Desktop\\Research\\Trajectory_Markov_Research\\Implementations\Dataset_Dailmer\\Data\\TrainingData\\2012-04-02_115542\\RectGrabber\\"

class pipeline:
    def showSampleImages(self):
        print("[+]Logging sample images' details\n---------------------")
        for i in range(5):
            img_left = "imgrect_00000000" + str(i) + "_c0.pgm"
            img_right = "imgrect_00000000" + str(i) + "_c1.pgm"

            try:
                imgL = cv2.imread(trainDataPath+img_left)
                imgR = cv2.imread(trainDataPath+img_right)
                
                # Resize images
                imLeft = cv2.resize(imgL, (0,0), fx=0.5, fy=0.5) 
                imRight = cv2.resize(imgR, (0,0), fx=0.5, fy=0.5)
                
                print(f"{i+1}. Image names: {img_left} & {img_right}\n\t\tShape-L: {imLeft.shape}     Shape-R: {imRight.shape}\n")
                cv2.imshow(f"Left {img_left}", imLeft)
                cv2.imshow(f"Right {img_right}", imRight)
                k = cv2.waitKey(0)
            
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
                
        print("---------------------\n[+] Finished viewing 5 initial samples.")
        
    def __init__(self):
        print(f"Loaded OpenCV version {cv2.__version__} @ {time.asctime(time.localtime(time.time()))}\n")
        self.showSampleImages()


a = pipeline()