import numpy as np
import roifile,time
import cv2, glob, os
from tkinter import filedialog
import matplotlib.pyplot as plt
import functions as fn
import pandas as pd
import pickle as plk

def translate(im,polyList,scaleX,scaleY,translateX,translateY,mirror,angle):
    left = []
    right = []
    for i in polyList:
        temp = i.copy()
        temp[:,0] = temp[:,0]*scaleX+translateX
        temp[:,1] = temp[:,1]*scaleY+translateY
        antiTemp = temp.copy()
        antiTemp[:,0] = -antiTemp[:,0]+2*mirror
        temp = rotate(temp,angle,im.shape[0]/2,im.shape[1]/2)
        antiTemp = rotate(antiTemp,angle,im.shape[0]/2,im.shape[1]/2)
        right.append(antiTemp)
        left.append(temp)
    return left+right

def rotate(poly,angle,x,y):
    poly[:,0] = poly[:,0]-x
    poly[:,1] = poly[:,1]-y
    newPol = list()
    for i in range(len(poly)):
        newPol.append([(poly[i,0]*np.cos(angle)-poly[i,1]*np.sin(angle))+x,(poly[i,0]*np.sin(angle)+poly[i,1]*np.cos(angle))+y])
    return np.array(newPol).astype(np.int64)

ROI_path = filedialog.askdirectory(initialdir=os.getcwd(),title='ROI folder')
#ROI_path = "ROIs"

file_path = filedialog.askopenfilename(initialdir=os.getcwd(),title='Pick your file')
align_file = glob.glob(os.path.dirname(file_path)+"/*.aligned")
[translateX,translateY,scaleX,scaleY,mirror,angle] = np.loadtxt(align_file[0])
#file_path = 'AS_WF186_Bin2_GoNoGo_Day8JCamF15Eavg.tif'
csv_name = "Sofie_change_this" ############### CHANGE NAME HERE ##################
wholeImg = fn.load_images(file_path)


ROI_list = glob.glob(ROI_path+'/*.roi')
polyList = []

for i in ROI_list: 
    polyList.append(roifile.ImagejRoi.fromfile(i).coordinates())

polyList = translate(wholeImg[0],polyList,scaleX,scaleY,translateX,translateY,mirror,angle)

data = []
tempList = []
startTime = time.perf_counter()

for i in wholeImg:
    for j in polyList:
        mask = np.zeros_like(wholeImg[0]).astype(np.uint8)
        cv2.fillPoly(mask, [j] ,(1,1,1))
        slV = [j[:,0].min(),j[:,0].max(),j[:,1].min(),j[:,1].max()]
        imgSlice = i[slV[2]:slV[3],:]
        imgSlice = imgSlice[:,slV[0]:slV[1]]
        min_mask = mask[slV[2]:slV[3],:]
        min_mask = min_mask[:,slV[0]:slV[1]]
        slicedImg = cv2.bitwise_and(imgSlice, imgSlice, mask=min_mask)
        if slicedImg.max()==0:
            tempList.append(np.mean(slicedImg[np.where(np.logical_or(slicedImg/slicedImg.min()<-0.00001,slicedImg/slicedImg.min()>0.00001))]))
        else:
            tempList.append(np.mean(slicedImg[np.where(np.logical_or(slicedImg/slicedImg.max()<-0.00001,slicedImg/slicedImg.max()>0.00001))]))
        
    data.append(tempList)
    tempList = []

nameList = []
for i in ROI_list:
    nameList.append(i.split('/')[-1].replace('.roi',''))

for i in ROI_list:
    nameList.append(i.split('/')[-1].replace('.roi','').replace('L-','R-'))

with open(csv_name+'.plk','wb') as f:
    plk.dump([nameList,polyList],f)
df = pd.DataFrame(data=data,columns = nameList)
df.to_csv(os.path.dirname(file_path)+'/'+csv_name+'.csv')
print('Time taken:',time.perf_counter()-startTime)