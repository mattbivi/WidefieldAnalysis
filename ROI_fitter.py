import numpy as np
import roifile
import cv2, glob, os
from tkinter import filedialog
import matplotlib.pyplot as plt
import functions as fn


#file_path = filedialog.askdirectory(initialdir=os.getcwd(),title='File path')
#ROI_path = filedialog.askdirectory(initialdir=os.getcwd(),title='ROI folder')
ROI_path = "ROIs"
#file_path = filedialog.askopenfilename(initialdir=os.getcwd(),title='Pick your file')
file_path = 'AS_WF186_Bin2_GoNoGo_Day8JCamF15processed.tif'
wholeImg = fn.load_images(file_path)

numFrames = len(wholeImg)
frameNo = 0

scaleX = 1
scaleY = 1
translateX = 0
translateY = 0
mirror_start = round(wholeImg[0].shape[1]/2)
mirror = mirror_start
angle = 0
contrast = 1
brightness = 0


wholeImg = fn.channel_rescale(wholeImg,255)
img = wholeImg[10].astype(np.uint8)

ROI_list = glob.glob(ROI_path+'/*.roi')
polyList = []
for i in ROI_list:
    	polyList.append(roifile.ImagejRoi.fromfile(i).coordinates())

def translate():
	global polyList
	polyCopy = polyList.copy()
	global img
	im = img.copy()

	global translateX
	global translateY
	global scaleX
	global scaleY
	global mirror
	global angle
	global contrast
	global brightness
	global frameNo
		
	for i in polyCopy:
		temp = i.copy()
		temp[:,0] = temp[:,0]*scaleX+translateX
		temp[:,1] = temp[:,1]*scaleY+translateY
		antiTemp = temp.copy()
		antiTemp[:,0] = -antiTemp[:,0]+2*mirror
		temp = rotate(temp,angle,im.shape[0]/2,im.shape[1]/2)
		antiTemp = rotate(antiTemp,angle,im.shape[0]/2,im.shape[1]/2)
		cv2.convertScaleAbs(im,im, alpha=contrast, beta=brightness)
		cv2.polylines(im,[temp],True,(255,255,255),2)
		cv2.polylines(im,[antiTemp],True,(255,255,255),2)
		cv2.putText(im,"Frame: "+str(frameNo), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255)

	return im

def rotate(poly,angle,x,y):
	poly[:,0] = poly[:,0]-x
	poly[:,1] = poly[:,1]-y
	newPol = list()
	for i in range(len(poly)):
		newPol.append([(poly[i,0]*np.cos(angle)-poly[i,1]*np.sin(angle))+x,(poly[i,0]*np.sin(angle)+poly[i,1]*np.cos(angle))+y])
	return np.array(newPol).astype(np.int64)

def trans_x(num):
	global translateX
	translateX = (num-100)*3
	im = translate()
	refresh(im)
	
def trans_y(num):
	global translateY
	translateY = (num-100)
	im = translate()
	refresh(im)

def refresh(result):	
	#cv2.destroyWindow("Brain")
	cv2.imshow("Brain", result)

def frameChange(num):
	global img
	global wholeImg
	global frameNo
	img = wholeImg[num].astype(np.uint8)
	frameNo = num
	im = translate()
	refresh(im)

def scale_x(num):
	num = num/100
	global scaleX
	scaleX = num
	im = translate()
	refresh(im)
	
def scale_y(num):
	num = num/100
	global scaleY
	scaleY = num
	im = translate()
	refresh(im)

def change_contrast(num):
	num = (num-50)/700+1
	global contrast
	contrast = num
	im = translate()
	refresh(im)

def change_brightness(num):
    num = num/10
    global brightness
    brightness = num
    im = translate()
    refresh(im)

def mirror_r(num):
	global mirror
	global mirror_start
	mirror = mirror_start+num-100
	im = translate()
	refresh(im)

def rotational(num):
	num = (num-100)/50
	global angle
	angle = num
	im = translate()
	refresh(im)


cv2.namedWindow("Position Controls", cv2.WINDOW_NORMAL)
cv2.namedWindow("Brain",cv2.WINDOW_AUTOSIZE)

cv2.createTrackbar("Translating X", "Position Controls", 100, 200, trans_x)
cv2.createTrackbar("Translating Y", "Position Controls", 100, 200, trans_y)
cv2.createTrackbar("Scaling X", "Position Controls", 100, 200, scale_x)
cv2.createTrackbar("Scaling Y", "Position Controls", 100, 200, scale_y)
cv2.createTrackbar("Mirrored ROI","Position Controls",100,200,mirror_r)
cv2.createTrackbar("Rotate ROI","Position Controls",100,200,rotational)
cv2.createTrackbar("Contrast","Position Controls",100,change_contrast)
cv2.createTrackbar("Brightness","Position Controls",0,100,change_brightness)

im = img.copy()
for i in polyList:
	cv2.polylines(im,[i],True,(255,255,255),2)
cv2.createTrackbar("Frame number: ","Position Controls",0,numFrames-1,frameChange)
cv2.imshow('Brain',im)


k = cv2.waitKey(0)
if k==32: # Hitting space bar causes it to save
    np.savetxt(file_path.replace('.tif','.aligned'),[translateX,translateY,scaleX,scaleY,mirror,angle])
cv2.destroyAllWindows()