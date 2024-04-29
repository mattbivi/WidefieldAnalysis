import tifffile as tiff
import cv2, glob,os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
file_path = filedialog.askdirectory(initialdir=os.getcwd())
os.chdir(file_path)

fileList = glob.glob('*F15.tif')
numFiles = len(fileList)
for j,i in enumerate(fileList):
    file = tiff.imread(i)
    print("File "+str(j+1)+"/"+str(numFiles))
    file1 = file[::2,:,:]
    file2 = file[1::2,:,:]
    checker = [np.median(file[0,:,:]),np.median(file[1,:,:])]
    
    if checker[0]>checker[1]:
        tiff.imwrite(i.replace('.tif','#G.tif'),file1)
        tiff.imwrite(i.replace('.tif','#B.tif'),file2)
    elif checker[0]<checker[1]:
        tiff.imwrite(i.replace('.tif','#B.tif'),file1)
        tiff.imwrite(i.replace('.tif','#G.tif'),file2)
        