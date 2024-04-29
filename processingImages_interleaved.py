import tkinter as tk
from tkinter import filedialog
import cv2, glob,os,time
import numpy as np
import pickle as plk
import functions as fn
import tifffile as tiff

root = tk.Tk()
root.withdraw()
file_path = filedialog.askdirectory(initialdir=os.getcwd())
try:
    os.chdir(file_path)
except:
    print('Sorry, something went wrong!')

fileList = glob.glob('*#G.tif')
cropFile = glob.glob('*.crop')

if cropFile==[]:
    imgs = fn.load_images(fileList[0])
    pd = fn.PolygonDrawer()
    mask = pd.run(imgs[1])
    with open(fileList[0].replace('.tif','.crop'),'wb') as f:
        plk.dump(pd.prevPoints,f)
    polyData = pd.prevPoints
else:
    with open(cropFile[0],'rb') as f:
        polyData = plk.load(f)
    imgs = fn.load_images(fileList[0])
    mask = fn.maskMaking(imgs[0],polyData)

overallMotionFrame = 40
OG_green = imgs[overallMotionFrame]


for i in fileList:
    #load in the images
    startTime = time.perf_counter()
    print('loading images...')
    if i==fileList[0]:
        green = imgs
        del imgs
        blue = fn.load_images(i.replace('#G','#B'))
    else:
        green = fn.load_images(i)
        blue = fn.load_images(i.replace('#G','#B'))        

    print('Correcting the motion')
    # do the image correction (uses movement in the green, and applies to both)
    ws = fn.create_warp_stack(green)
    green_corr = fn.apply_warping_fullview(green,ws)
    blue_corr = fn.apply_warping_fullview(blue,ws)
    del blue,green
    #print('blue shape',blue_corr[10].shape,',and green shape', green_corr[10].shape)
    #tiff.imwrite(i.replace('.tif','motCorr.tif'),blue_corr)

    print('Correcting for overall motion')
    ws = fn.create_single_warp([green_corr[overallMotionFrame],OG_green],len(green_corr))
    green_corr = fn.apply_warping_fullview(green_corr,ws)
    blue_corr = fn.apply_warping_fullview(blue_corr,ws)

    print('spacial gaussian blur')
    blue_corr = fn.spacial_gaussian_filter(blue_corr,7)
    green_corr = fn.spacial_gaussian_filter(green_corr,7)

    print('averaging the frames...')
    # averaging frames
    indexes = [25,75] # This is where you change the baseline images
    background_green = fn.averaging_frames(green_corr,indexes[0],indexes[1])
    background_blue = fn.averaging_frames(blue_corr,indexes[0],indexes[1])
    print('getting the df/d0')
    # getting the df/d0
    #green_corr = green_corr[indexes[1]:]
    #blue_corr = blue_corr[indexes[1]:]
    green = fn.calculate_df_fO(green_corr,mask,background_green)
    blue = fn.calculate_df_fO(blue_corr,mask,background_blue)
    del green_corr,blue_corr

    # Channel subtraction
    blue = fn.boxCrop(polyData,blue,10)
    green = fn.boxCrop(polyData,green,10)

    #blue = fn.spacial_gaussian_filter(blue,7)
    #green = fn.spacial_gaussian_filter(green,7)

    print('low pass filter')
    blue = fn.temporal_gaussian_blur(blue,10)
    green = fn.temporal_gaussian_blur(green,10)
    
    print('getting the channel subtraction')
    signal = fn.channel_subtraction(blue,green)
    del green,blue


    print('GSR and then saving')
    #applying the gsr
    final_data = fn.gsr_correction(signal)
    tiff.imwrite(i.replace('#G.tif','processed_pointy.tif'),final_data.astype(np.float16))
    #np.save(i.replace('.tif','processed.npy'),final_data)
    print('Total time taken:', round(time.perf_counter()-startTime,1))





