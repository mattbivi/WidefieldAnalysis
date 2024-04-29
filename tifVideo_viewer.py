import tifffile as tiff
import cv2,glob
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
import pickle as plk

#file = tiff.imread("AS_WF186_Bin2_GoNoGo_Day8JCamF15processed_wSGB_diffCrop.tif")
#file = np.load('AS_WF193_Bin2_GoNoGo_Day9JCamF21_green_adj.tif')
frames = fn.load_images("SofiesFunctions/images/AS_WF186_Bin2_GoNoGo_Day8JCamF16#G.tif")

smoothFrames = fn.channel_rescale(frames,255)
file = np.stack(smoothFrames, axis=0)

file = file.astype(np.uint8)
thing = file.shape
cv2.imshow('file',file[0,:,:])

cv2.waitKey(1000)

for i in range(thing[0]):
    img = cv2.applyColorMap(file[i,:,:],cv2.COLORMAP_JET)
    cv2.imshow('file',img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
 