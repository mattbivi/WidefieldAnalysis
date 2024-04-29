import tifffile as tiff
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from scipy.ndimage import gaussian_filter1d as gaussianFilter
from scipy import signal
from numba import njit
from numpy import pi, sqrt, exp



def deinterlacer(file):
    # One file at at time
    # File zero will always be green and one will be blue
    file1 = file[::2]
    file2 = file[1::2]
    checker = [np.median(file[0]),np.median(file[1])]
    
    if checker[0]>checker[1]:
        return(file1,file2)
    elif checker[0]<checker[1]:
        return (file2,file1)

def boxCrop(polyData,frames,extra):
    minMaxList = np.zeros((len(polyData),4))    
    for i,j in enumerate(polyData):
        tempArray = np.array(j)
        minMaxList[i,:] = np.array([tempArray[0][:,0].max(),tempArray[0][:,0].min(),tempArray[0][:,1].max(),tempArray[0][:,1].min()])
    var = [minMaxList[:,0].max()+extra,minMaxList[:,1].min()-extra,minMaxList[:,2].max()+extra,minMaxList[:,3].min()-extra]
    newFrames = list()
    for i in frames:
        newFrames.append(i[int(var[3]):int(var[2]),int(var[1]):int(var[0])])
    return newFrames


#Motion correction section
def load_images(PATH):
    imgMat = tiff.imread(PATH)
    imgMat = imgMat.astype(np.float32)
    imgs = list()

    for i in range(imgMat.shape[0]):
        imgs.append(imgMat[i,:,:])
    return imgs


def get_border_pads(img_shape, warp_stack):
    maxmin = []
    corners = np.array([[0,0,1], [img_shape[1], 0, 1], [0, img_shape[0],1], [img_shape[1], img_shape[0], 1]]).T
    warp_prev = np.eye(3)
    for warp in warp_stack:
        warp = np.concatenate([warp, [[0,0,1]]])
        warp = np.matmul(warp, warp_prev)
        warp_invs = np.linalg.inv(warp)
        new_corners = np.matmul(warp_invs, corners)
        xmax,xmin = new_corners[0].max(), new_corners[0].min()
        ymax,ymin = new_corners[1].max(), new_corners[1].min()
        maxmin += [[ymax,xmax], [ymin,xmin]]
        warp_prev = warp.copy()
    maxmin = np.array(maxmin)
    bottom = maxmin[:,0].max()
    #print('bottom', maxmin[:,0].argmax()//2)
    top = maxmin[:,0].min()
    #print('top', maxmin[:,0].argmin()//2)
    left = maxmin[:,1].min()
    #print('right', maxmin[:,1].argmax()//2)
    right = maxmin[:,1].max()
    #print('left', maxmin[:,1].argmin()//2)
    return int(-top), int(bottom-img_shape[0]), int(-left), int(right-img_shape[1])

### CORE FUNCTIONS
## FINDING THE TRAJECTORY
def get_homography(imga, imgb, motion = cv2.MOTION_EUCLIDEAN):
    imga = cv2.normalize(imga, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    imgb = cv2.normalize(imga, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if len(imga.shape) == 3:
        imga = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
    if len(imgb.shape) == 3:
        imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
    if motion == cv2.MOTION_HOMOGRAPHY:
        warpMatrix=np.eye(3, 3, dtype=np.float32)
    else:
        warpMatrix=np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50,  0.001)
    warp_matrix = cv2.findTransformECC(templateImage=imga,inputImage=imgb,warpMatrix=warpMatrix, motionType=motion,criteria=criteria,inputMask=None,gaussFiltSize=3)[1]
    return warp_matrix 

def get_homography_pointy(imga,imgb):
    imga = cv2.normalize(imga, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    imgb = cv2.normalize(imga, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    lk_params = dict( winSize = (15, 15), 
                    maxLevel = 2, 
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                                10, 0.03)) 

    first_points = cv2.goodFeaturesToTrack(imga,100,0.1,minDistance=10,blockSize=3,useHarrisDetector=True)
    second_points, status, err = cv2.calcOpticalFlowPyrLK(imga,imgb, first_points ,None, **lk_params)
    transform, thing = cv2.estimateAffinePartial2D(first_points,second_points)
    return transform

def create_warp_stack(imgs):
    warp_stack = []
    #for i, img in enumerate(imgs[:-1]):
    for i in range(len(imgs)-1):
        warp_stack += [get_homography_pointy(imgs[i], imgs[i+1])]
    return np.array(warp_stack)


def create_single_warp(imgs,length):
    warp = get_homography_pointy(imgs[0], imgs[1])
    warp_stack = [warp]
    eye = np.array([[1,0,0],[0,1,0]])
    for i in range(length):
        warp_stack += [eye]
    return np.array(warp_stack)


def homography_gen(warp_stack):
    H_tot = np.eye(3)
    wsp = np.dstack([warp_stack[:,0,:], warp_stack[:,1,:], np.array([[0,0,1]]*warp_stack.shape[0])])
    for i in range(len(warp_stack)):
        H_tot = np.matmul(wsp[i].T, H_tot)
        yield np.linalg.inv(H_tot)#[:2]
        

## APPLYING THE SMOOTHED TRAJECTORY TO THE IMAGES
def apply_warping_fullview(images, warp_stack):
    top, bottom, left, right = get_border_pads(img_shape=images[0].shape, warp_stack=warp_stack)
    H = homography_gen(warp_stack)
    imgs = []
    for i, img in enumerate(images[1:]):
        H_tot = next(H)+np.array([[0,0,left],[0,0,top],[0,0,0]])
        img_warp = cv2.warpPerspective(img, H_tot, (img.shape[1]+left+right, img.shape[0]+top+bottom))
        imgs.append(img_warp)
    return imgs

def motion_correction(imgs):
    ws = create_warp_stack(imgs)
    corr_imgs = apply_warping_fullview(imgs,ws)
    return corr_imgs

class PolygonDrawer(object):
    def __init__(self):
        self.window_name = 'Image Cropper' # Name for our window
        self.done = False # Flag signalling we're done
        self.next = False # Flag signalling that we're going onto the next shape
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.prevPoints = []
        self.points = [] # List of points defining our polygon
        self.CANVAS_SIZE = None
        self.LINE_COLOR = (255, 255, 255)

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True


    def run(self,img):
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.CANVAS_SIZE = img.shape

        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, np.zeros(self.CANVAS_SIZE, np.uint8))
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            canvas = img.copy() #np.zeros(self.CANVAS_SIZE, np.uint8)
            if (len(self.prevPoints)>0):
                    for i in self.prevPoints:
                        cv2.polylines(canvas,i,False,self.LINE_COLOR,2)
            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, self.LINE_COLOR, 2)
                # And  also show what the current segment would look like
                cv2.line(canvas, self.points[-1], self.current,self.LINE_COLOR,2)
            # Update the window
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            k = cv2.waitKey(50)
            if k == 27: # ESC hit
                self.done = True
                if self.points!=[]:
                    self.prevPoints.append(np.array([self.points]))
            elif k==32: # Hitting space bar
                self.next = True
                if self.points!=[]:
                    self.prevPoints.append(np.array([self.points]))
                self.points = []

        # User finised entering the polygon points, so let's make the final drawing
        mask = np.zeros(self.CANVAS_SIZE, np.uint8)
        # of a filled polygon
        if (len(self.points) > 0 or len(self.prevPoints) > 0):
            for i in self.prevPoints:
                cv2.fillPoly(mask, i ,self.LINE_COLOR)
            masked = cv2.bitwise_and(img, img, mask=mask)
        # And show it
        cv2.imshow(self.window_name, masked)
        # Waiting for the user to press any key
        cv2.waitKey()
        cv2.destroyWindow(self.window_name)
        
        return mask

def maskMaking(img,polygons):
    mask = np.zeros(img.shape, np.uint8)
    for i in polygons:
        cv2.fillPoly(mask, i ,(1))
    return mask

def averaging_frames(stack,index1,index2):
    avgImage = np.zeros(stack[0].shape)
    for i in stack[index1:index2]:
        avgImage = np.add(i,avgImage,dtype=np.float32)
    avgImage = avgImage/(index2-index1)
    return avgImage

def calculate_df_fO(stack,mask,background):
    dims = mask.shape
    newStack = list()
    for i in stack:
        newStack.append(np.divide(np.subtract(i,background,dtype=np.float32),background)[:dims[0],:dims[1]]*mask)
    return newStack

def channel_rescale(stack,scale):
    top = 0
    bottom = 1000
    for i in stack:
        topTemp = i.max()
        if topTemp > top:
            top = topTemp
        bottomTemp = i.min()
        if bottomTemp < bottom:
            bottom = bottomTemp
    diff = (top-bottom)/scale
    newStack = list()
    for i in stack:
        newStack.append((i-bottom)/diff)
    return newStack

#@njit(parallel=True, fastmath=True)
def temporal_gaussian_blur(stack,passes):
    frames = np.stack(stack, axis=0)
    frames = frames.astype(np.float32)
    for i in range(0,passes):
        frames[1:-1,:,:] = frames[1:-1,:,:]+frames[:-2,:,:]+frames[2:,:,:]
        frames = frames/3
    return frames



def channel_subtraction_list(stack1,stack2):
    result = list()
    for i in range(len(stack1)):
        result.append(np.subtract(stack1[i],stack2[i],dtype=np.float32))
    return result

def channel_subtraction(stack1,stack2):
    return np.subtract(stack1,stack2,dtype=np.float32)

def gsr_correction(frames):
    dims = frames[2].shape
    corr_frames = gsr(frames,dims[0],dims[1])
    return corr_frames

def lowpassfilter(stack,frequency,framerate):
    frames = np.stack(stack, axis=0)
    dims = frames.shape
    sos = signal.butter(1,frequency,'low',fs=framerate,output='sos')

    for i in range(dims[1]):
        for j in range(dims[2]):
            frames[:,i,j] = signal.sosfilt(sos,frames[:,i,j])
    return frames


def gsr(frames, width, height):
    frames[np.isnan(frames)] = 0
    # Reshape into time and space
    frames = np.reshape(frames, (frames.shape[0], width*height))
    mean_g = np.mean(frames, axis=1, dtype=np.float32)
    g_plus = np.squeeze(np.linalg.pinv([mean_g]))
    beta_g = np.dot(g_plus, frames)
    global_signal = np.dot(np.asarray([mean_g]).T, [beta_g])
    frames = frames - global_signal
    frames = np.reshape(frames, (frames.shape[0], width, height))
    return frames

def spacial_gaussian_filter(frames,filtSize):
    newStack = list()
    for i in frames:
        newStack.append(cv2.GaussianBlur(i,(filtSize,filtSize),5))
    return newStack

# Leave set origin for later. Needs debugging


