import os
import pickle
import gzip
import csv
import sys
import random
import scipy
import numpy as np
import dicom
import cv2
import glob
import math
from collections import defaultdict
import pickle
import pylab
from skimage.restoration import denoise_bilateral
from skimage import transform
import pylab
import cv2
import matplotlib.pyplot as plt

#Create folder in the system
def mkdir(fname):
    try:
        os.mkdir(fname)
    except:
        pass

favourite_color = {"lion":"yellow", "kitty":"red"}
outputfilename = "C:\\Users\\cheun\\Desktop\\TestFolder\\test1\\save.txt.gz"
def save(object, filename, bin=1):
    tempfile = gzip.GzipFile(filename,"wb")
    try:
        tempfile.write(pickle.dumps(object,bin))
    finally:
        tempfile.close()

root_path = "C:\\Users\\Zheng Zhang\\Desktop\\TestFolder\\1"
#root_path = "C:\\Users\\cheung\\Desktop\\TestFolder\\1"
#train_csv_path = "C:\\Users\\cheung\\Desktop\\TestFolder\\train.csv"
train_csv_path = "C:\\Users\\Zheng Zhang\\Desktop\\TestFolder\\train.csv"
train_label_csv = "C:\\Users\\Zheng Zhang\\Desktop\\TestFolder\\train-label.csv"
#train_label_csv = "C:\\Users\\cheung\\Desktop\\TestFolder\\train-label.csv"
#Frame refers to the file address, sort the file address and
#based on the patients number
def get_frames(root_path):
    i = 0
    t = 0
    counter = 0
    counter2 = 0
    filesColl = []
    ret = []
    for root, _, files in os.walk(root_path):
        if len(files) == 0 or not files[0].endswith(".dcm"):
                continue
        prefix = files[0].rsplit('-',1)[0] #Take the patient name
        #set object of constructing and manipulating unordered collections of unique elements
        fileset = set(files) 
        expected =["%s-%04d.dcm" % (prefix,i + 1) for i in range(30)]
        if all(x in fileset for x in expected):
            ret.append([root + "/" + x for x in expected])
            counter += 1
            counter2 += 1
    return sorted(ret, key = lambda x: x[0])

def get_label_map(fname):
    labelmap = {}
    fi = open(fname)
    fi.readline()
    for line in fi:
        arr = line.split(',')
        labelmap[int(arr[0])] = line
    return labelmap

#print label_map
def write_label_csv(fname, frames, label_map):
    fo = open(fname, "w")
    for lst in frames:
        #print(lst[0])
        index = int(lst[0].split("\\")[5])
        #print label_map[index]
        if label_map != None:
            fo.write(label_map[index])
        else:
            fo.write("%d,0,0\n" % index)
    fo.close()

PatientSex = {}
PatientAge = {}
SliceLocation = {}

show_images = False
show_circles = False
show_combined_centers = False
show_main_center = False
center_distance_devider = 3
best_cluster_diviter = 25.0
contour_roundness = 2.5
black_removal = 200
max_area_devider = 12
minimal_median = 0
xmeanspacing = 1.25826490244
ymeanspacing = 1.25826490244

def crop_resize_other(img, pixelspacing):#normalize image
        #----------------
        #thresholdval = 20
        #r,g,b = img.splitChannels()
        #img = g.equalize().threshold(thresholdval).invert()
        #img.show()
        #---------------

        xmeanspacing = 1.25826490244
        ymeanspacing = 1.25826490244

        pixelspacing = (PixelSpacingx, PixelSpacingy)
        xmeanspacing = float(xmeanspacing)
        ymeanspacing = float(ymeanspacing)
        xscale = float(pixelspacing[0])/xmeanspacing
        yscale = float(pixelspacing[1])/ymeanspacing
        xnewdim = round(xscale*np.array(img).shape[0])
        ynewdim = round(yscale*np.array(img).shape[1])
        img = transform.resize(img,(xnewdim, ynewdim))
        img = np.uint8(img*255)

        #img = denoise_bilateral(img, sigma_color=0.05, sigma_spatial=15,multichannel=False)
        #img = denoise_bilateral(img,sigma_range=0.05,multichannel=False)
        if img.shape[0] < img.shape[1]:
            img = img.T
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy:yy + short_edge,xx:xx + short_edge]
        crop_img *= 255
       
        return crop_img.astype("uint8")
        

        #--------------Show dicom image---------------
        #pylab.imshow(f.pixel_array,cmap=pylab.cm.bone)
        #pylab.show()
        #-------------------------------------------


frames = get_frames(root_path)
label_map = get_label_map(train_csv_path)
#print frames
#print label_map

for lst in frames:
    data = []
    imglist = []
    circlesall = []
    for path in lst:
        f = dicom.read_file(path)
        (PixelSpacingx, PixelSpacingy) = f.PixelSpacing
        (PixelSpacingx, PixelSpacingy) = (float(PixelSpacingx), float(PixelSpacingy))
        pixelspacing = (PixelSpacingx, PixelSpacingy)    
        img = f.pixel_array.astype('uint8')
        #print f.PixelSpacing

        #print 'img: ',img
        #cv2.imshow('img',img)
        #cv2.waitKey()
        img = cv2.equalizeHist(img)
        imglist.append(crop_resize_other(img,pixelspacing))
        #print "working"
        #cv2.imshow('img',img)
        #cv2.waitKey()
        #print imglist
    for img in imglist:
        v = np.median(img)
        upper = int(min(255,(1.0 + 5) * v))
        print("upper {}".format(upper))
        i = 40

        while True:
            circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,50,param1=upper,param2=i,minRadius=0,maxRadius=40)

            i -= 1
            if circles is None:
                pass
            else:
                circles = np.uint16(np.around(circles))
                break

    for i in circles[0,:]:
    # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
    cv2.imshow('detected circles',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print 'circles ',circles
    break



    """
    print "working"
    tempimg = np.array(imglist)[1]
    #tempimgg = cv2.cvtColor(tempimg,cv2.COLOR_BGR2GRAY)
    v = np.median(tempimg)
    upper = int(min(255, (1.0 + 5) * v))
    i=40
    #circles = cv2.HoughCircles(tempimg,cv2.cv.CV_HOUGH_GRADIENT,1,50,param1=upper,param2=i,minRadius=0,maxRadius=40)
    
    #cv2.imshow('circles',circles)
    pylab.imshow(tempimg,cmap=pylab.cm.bone)
    pylab.show()
    #cv2.imshow('circle',tempimg)
    #circles = cv2.HoughCircles(tempimg,cv2.HOUGH_GRADIENT,1,50,param1=upper,param2=i,minRadius=0,maxRadius=40)
    #cv2.imshow('circle',circles)

    break
    #---------print and show the image after normalization---------------------
    #print "imglist",np.array(imglist)[1]
    #pylab.imshow(np.array(imglist)[1],cmap=pylab.cm.bone)
    #pylab.show()
    #-----------------------------------------------------------------------
    """
