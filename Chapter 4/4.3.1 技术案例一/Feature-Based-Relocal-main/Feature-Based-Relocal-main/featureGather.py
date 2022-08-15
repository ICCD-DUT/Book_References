from wsgiref import headers
import cv2
from cv2 import KeyPoint
import numpy as np
import argparse
import os
from utils import *
from utils_lin import *


def readAndRescale(img1, img2, scale):
    """Helper to read images, scale them and convert to grayscale.
        it returns original, gray and scaled images

    Typical use:
        t, s, t_gray, s_gray, t_full, s_full = readAndRescale("cat1.jpg", "cat2.jpg", 0.3)

    img1: target image name
    img2: source image name
    scale: scaling factor, keeping aspect ratio
    """
    target = cv2.imread(img1)
    source = cv2.imread(img2)

    width = int(target.shape[1] * scale)
    height = int(source.shape[0] * scale)
    dim = (width, height)

    target_s = cv2.resize(target, dim, interpolation=cv2.INTER_AREA)
    source_s = cv2.resize(source, dim, interpolation=cv2.INTER_AREA)

    gray1 = cv2.cvtColor(target_s, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(source_s, cv2.COLOR_BGR2GRAY)

    return target_s, source_s, gray1, gray2, target, source


def getKeypointAndDescriptors(target_gray, source_gray):
    """Helper to get Harris points of interest and use them as landmarks for sift descriptors.
        it returns these landmarks and their descriptors

    Typical use:
        lmk1, lmk2, desc1, desc2 = getKeypointAndDescriptors(target_gray, source_gray)

    target_gray, source_gray: grayscaled target and source images as np.ndarray
    """
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(target_gray, None)
    pts1 = np.array([kp1[idx].pt for idx in range(len(kp1))])
    kp2, des2 = sift.detectAndCompute(source_gray, None)
    pts2 = np.array([kp2[idx].pt for idx in range(len(kp2))])
    return pts1, pts2, des1, des2

def ORB(img):
     # Initiate ORB detector
    orb = cv2.ORB_create()
 
     # find the keypoints with ORB
    kp = orb.detect(img, None)

     # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    kp = np.array([kp[idx].pt for idx in range(len(kp))])
    return kp, des 

def match(lmk1, lmk2, desc1, desc2, orb_error=0.7):
    """Helper to find the pair of matches between two keypoints lists
    it return two np.ndarray of landmarks in an order respecting the matching

    Typical use:
        lmk1, lmk2 = match(lmk1, lmk2, desc1, desc2)

    lmk1: landmarks of target and source images respectively as np.ndarray
    PCmk2: pointcloud of desc2
    desc1, desc2: descriptors of target and source images respectively as np.ndarray
    sift_error: if the ratio between the distance to the closest match and the second closest is less than sift_error
    reject this landmark.
    """
    match1, match2, matchdesc2 = [], [], []
    for i in range(len(desc1)):
        distance = np.sqrt(np.sum((desc1[i] - desc2) ** 2, axis=1))
        indices = np.argsort(distance)
        if distance[indices[0]] / distance[indices[1]] < orb_error:
            match1.append(lmk1[i])
            match2.append(lmk2[indices[0]])
            matchdesc2.append(desc2[indices[0]])
    return np.array(match1), np.array(match2), np.array(matchdesc2)

def ransac(kp1, kp2, matchdesc2):
    """Helper to apply ransac (RANdom SAmple Consensus) algorithm on two arrays of landmarks
    it returns the inliers and outliers in both arrays

    Typical use:
        lmk1, lmk2, outliers1, outliers2 = ransac(lmk1, lmk2)

    kp1, kp2: landmarks of target and source images respectively as np.ndarray
    """
    ransac_model = linear_model.RANSACRegressor()
    ransac_model.fit(kp1, kp2)
    inlier_mask = ransac_model.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    return kp1[inlier_mask], kp2[inlier_mask], kp1[outlier_mask], kp2[outlier_mask], matchdesc2[inlier_mask]

def get3D_Desc(S2D, S3D,desc):
    OT3d=[]
    OTdesc=[]
    for i,item in enumerate(S2D):
        if S3D[int(item[1])][int(item[0])][0]!=0 and S3D[int(item[1])][int(item[0])][1]!=0 and S3D[int(item[1])][int(item[0])][2]!=0:
            OT3d.append(S3D[int(item[1])][int(item[0])])
            OTdesc.append(desc[i])
    return np.array(OT3d), np.array(OTdesc)

def get3D_2DofSIFT(T2D,S3D,S2D):
    OT3D=[]
    OT2D=[]
    for i,item in enumerate(S2D):
        if S3D[int(item[1])][int(item[0])][0]!=0 and S3D[int(item[1])][int(item[0])][1]!=0 and S3D[int(item[1])][int(item[0])][2]!=0:
            OT3D.append(S3D[int(item[1])][int(item[0])])
            OT2D.append(T2D[i])
    return OT3D, OT2D

for scenename in ["fire","heads","office","pumpkin","redkitchen","stairs"]:

    rgb_dir = "datasets\\7scenes_"+scenename+"\\train\\rgb\\"
    rgb_files = os.listdir(rgb_dir)
    rgb_files = [rgb_dir + f for f in rgb_files]
    rgb_files.sort()

    depth_dir = "datasets\\7scenes_"+scenename+"\\train\\depth\\"
    depth_files = os.listdir(depth_dir)
    depth_files = [depth_dir + f for f in depth_files]
    depth_files.sort()

    pose_dir = "datasets\\7scenes_"+scenename+"\\train\\poses\\"
    pose_files = os.listdir(pose_dir)
    pose_files = [pose_dir + f for f in pose_files]
    pose_files.sort()

    Datasets_3D = []
    Datasets_des = []
    CameraMatrix = np.array([[525, 0, 324],
                        [0, 525, 244],
                        [0, 0, 1]], dtype=np.float32)
    for rgb_index in range(1,len(rgb_files),50):
        print(rgb_index)
        target, source, target_gray, source_gray, target_full, source_full = readAndRescale(rgb_files[rgb_index],rgb_files[rgb_index+1],1)
        lmk1, desc1 = ORB(target_gray)
        lmk2, desc2 = ORB(source_gray)
        lmk1_1, lmk2_2, matchdesc = match(lmk1, lmk2, desc1, desc2)
        lmk1, lmk2, outliers1, outliers2, inliermatchdesc = ransac(np.array(list(lmk1_1)), np.array(list(lmk2_2)), matchdesc)

        pose_S = getpose4x4txt(pose_files[rgb_index+1])
        PointCloud_S = Depth2PointCloud(depth_files[rgb_index+1], np.linalg.inv(pose_S), 525, 525, 1000.0 )

        OT3d, OTdesc = get3D_Desc(lmk2,PointCloud_S,inliermatchdesc)
        if len(Datasets_3D)==0:
            Datasets_3D=OT3d
            Datasets_des=OTdesc
        elif len(OT3d)>0:
            Datasets_3D=np.concatenate((Datasets_3D,OT3d),0)
            Datasets_des=np.concatenate((Datasets_des,OTdesc),0)
        print(Datasets_3D.shape)
        print(Datasets_des.shape)

    np.save("Datasets_3D_"+scenename+".npy",Datasets_3D)
    np.save("Datasets_des_"+scenename+".npy",Datasets_des)