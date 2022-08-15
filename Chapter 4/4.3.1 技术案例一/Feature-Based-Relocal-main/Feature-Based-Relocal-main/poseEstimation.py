import cv2
from cv2 import KeyPoint
import numpy as np
import argparse
import os
from utils import *
from utils_lin import *
import time


def readAndRescale(img1, scale):
    """Helper to read images, scale them and convert to grayscale.
        it returns original, gray and scaled images

    Typical use:
        t, s, t_gray, s_gray, t_full, s_full = readAndRescale("cat1.jpg", "cat2.jpg", 0.3)

    img1: target image name
    img2: source image name
    scale: scaling factor, keeping aspect ratio
    """
    # target = cv2.imread(os.path.join("data", img1))
    # source = cv2.imread(os.path.join("data", img2))
    target = cv2.imread(img1)

    width = int(target.shape[1] * scale)
    height = int(target.shape[0] * scale)
    dim = (width, height)

    target_s = cv2.resize(target, dim, interpolation=cv2.INTER_AREA)

    gray1 = cv2.cvtColor(target_s, cv2.COLOR_BGR2GRAY)

    return target_s, gray1, target

def ORB(img):
     # Initiate ORB detector
    orb = cv2.ORB_create()
 
     # find the keypoints with ORB
    kp = orb.detect(img, None)

     # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    kp = np.array([kp[idx].pt for idx in range(len(kp))])
    return kp, des 

def match(lmk1, desc1, desc2, S3D, orb_error=0.9):
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
    match2d, match3d, matchdesc2 = [], [], []
    for i in range(len(desc1)):
        distance = np.sqrt(np.sum((desc1[i] - desc2) ** 2, axis=1))
        indices = np.argsort(distance)
        if distance[indices[0]] / distance[indices[1]] < orb_error:
            match2d.append(lmk1[i])
            match3d.append(S3D[indices[0]])
            matchdesc2.append(desc2[indices[0]])
    return np.array(match2d), np.array(match3d), np.array(matchdesc2)

for scenename in ["fire","heads","office","pumpkin","redkitchen","stairs"]:
    print(scenename)
    rgb_dir = "datasets\\7scenes_"+scenename+"\\test\\rgb\\"
    rgb_files = os.listdir(rgb_dir)
    rgb_files = [rgb_dir + f for f in rgb_files]
    rgb_files.sort()

    depth_dir = "datasets\\7scenes_"+scenename+"\\test\\depth\\"
    depth_files = os.listdir(depth_dir)
    depth_files = [depth_dir + f for f in depth_files]
    depth_files.sort()

    pose_dir = "datasets\\7scenes_"+scenename+"\\test\\poses\\"
    pose_files = os.listdir(pose_dir)
    pose_files = [pose_dir + f for f in pose_files]
    pose_files.sort()

    Datasets_3D = np.load("Datasets_3D_"+scenename+".npy")
    Datasets_des = np.load("Datasets_des_"+scenename+".npy")

    CameraMatrix = np.array([[525, 0, 324],
                        [0, 525, 244],
                        [0, 0, 1]], dtype=np.float32)
    rErrs = []
    tErrs = []
    avg_time = 0

    pct5 = 0
    pct2 = 0
    pct1 = 0

    for rgb_index in range(len(rgb_files)):
        start_time = time.time()
        target, target_gray, target_full = readAndRescale(rgb_files[rgb_index],1)
        lmk1, desc1 = ORB(target_gray)

        match2d,match3d,matchdesc = match(lmk1,desc1,Datasets_des,Datasets_3D)
        distCoeffs = np.zeros(5, dtype = np.float32)
        
        tPosePred = Ransac_opencv(match3d, match2d, CameraMatrix, distCoeffs)
        avg_time += time.time()-start_time
        
        # Test part: GT pose
        pose_T = getpose4x4txt(pose_files[rgb_index])

        RMat_T = pose_T[0:3, 0:3]
        tMat_T = pose_T[0:3, 3:4]
        RMat_P = tPosePred[0:3, 0:3]
        tMat_P = tPosePred[0:3, 3:4]
        tErr = TranslationErr(tMat_T, tMat_P)
        rErr = RotationErr(RMat_T, RMat_P)
        
        rErrs.append(rErr)
        tErrs.append(tErr * 100)
        if rErr < 5 and tErr < 0.05:
            pct5 += 1
        if rErr < 2 and tErr < 0.02:
            pct2 += 1
        if rErr < 1 and tErr < 0.01:
            pct1 += 1

    median_idx = int(len(rErrs)/2)
    tErrs.sort()
    rErrs.sort()
    avg_time /= len(rErrs)

    print("\n===================================================")
    print("\n"+scenename+"Test complete.")

    print('\nAccuracy:')
    print('\n5cm5deg: %.1f%%' %(pct5 / len(rErrs) * 100))
    print('2cm2deg: %.1f%%' % (pct2 / len(rErrs) * 100))
    print('1cm1deg: %.1f%%' % (pct1 / len(rErrs) * 100))

    print("\nMedian Error: %.1fdeg, %.1fcm" % (rErrs[median_idx], tErrs[median_idx]))
    print("Avg. processing time: %4.1fms" % (avg_time * 1000))