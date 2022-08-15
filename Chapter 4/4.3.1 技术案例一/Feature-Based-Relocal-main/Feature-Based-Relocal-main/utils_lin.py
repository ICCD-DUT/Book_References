
import numpy as np
from PIL import Image
import cv2
import open3d
import json
import math
import os

def rotation2quaternion(M):
    tr = np.trace(M)
    m = M.reshape(-1)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[7] - m[5]) / s
        y = (m[2] - m[6]) / s
        z = (m[3] - m[1]) / s
    elif m[0] > m[4] and m[0] > m[8]:
        s = np.sqrt(1.0 + m[0] - m[4] - m[8]) * 2
        w = (m[7] - m[5]) / s
        x = 0.25 * s
        y = (m[1] + m[3]) / s
        z = (m[2] + m[6]) / s
    elif m[4] > m[8]:
        s = np.sqrt(1.0 + m[4] - m[0] - m[8]) * 2
        w = (m[2] - m[6]) / s
        x = (m[1] + m[3]) / s
        y = 0.25 * s
        z = (m[5] + m[7]) / s
    else:
        s = np.sqrt(1.0 + m[8] - m[0] - m[4]) * 2
        w = (m[3] - m[1]) / s
        x = (m[2] + m[6]) / s
        y = (m[5] + m[7]) / s
        z = 0.25 * s
 
    Q = np.array([w, x, y, z]).reshape(-1)
    return Q
 
def quaternion2rotation(quat):
    # quat:(w,x,y,z)
    assert (len(quat) == 4)
    # normalize first
    quat = quat / np.linalg.norm(quat)
    a, b, c, d = quat
 
    a2 = a * a
    b2 = b * b
    c2 = c * c
    d2 = d * d
    ab = a * b
    ac = a * c
    ad = a * d
    bc = b * c
    bd = b * d
    cd = c * d
 
    # s = a2 + b2 + c2 + d2
 
    m0 = a2 + b2 - c2 - d2
    m1 = 2 * (bc - ad)
    m2 = 2 * (bd + ac)
    m3 = 2 * (bc + ad)
    m4 = a2 - b2 + c2 - d2
    m5 = 2 * (cd - ab)
    m6 = 2 * (bd - ac)
    m7 = 2 * (cd + ab)
    m8 = a2 - b2 - c2 + d2
 
    return np.array([m0, m1, m2, m3, m4, m5, m6, m7, m8]).reshape(3, 3)

def get_image(img_path):
    """
    读取image文件到numpy数组
    out：HxWx3 numpy array
    """
    data = Image.open(img_path)
    imageWidth, imageHeight = data.size
    myimage = np.zeros((imageHeight, imageWidth, 3), dtype = np.float32)
    myimage[0:imageHeight, 0:imageWidth, :] = data
    return myimage

def json2Dic():
    # 将 JSON 对象转换为 Python 字典
    dic = json.loads(json_str)
    return dic

def dic2Json():
    # Python 字典类型转换为 JSON 对象
    json_str = json.dumps(data1)
    return json_str

def readJsonFile(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def writeJsonFile(data,filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f)

def getPose4x4FromJson(filepath):
    json = readJsonFile(filepath)
    pose = np.zeros((16), dtype  =  np.float32)
    pose[0:16]=json['extrinsic']
    return pose.reshape(4,4).T

def getpose4x4txt(pose_path):
    """
    读取pose文件里的4x4位姿内容为一个np矩阵
    """
    list = []
    pose = np.zeros((4, 4), dtype  =  np.float32)
    with open(pose_path)as f:
        list = f.readlines()
    for n in range(4):
        for m in range(4):
            pose[n][m] = list[n].split("[")[-1].split("]")[0].split()[m]
    return pose

def writepose4x4txt(pose, pose_path):
    """
    写入4x4位姿到pose_path文件里
    """
    # list = []
    # pose = np.zeros((4, 4), dtype  =  np.float32)
    with open(pose_path, "w")as f:
        f.writelines(str(pose))
        #  = list[n].split()[m]
    return pose

def getpose1x7txt(pose_path):
    """
    读取pose文件里的内容为一个np矩阵
    """
    text = []
    pose = np.zeros((7), dtype  =  np.float32)
    with open(pose_path)as f:
        text = f.readline()
    for n in range(7):
            pose[n] = text.split()[n]
    return pose

def Depth2PointCloud(depth_path, pose, focal_x, focal_y, scale):
    """
    根据公式来生成对应的label
    depth_path: pngpath
    pose: 4x4 np array
    focal_x focal_y: focal 525 for 7scenes(kinect)
    scale: scale of depth， 1000.0 for 7scenes(kinect)
    """
    # pose = pose.numpy()
    depth = Image.open(depth_path)
    rMat = pose[0:3, 0:3]
    tMat = pose[0:3, 3:4] 
    # print(depth.size)
    imageWidth, imageHeight = depth.size
    mydepth = np.zeros((imageHeight, imageWidth), dtype = np.float32)#和rgb一样图像给加一个边
    mydepth[0:imageHeight, 0:imageWidth] = depth
    pointCloud = np.zeros((imageHeight, imageWidth,3), dtype = np.float32)

    for x in range(imageWidth):#按照rgb里面取得的xy来进行lable的生成，所以，和
        for y in range(imageHeight):
            point = np.array(
                    [[0], [0], [0]]
                )
            if (mydepth[y][x] != 0):#因为图像里的高宽与数组的高宽是反过来的，所以y值要在前面，要不然就会下标越界，如果深度值没有缺失，那么计算，如果缺失了则默认全0
                point = np.array(#图像坐标系转到相机坐标系
                        [[(x - ((imageWidth) / 2.0)) / (focal_x / (mydepth[y][x] / scale))], #(x-p_x)/(f_x/d),
                        [(y - ((imageHeight) / 2.0)) / (focal_y / (mydepth[y][x] / scale))], 
                        [mydepth[y][x] / scale]],dtype = np.float32)
                #相机坐标系转世界坐标系
                newMat = point - tMat
                point = rMat.T @ newMat
            pointCloud[y][x]=np.squeeze(point)
    return pointCloud

def ShowNumpy(numpyArray):
    """
    显示 HxWx3的numpy数组为图像，opencv自动0-255区间化
    """
    cv2.imshow('test', numpyArray)
    cv2.waitKey()

def RotationErr(RMat1, RMat2):
    RMat2_t = np.transpose(RMat2)
    rotDiff = np.matmul(RMat1, RMat2_t)
    trace = np.trace(rotDiff)
    high = 3.0
    low = -1.0
    if(trace > high):
        trace = high
    if(trace < low):
        trace = low
    ACos = np.arccos((trace-1.0)/2.0)
    roterr = 180.0 * ACos /3.141592653
    return roterr 

def TranslationErr(tMat1, tMat2):
    t_diff = np.linalg.norm(tMat1-tMat2,ord=2)
    return t_diff

def Ransac_opencv_source(ScenPosition, PicPosition, Pose, CameraMatrix, distCoeffs):
    '直接调用cv2的pnpRansac算法，该函数作为对比'
    """
    ScenPosition：3D世界坐标值列表 list即可
    PicPosition：对应的图像坐标值 list即可
    Pose：posede GT值
    CameraMatrix：3x3 相机内参
    """
    RanPw = np.array(ScenPosition, dtype = np.float32)   #变为307200×3
    RanPc = np.array(PicPosition, dtype = np.float32)

    _, R, trans, inliers = cv2.solvePnPRansac(RanPw, RanPc, CameraMatrix, distCoeffs, reprojectionError = 10)
    r, _ = cv2.Rodrigues(R)
    RMat1 = Pose[0:3, 0:3]
    tMat1 = Pose[0:3, 3:4]
    tErr = TranslationErr(trans, tMat1)
    rErr = RotationErr(RMat1, r)
    return tErr, rErr

def Ransac_opencv(ScenPosition, PicPosition, CameraMatrix, distCoeffs):
    '直接调用cv2的pnpRansac算法，该函数作为对比'
    """
    ScenPosition：3D世界坐标值列表 list即可
    PicPosition：对应的图像坐标值 list即可
    Pose：posede GT值
    CameraMatrix：3x3 相机内参
    """
    RanPw = np.array(ScenPosition, dtype = np.float32)   #变为307200×3
    RanPc = np.array(PicPosition, dtype = np.float32)

    _, R, trans, inliers = cv2.solvePnPRansac(RanPw, RanPc, CameraMatrix, distCoeffs, reprojectionError = 10)
    r, _ = cv2.Rodrigues(R)

    pose = np.zeros((4, 4), dtype = np.float32)

    pose[0:3, 0:3] = r
    pose[0:3, 3:4] = trans
    pose[3,3]=1
    # RMat1 = Pose[0:3, 0:3]
    # tMat1 = Pose[0:3, 3:4]
    # tErr = TranslationErr(trans, tMat1)
    # rErr = RotationErr(RMat1, r)
    return np.linalg.inv(pose)

def showPointCloudFromFile(filename):
    """
    可视化点云文件，open3d支持，
    filename: 点云文件路径
    """
    pcd = o3d.io.read_point_cloud(filename)#这里的cat.ply替换成需要查看的点云文件
    o3d.visualization.draw_geometries([pcd])

def showPointCloudFromNumpy(npPointCloud):
    """
    可视化点云numpy数组,open3d支持
    npPointCloud: numpy 点云文件
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])

def o3dPoints2Numpy(pcd):
    """
    open3D点云转numpy数组
    """
    xyz_load = np.asarray(pcd_load.points)
    return xyz_load

def numpy2O3dPoints(npPointCloud):
    """
    numpy数组转open3d点云
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def readFile2O3dPoints(filename):
    """
    读取3D文件到open3d点云文件
    """
    pcd_load = o3d.io.read_point_cloud(filename)
    return pcd_load

# test code:
# pose = getpose4x4txt("data/frame-000000.pose.txt")
# print(pose)
# PointCloud = Depth2PointCloud("data/frame-000000.depth.png",pose,525,525,1000.0)
# print(PointCloud)
