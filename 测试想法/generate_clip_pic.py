import datetime
import os
import h5py
import numpy as np
import cv2
from PIL import Image
import time

# f = h5py.File('path/filename.h5','r') #打开h5文件
f = h5py.File('/home/wuxr/graspping/gpd-master/pytorch/gpd_orgin/data/for_test.h5','r')

print(f['images'][:].shape)
print(f['labels'][:].shape)

img1=f['images'][2005:2006]
print(f['labels'][2005:2006])
#img_type=(img1[-1,:,:,:3])
#print(img_type.shape)

img_1=(img1[-1,:,:,0:1])
img_2=(img1[-1,:,:,1:2])
img_3=(img1[-1,:,:,2:3])
img_4=(img1[-1,:,:,3:4])
img_5=(img1[-1,:,:,4:5])
img_6=(img1[-1,:,:,5:6])
img_7=(img1[-1,:,:,6:7])
img_8=(img1[-1,:,:,7:8])
img_9=(img1[-1,:,:,8:9])
img_10=(img1[-1,:,:,9:10])
img_11=(img1[-1,:,:,10:11])
img_12=(img1[-1,:,:,11:12])


print(img_2.shape)

# 定义函数，第一个参数是缩放比例，第二个参数是需要显示的图片组成的元组或者列表
def ManyImgs(scale, imgarray):
    rows = len(imgarray)         # 元组或者列表的长度
    cols = len(imgarray[0])      # 如果imgarray是列表，返回列表里第一幅图像的通道数，如果是元组，返回元组里包含的第一个列表的长度
    # print("rows=", rows, "cols=", cols)

    # 判断imgarray[0]的类型是否是list
    # 是list，表明imgarray是一个元组，需要垂直显示
    rowsAvailable = isinstance(imgarray[0], list)

    # 第一张图片的宽高
    width = imgarray[0][0].shape[1]
    height = imgarray[0][0].shape[0]
    # print("width=", width, "height=", height)

    # 如果传入的是一个元组
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                # 遍历元组，如果是第一幅图像，不做变换
                if imgarray[x][y].shape[:2] == imgarray[0][0].shape[:2]:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (0, 0), None, scale, scale)
                # 将其他矩阵变换为与第一幅图像相同大小，缩放比例为scale
                else:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (imgarray[0][0].shape[1], imgarray[0][0].shape[0]), None, scale, scale)
                # 如果图像是灰度图，将其转换成彩色显示
                if  len(imgarray[x][y].shape) == 2:
                    imgarray[x][y] = cv2.cvtColor(imgarray[x][y], cv2.COLOR_GRAY2BGR)

        # 创建一个空白画布，与第一张图片大小相同
        imgBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imgBlank] * rows   # 与第一张图片大小相同，与元组包含列表数相同的水平空白图像
        for x in range(0, rows):
            # 将元组里第x个列表水平排列
            hor[x] = np.hstack(imgarray[x])
        ver = np.vstack(hor)   # 将不同列表垂直拼接
    # 如果传入的是一个列表
    else:
        # 变换操作，与前面相同
        for x in range(0, rows):
            if imgarray[x].shape[:2] == imgarray[0].shape[:2]:
                imgarray[x] = cv2.resize(imgarray[x], (0, 0), None, scale, scale)
            else:
                imgarray[x] = cv2.resize(imgarray[x], (imgarray[0].shape[1], imgarray[0].shape[0]), None, scale, scale)
            if len(imgarray[x].shape) == 2:
                imgarray[x] = cv2.cvtColor(imgarray[x], cv2.COLOR_GRAY2BGR)
        # 将列表水平排列
        hor = np.hstack(imgarray)
        ver = hor
    return ver

stack=ManyImgs(1,([img_1,img_2,img_3,img_4],[img_5,img_6,img_7,img_8],[img_9,img_10,img_11,img_12]))

t = time.time()
tname = str(t)[5:10]


#cv2.namedWindow('showimage')
#cv2.imshow('img_type2',img_type2)
cv2.imshow('img_type2',stack)
cv2.imwrite("./"+str(tname)+'.png',stack)
cv2.waitKey(0)




