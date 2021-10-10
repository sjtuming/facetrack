import cv2
import os
import numpy as np
from PIL import Image
def getImageAndLabels(path):
    facesSamples=[]
    ids=[]
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #检测人脸
    face_detector = cv2.CascadeClassifier('D:/Python/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    #打印数组imagePaths
    print('数据排列：',imagePaths)
    #遍历列表中的图片
    for imagePath in imagePaths:
        #打开图片,黑白化
        PIL_img=Image.open(imagePath).convert('L')#L表示打开模式
        #将图像转换为数组，以黑白深浅
        #PIL_img = cv2.resize(PIL_img, dsize=(400, 400))
        img_numpy=np.array(PIL_img,'uint8')
        #获取图片人脸特征
        faces = face_detector.detectMultiScale(img_numpy)
        #获取每张图片的id和姓名
        id = int(os.path.split(imagePath)[1].split('.')[0])
        #预防无面容照片
        for x,y,w,h in faces:
            ids.append(id)
            facesSamples.append(img_numpy[y:y+h,x:x+w])
        #打印脸部特征和id
        #print('fs:', facesSamples)
        print('id:', id)
        #print('fs:', facesSamples[id])
    print('fs:', facesSamples)
    #print('脸部例子：',facesSamples[0])
    #print('身份信息：',ids[0])
    return facesSamples,ids

if __name__ == '__main__':
    path = './'                                  #图片路径
    faces,ids=getImageAndLabels(path)                  #获取图像路径和ID数组签名
    recognizer=cv2.face.LBPHFaceRecognizer_create()    #加载识别器
    recognizer.train(faces,np.array(ids))              #训练
    recognizer.write('./Train.yml')              #保存训练文件
