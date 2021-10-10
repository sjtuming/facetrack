import numpy as np
import cv2 as cv
import time  
import RPi.GPIO as GPIO
import Adafruit_PCA9685
import threading
import importlib,sys
#导入必要的库文件
importlib.reload(sys)
servo_pwm = Adafruit_PCA9685.PCA9685()
#设置二维舵机对象
servo_pwm.set_pwm_freq(50)  # 设置频率
servo_pwm.set_pwm(4,0,325)  # 底座舵机
servo_pwm.set_pwm(9,0,325)  # 倾斜舵机
face_cascade = cv.CascadeClassifier('./haar/haarcascade_frontalface_default.xml')
#设置调用级联分级器对象并调用opencv自带的Haar文件
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('./Train.yml')
#设置比对方法和训练文件
cap = cv.VideoCapture(0)
#初始化摄像头
usb_cap.set(3, 640)
usb_cap.set(4, 480)
#设置视频分辨率
pid_x=0
pid_y=0
pid_w=0
pid_h=0
#舵机云台的每个自由度需要4个变量
pid_thisError_x=0   #当前误差值
pid_lastError_x=0   #上次误差值
pid_thisError_y=0
pid_lastError_y=0
#舵机初始角度
pid_X_P = 330
pid_Y_P = 330
#下达舵机转动角度
pid_flag=0
size=0
#图片尺寸，用于判断大小
makerobo_facebool = False
#判断要不要让舵机执行转向
GPIO.setwarnings(False)
# 机器人舵机旋转
def Robot_servo():
    while True:
        servo_pwm.set_pwm(4,0,595-pid_X_P)
        servo_pwm.set_pwm(9,0,80+pid_Y_P)
#定义舵机初始位置
servo_tid=threading.Thread(target=Robot_servo)  # 多线程
servo_tid.setDaemon(True)
servo_tid.start()                               # 开启线程
while (usb_cap.isOpened()):                     #先看看摄像头开了没
    size=0
    ret,img = usb_cap.read()                    #获取当前帧
    face = np.zeros((640,640,3),np.uint8)       #统一用三通道方图处理图片
    face[80:560,0:640] = img[0:480,0:640]       #图片无损接入
    gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)#灰度处理
    faces = face_cascade.detectMultiScale(gray,1.3,3,0,(50,50),(400,400))#设置好合适的人脸参数
    if len(faces)==0:#faces容器内没有任何元素，没有检测到人任何一张脸
        face=np.rot90(face)                     #由于手机支架只有横竖两种工作模式，尝试将图片矩阵反转90°
        gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)  # 灰度处理
        faces = face_cascade.detectMultiScale(gray,1.3,5,0,(50,50),(400,400))
        if len(faces)==0:
            pid_x,pid_y,pid_w,pid_h = 320,240,0,0
            #如果横竖都没有检测到人脸就定位在中央，舵机不再转动
    if len(faces)!=0:
        idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        if confidence > 30:
            pid_x, pid_y, pid_w, pid_h = 320, 240, 0, 0
            #当前图片即使有人脸，但也不是我们需要追踪的对象
        else
            for x,y,w,h in faces:
                if w*h > size:
                    pid_x,pid_y,pid_w,pid_h = x,y,w,h
                    size = w*h
    #找到最大人脸坐标
    face=np.array(face)
    cv.rectangle(face,(pid_x,pid_y),(pid_x+pid_w,pid_y+pid_h),(0,255,0),1)#用矩形方框标记出人脸
    result=(pid_x,pid_y,pid_w,pid_h)
    pid_x=result[0]+pid_w/2
    pid_y=result[1]+pid_h/2
    makerobo_facebool = True

    # 误差值处理，根据舵机实际情况调整
    pid_thisError_x=pid_x-160
    pid_thisError_y=pid_y-120

    #自行对P和D两个值进行调整，以达到最稳定的追踪状态
    pwm_x = pid_thisError_x*5+1*(pid_thisError_x-pid_lastError_x)
    pwm_y = pid_thisError_y*5+1*(pid_thisError_y-pid_lastError_y)
        
    #迭代误差值操作
    pid_lastError_x = pid_thisError_x
    pid_lastError_y = pid_thisError_y

    #舵机拟需转动角度
    pid_XP=pwm_x/100
    pid_YP=pwm_y/100
        
    # pid_X_P pid_Y_P 为最终PID值
    pid_X_P=pid_X_P+int(pid_XP)
    pid_Y_P=pid_Y_P+int(pid_YP)
        
    #限值舵机在一定的范围之内
    if pid_X_P>650:
        pid_X_P=650
    if pid_X_P<0:
        pid_X_P=0
    if pid_Y_P>650:
        pid_Y_P=650
    if pid_Y_P<0:
        pid_Y_P=0
    cv.imshow("摄像头实时页面", face)
    if cv.waitKey(1)==119:
        break
print("\n [INFO] Exiting Program and cleanup stuff \n")
#释放资源
usb_cap.release()
cv.destroyAllWindows()