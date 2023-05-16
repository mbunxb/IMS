import cv2 as cv
import mediapipe as mp
import time
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
# from twilio.rest import Client
from requests import get, post
import sys
import requests
import json
from PIL import Image
import joblib
import logging
from tuya_connector import TuyaOpenAPI, TUYA_LOGGER

# menu = int(input('请选择检测模式：1. 打开摄像头检测\t2. 从本地导入视频检测\t3. 退出\n'))
menu = 1  # 预设模式
print('Video mode is : ', menu)

# 设定意图理解目标位置
destination = np.zeros((3,), dtype=[('dx', 'i4'), ('dy', 'i4')])
destination[0] = (0, 240)
destination[1] = (640, 240)
destination[2] = (320, 480)
print('Destinations are :', destination)

# 设置吃药喝水地点
drinkingplace = (320, 240)

# 以下为WIN路径配置
recognizer_path = 'D:\\opencv\\trainner\\trainner.yml'
face_cascade_path = 'D:\\opencv\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml'
clf_path = 'D:\\opencv\\trainner\\clfSVCov4Rand.model'
video_path = 'D:\\opencv\\testvideo\\'

# 以下为MACOS路径配置
# recognizer_path = '/Users/mbunxb/PycharmProjects/trainner/trainner.yml'
# face_cascade_path = '/Users/mbunxb/PycharmProjects/haarcascades/haarcascade_frontalface_default.xml'
# clf_path = '/Users/mbunxb/PycharmProjects/trainner/clfSVCov4Rand.model'
# video_path = '/Users/mbunxb/PycharmProjects/testvideo/'

# 导入solution
mp_pose = mp.solutions.pose
# 导入绘图函数
mp_drawing = mp.solutions.drawing_utils
# 导入人脸识别分类器
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read(recognizer_path)
face_cascade = cv.CascadeClassifier(face_cascade_path)
# 字体
font = cv.FONT_HERSHEY_SIMPLEX
names = ['xkx', 'sy', 'lzj']

# 导入模型
pose = mp_pose.Pose(static_image_mode=False,  # 是静态图片还是连续视频帧
                    model_complexity=1,  # 选择人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于两者之间
                    smooth_landmarks=True,  # 是否平滑关键点
                    enable_segmentation=True,  # 是否人体抠图
                    min_detection_confidence=0.5,  # 置信度阈值
                    min_tracking_confidence=0.9  # 追踪阈值
                    )
global occupied
global str_pose
global ax
global keypoints
global videofps
global accessToken
global user
global results

ACCESS_ID = "5jme9fvk3qqsek84wjs7"
ACCESS_KEY = "13062ec72fa44b82a4cea6302162d384"
API_ENDPOINT = "https://openapi.tuyacn.com"
MQ_ENDPOINT = "wss://mqe.tuyacn.com:8285/"

# # Enable debug log
# TUYA_LOGGER.setLevel(logging.DEBUG)

# 初始化api
openapi = TuyaOpenAPI(API_ENDPOINT, ACCESS_ID, ACCESS_KEY)
openapi.connect()

# 设定device_id
# DEVICE_ID = "vdevo168189205027299"
DEVICE_ID = "6cc25dbf915a01a266ojsb"


# # 获取指令集
# response = openapi.get("/v1.0/iot-03/devices/{}/functions".format(DEVICE_ID))

# commands = {'commands': [{'code': 'switch_led', 'value': True}]}
# openapi.post('/v1.0/iot-03/devices/{}/commands'.format(DEVICE_ID), commands)

# exit()


def send_message(to_user, access_token, poseid):
    posestr = []
    if poseid == 0:
        posestr = '站'
    elif poseid == 1:
        posestr = '卧'
    elif poseid == 2:
        posestr = '坐'
    elif poseid == 3:
        posestr = '跌倒'
    elif poseid == 4:
        posestr = '需要帮助'
    url = "https://api.weixin.qq.com/cgi-bin/message/template/send?access_token={}".format(access_token)
    data = {
        "touser": to_user,
        "template_id": "IZDtYSHE1oT2nMPSMzvDnwn2F_qsfls58gdLHRB05Zk",
        "url": "http://weixin.qq.com/download",
        "topcolor": "#FF0000",
        "data": {
            "pose": {
                "value": posestr,
                "color": "#173177"
            }
        }
    }

    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
    }
    response = post(url, headers=headers, json=data).json()
    if response["errcode"] == 40037:
        print("推送消息失败，请检查模板id是否正确")
    elif response["errcode"] == 40036:
        print("推送消息失败，请检查模板id是否为空")
    elif response["errcode"] == 40003:
        print("推送消息失败，请检查微信号是否正确")
    elif response["errcode"] == 0:
        print("推送消息成功")
    else:
        print(response)


def get_access_token():
    # appId
    app_id = "wxf36fe652cc123dd3"
    # appSecret
    app_secret = "0b031cff0e9f3b32c846ca4458d539ae"
    post_url = ("https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid={}&secret={}"
                .format(app_id, app_secret))
    try:
        access_token = get(post_url).json()['access_token']
    except KeyError:
        print("获取access_token失败，请检查app_id和app_secret是否正确")
        sys.exit(1)
    # print(access_token)
    return access_token


# 计算夹角的函数
def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180
    # cross = v2[0] * v1[1] - v2[1] * v1[0]
    # if cross < 0:
    #     angle = - angle
    return angle


# 举双手				左手矢量小于0右手矢量夹角大于0
# 举左手				左手矢量小于0右手矢量小于0
# 举右手				左手矢量大于0右手矢量大于0
# 比三角形				举双手的同时，大臂与小臂的夹角小于120度
# 正常				左手矢量大于0右手矢量夹角小于0
# 叉腰				正常情况下，左手肘夹角小于120度，右手肘夹角也小于0


def get_pos(keyPoints, keyPoints_world):
    Str_pose = ""

    #
    # # 计算左臂与水平方向的夹角
    # keypoints = np.array(keypoints)
    # v1 = keypoints[12] - keypoints[11]
    # v2 = keypoints[13] - keypoints[11]
    # angle_left_arm = get_angle(v1, v2)
    # # 计算右臂与水平方向的夹角
    # v1 = keypoints[11] - keypoints[12]
    # v2 = keypoints[14] - keypoints[12]
    # angle_right_arm = get_angle(v1, v2)
    # # 计算左肘的夹角
    # v1 = keypoints[11] - keypoints[13]
    # v2 = keypoints[15] - keypoints[13]
    # angle_left_elow = get_angle(v1, v2)
    # # 计算右肘的夹角
    # v1 = keypoints[12] - keypoints[14]
    # v2 = keypoints[16] - keypoints[14]
    # angle_right_elow = get_angle(v1, v2)
    # # 计算左手与肩膀水平方向的夹角
    # v1 = keypoints[12] - keypoints[11]
    # v2 = keypoints[15] - keypoints[11]
    # angle_left_hand = get_angle(v1, v2)
    # # 计算右手与肩膀水平方向的夹角
    # v1 = keypoints[11] - keypoints[12]
    # v2 = keypoints[16] - keypoints[12]
    # angle_right_hand = get_angle(v1, v2)
    #
    # if angle_left_hand < 0 and angle_right_hand < 0:
    #     Str_pose = "Left_up"
    # elif angle_left_hand > 0 and angle_right_hand > 0:
    #     Str_pose = "Right_up"
    # elif angle_left_hand < 0 and angle_right_hand > 0:
    #     Str_pose = "All_hands_up"
    #     # if abs(angle_left_elow) < 120 and abs(angle_right_elow) < 120:
    #     #     Str_pose = "TRIANGLE"
    # elif angle_left_arm > 0 and angle_right_arm < 0 and angle_left_hand > 0 and angle_right_hand < 0:
    #     Str_pose = "Normal"
    #     # if abs(angle_left_elow) < 120 and abs(angle_right_elow) < 120:
    #     #     Str_pose = "AKIMBO"
    mouth = (np.add(keyPoints_world[9], keyPoints_world[10]) / 2).astype('int32')

    if keyPoints[15][1] > keyPoints[11][1] and keyPoints[16][1] < keyPoints[12][1]:
        Str_pose = "Right_up"
    if keyPoints[16][1] > keyPoints[12][1] and keyPoints[15][1] < keyPoints[11][1]:
        Str_pose = "Left_up"
    if keyPoints[16][1] > keyPoints[12][1] and keyPoints[15][1] > keyPoints[11][1]:
        Str_pose = "Normal"
    if keyPoints[16][1] < keyPoints[12][1] and keyPoints[15][1] < keyPoints[11][1]:
        Str_pose = "All_hands_up"
    if min(np.linalg.norm(np.subtract(mouth, keyPoints_world[16])),
           np.linalg.norm(np.subtract(mouth, keyPoints_world[15]))) < np.linalg.norm(
        np.subtract(keyPoints_world[11], keyPoints_world[12])):
        Str_pose = "Drinking"

    # print('RH:', keypoints[15][2], ' RS:', keypoints[11][2], ' LH:', keypoints[16][2], ' LS:', keypoints[12][2])
    return Str_pose


# 处理单帧的函数
def process_frame(img):
    global str_pose
    str_pose = ""
    global occupied
    global width
    global height
    global ax
    global keypoints
    global results
    global mp_pose
    keypoints = ['' for i in range(33)]
    keypoints_world = ['' for i in range(33)]
    deepth = ['' for i in range(33)]
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    mean = [0, 0]
    # meanadd = [0, 0, 0]
    # print(results.pose_landmarks)
    x_max = 0
    y_max = 0
    x_min = width
    y_min = height
    aspect_ratio = 0
    Relative_H_ofHead = 0
    Relative_W_ofHead = 0
    leg_angle = 0

    if results.pose_landmarks:
        occupied = 1
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        for i in range(33):
            cx = int(results.pose_landmarks.landmark[i].x * width)
            cy = int(results.pose_landmarks.landmark[i].y * height)
            cz = int(results.pose_landmarks.landmark[i].z * width)
            wx = int(results.pose_world_landmarks.landmark[i].x * 100)
            wy = int(results.pose_world_landmarks.landmark[i].y * 100)
            wz = int(results.pose_world_landmarks.landmark[i].z * 100)
            if cx > x_max:
                x_max = cx
            if cx < x_min:
                x_min = cx
            if cy > y_max:
                y_max = cy
            if cy < y_min:
                y_min = cy
            keypoints[i] = (cx, cy)
            deepth[i] = cz
            keypoints_world[i] = (wx, wy, wz)
            if keypoints[32]:
                str_pose = get_pos(keypoints, keypoints_world)
            # meanadd[0] = meanadd[0] + cx
            # meanadd[1] = meanadd[1] + cy
            # meanadd[2] = meanadd[2] + cz
        # 这里原本是计算整个身体的质心
        # mean[0] = meanadd[0] / 33
        # mean[1] = meanadd[1] / 33
        # mean[2] = meanadd[2] / 33

        # SVM输入参数之一：高宽比
        aspect_ratio = (y_max - y_min) / (x_max - x_min)

        # SVM输入参数之二：计算头与臀部相对高度
        Relative_H_ofHead = ((keypoints[23][1] + keypoints[24][1]) / 2 - keypoints[0][1])

        # SVM输入参数之三：计算头与臀部相对宽度
        Relative_W_ofHead = ((keypoints[23][0] + keypoints[24][0]) / 2 - keypoints[0][0])

        # SVM输入参数之四：大腿和身体的角度
        leg_angle = (get_angle(np.subtract(keypoints_world[25], keypoints_world[23]),
                               np.subtract(keypoints_world[11], keypoints_world[23])) +
                     get_angle(np.subtract(keypoints_world[26], keypoints_world[24]),
                               np.subtract(keypoints_world[12], keypoints_world[24]))) / 2

        # 这里改用计算两肩中心的位置
        mean = (np.add(keypoints[11], keypoints[12]) / 2).astype('int32')
        # print(mean[0], mean[1], mean[2])

    else:
        occupied = 0

    return img, mean, x_min, y_min, x_max, y_max, aspect_ratio, Relative_H_ofHead, Relative_W_ofHead, leg_angle


# 读取视频帧的函数
def process_cap(cap, gap):
    global occupied
    global ax
    global videofps
    global accessToken
    global user
    global width
    global height
    global results
    global mp_pose
    global keypoints
    # 获取accessToken
    accessToken = get_access_token()
    # 接收的微信用户
    user = "oqs5k6MGcmJ2rCxfhsqzzUk8JfXc"
    # 是否检测到人体指示器
    occupied = 0
    H = 36
    old_timestamp = StartTime = time.time()
    # Timeout是每帧的最小延时时间
    # FPS = 1/TIMEOUT
    # 若要求FPS <= 30 则 TIMEOUT = .033
    # 若对FPS上限没要求则可将其设为0
    TIMEOUT = .0333333

    # FPS初始化，这里填入的videofps指的是是一秒刷新两次平均FPS
    meanfps = np.zeros(int(videofps / 2), dtype=int, order='C')

    # 速度和位置初始化
    mean = [0, 0]
    vel = 0
    destinationvel = np.zeros(3, dtype=float, order='C')
    destinationdist = np.zeros(3, dtype=float, order='C')
    destinationfrontdist = np.zeros(3, dtype=float, order='C')
    destinationvelmean = np.zeros((10, 3), dtype=float, order='C')

    # 初始化
    i = 0
    FallRemainedTime = 0
    wxRemainedTime = 0
    wxHandsUpRemainedTime = 0
    minW = 0.01 * width
    minH = 0.01 * height
    faceregcount = 0
    faceregcountini = 4
    bodyregcount = 0
    bodyregcountini = 2
    acc = 0
    y_min, y_max, x_min, x_max = 0, 0, 0, 0
    faces = 0
    gray = 0
    velmean = []
    # 间隔多少秒推送一次老人姿态
    wxAutoPosePushCountini = 20
    wxAutoPosePushCount = 0
    # SVM相关参数初始化
    clf = joblib.load(clf_path)
    SVMclf = 0

    # 画图
    fig, ax = plt.subplots()
    ax.axis([0, 100, -50 * faceregcountini, 50 * faceregcountini])
    fig.canvas.manager.show()
    ploty = np.zeros(100, dtype=float, order='C')
    lines = ax.plot(ploty)

    while cap.isOpened():
        # 读视频帧
        if (time.time() - old_timestamp) > TIMEOUT:
            old_timestamp = time.time()
            ret, frame = cap.read()
            if ret:  # 判断是否读取成功
                # 如果读取成功则处理该帧

                # 人脸识别设定几帧处理一次
                if faceregcount == 0:
                    # 先读取灰度图供后面人脸识别使用，检测人脸
                    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=10,
                        minSize=(int(minW), int(minH))
                    )
                    faceregcount = faceregcountini - 1
                else:
                    faceregcount = faceregcount - 1
                # 后面还有对人脸的处理

                # 身体设定几帧计算一次
                # 这里是通过两帧之间的关系计算mean的加速度
                if bodyregcount == 0:
                    frontvel = vel
                    frontmean = mean
                    frame, mean, x_min, y_min, x_max, y_max, aspect_ratio, Relative_H_ofHead, Relative_W_ofHead, leg_angle = process_frame(
                        frame)
                    SVMclf = clf.predict([[aspect_ratio, Relative_H_ofHead, Relative_W_ofHead, leg_angle]])
                    if occupied == 0:
                        acc = frontvel = vel = frontmean = mean = 0
                    else:
                        vel = np.linalg.norm(np.subtract(mean, frontmean))
                        acc = vel - frontvel
                    bodyregcount = bodyregcountini - 1
                else:
                    bodyregcount = bodyregcount - 1
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # # 摄像头模式左右翻转
                # if menu == 1:
                #     frame = cv.flip(frame, 1)

                # 加速度判断
                if (50 * bodyregcountini) > abs(acc) >= (8 * bodyregcountini):
                    print(i, 'High ACC!')
                    i = i + 1
                    FallRemainedTime = videofps
                    # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

                # 只有在加速度过大后的一段时间内有摔倒姿态才算摔倒，摔倒姿态由SVM得出
                if FallRemainedTime > 0 and int(SVMclf) == 1:
                    # 显示指示框
                    cv.rectangle(frame, (x_min, y_min - H), (x_min + 320, y_min), (0, 0, 255), -1, 8)
                    cv.putText(frame, 'Fall_down' + "{:.1f}".format(FallRemainedTime / videofps), (x_min, y_min),
                               cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
                    cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    print(i, 'FALL DOWN!')
                    if wxRemainedTime == 0:
                        # 公众号推送消息
                        send_message(user, accessToken, 3)
                        # 至少多少秒推送一次
                        wxRemainedTime = int(videofps * 30)

                # 这里是减少摔倒指示框的保持时间
                if FallRemainedTime > 0:
                    FallRemainedTime = FallRemainedTime - 1
                # 微信提醒延迟时间
                if wxRemainedTime > 0:
                    wxRemainedTime = wxRemainedTime - 1
                # 手势微信提醒延迟时间
                if wxHandsUpRemainedTime > 0:
                    wxHandsUpRemainedTime = wxHandsUpRemainedTime - 1

                # 刷新图表图像
                ploty[0] = acc
                lines[0].set_ydata(ploty)
                ploty = np.roll(ploty, 1)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

                # 显示是否检测到人体
                if occupied:
                    cv.rectangle(frame, (10, 30 - H), (10 + 290, 30), (0, 0, 255), -1, 8)
                    cv.putText(frame, 'Detected', (10, 30), cv.FONT_HERSHEY_PLAIN, 3,
                               (255, 255, 255), 2)
                    cv.rectangle(frame, (10 + 300, 30 - H), (10 + 290 + 330, 30), (0, 0, 255), -1, 8)

                    # 静态SVM判断是否卧姿（0为站立，1为卧，2为坐）
                    cv.rectangle(frame, (x_min, y_min), (x_min + 30, y_min + H), (0, 0, 255), -1, 8)
                    cv.putText(frame, str(int(SVMclf)), (x_min, y_min + H),
                               cv.FONT_HERSHEY_PLAIN, 3,
                               (255, 255, 255), 2)

                    if wxAutoPosePushCount == 0:
                        send_message(user, accessToken, int(SVMclf))
                        wxAutoPosePushCount = int(wxAutoPosePushCountini * videofps)
                    else:
                        wxAutoPosePushCount = wxAutoPosePushCount - 1

                    # 显示姿态
                    cv.putText(frame, str_pose, (10 + 300, 30), cv.FONT_HERSHEY_PLAIN, 3,
                               (255, 255, 255), 2)

                    if str_pose == "All_hands_up" and wxHandsUpRemainedTime == 0:
                        send_message(user, accessToken, 4)
                        # 至少多少秒推送一次
                        wxHandsUpRemainedTime = int(videofps * 30)

                    # 检测到人体后再检测人脸
                    if len(faces) != 0:
                        for (x, y, w, h) in faces:
                            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                            if confidence < 80:
                                idum = names[idnum]
                                confidence = "{0}%".format(round(100 - confidence))
                            else:
                                idum = "unknown"
                                confidence = "{0}%".format(round(100 - confidence))
                            cv.putText(frame, str(idum), (x + 5, y - 5), font, 1, (0, 0, 255), 2)
                            cv.putText(frame, str(confidence), (x + 5, y + h - 5), font, 1, (0, 0, 255), 2)

                else:
                    cv.rectangle(frame, (10, 30 - H), (10 + 290, 30), (0, 255, 0), -1, 8)
                    cv.putText(frame, 'Undetected', (10, 30), cv.FONT_HERSHEY_PLAIN, 3,
                               (255, 255, 255), 2)

                # 绘制吃药喝水地点

                cv.circle(frame, drinkingplace, 10, (0, 255, 255), -1)
                # 检测是否吃药喝水
                if np.linalg.norm(
                        np.subtract(mean, drinkingplace)) < 30 and str_pose == 'Drinking' and occupied:
                    cv.circle(frame, drinkingplace, 10, (0, 255, 0), -1)

                for count in range(3):
                    # 绘制目的地连线
                    cv.circle(frame, destination[count], 10, (255, 0, 0), -1)
                if occupied:
                    for count in range(3):
                        cv.line(frame, destination[count], tuple(mean), (255, 0, 0), 3)
                        # 测算到目的地的距离
                        destinationfrontdist[count] = destinationdist[count]
                        destinationdist[count] = np.linalg.norm(np.subtract(mean, list(destination[count])))
                    # 计算到目的地的速度
                    destinationvel = np.subtract(destinationfrontdist, destinationdist)
                    destinationvelmean = np.roll(destinationvelmean, 1, axis=0)
                    destinationvelmean[0] = destinationvel
                    # print(destinationvelmean)
                    frontvelmean = velmean
                    velmean = np.mean(destinationvelmean, axis=0)
                    # 判定运动方向
                    if 50 > np.linalg.norm(np.subtract(velmean, [0, 0, 0])) > 5:
                        cv.line(frame, destination[np.argmax(velmean)], tuple(mean),
                                (255, 255, 0), 3)
                        # print(np.argmax(velmean), np.argmax(frontvelmean))
                        if np.argmax(velmean) == 1 and np.argmax(frontvelmean) != 1:
                            commands = {'commands': [{'code': 'switch_led', 'value': True}]}
                            openapi.post('/v1.0/iot-03/devices/{}/commands'.format(DEVICE_ID), commands)
                        elif np.argmax(velmean) != 1 and np.argmax(frontvelmean) == 1:
                            commands = {'commands': [{'code': 'switch_led', 'value': False}]}
                            openapi.post('/v1.0/iot-03/devices/{}/commands'.format(DEVICE_ID), commands)
                else:
                    # 无人则清零
                    destinationfrontdist.fill(0)
                    destinationdist.fill(0)
                    destinationvel.fill(0)
                    destinationvelmean.fill(0)

                # 显示平均FPS
                cv.rectangle(frame, (10, 70 - H), (10 + 60, 70), (0, 0, 255), -1, 8)
                cv.putText(frame, str(int(np.mean(meanfps))), (10, 70), cv.FONT_HERSHEY_PLAIN, 3,
                           (255, 255, 255), 2)

                # 计算平均FPS
                meanfps = np.roll(meanfps, 1)
                meanfps[0] = 1 / (time.time() - StartTime)
                StartTime = time.time()

                # 展示处理后的三通道图像
                cv.imshow('video', frame)
                # 按键盘的esc或者q退出
                if cv.waitKey(gap) in [ord('q'), 27]:
                    break
                # elif cv.waitKey(1) == ord(']'):
                #     zoom = cap.get(cv.CAP_PROP_ZOOM)
                #     cap.set(27, zoom + 10)
                #     print(zoom)
                # elif cv.waitKey(1) == ord('['):
                #     zoom = cap.get(cv.CAP_PROP_ZOOM)
                #     cap.set(27, zoom - 10)
                #     print(zoom)
                # elif cv.waitKey(1) == ord(" "):
                #     cv.waitKey(0)
            else:
                print('error!')
                break

    # 关闭摄像头
    cap.release()
    # 关闭图像窗口
    cv.destroyAllWindows()


# 从摄像头实时检测
def detect_camera():
    # 创建窗口
    global width
    global height
    global videofps
    cv.namedWindow('video', cv.WINDOW_NORMAL)
    # 调用摄像头获取画面  0是windows系统下默认的摄像头，1是Mac系统
    print('Opening the camera')
    cap = cv.VideoCapture(1)
    # cap.set(cv.CAP_PROP_FPS, 60)
    print('Successfully opened')
    videofps = cap.get(cv.CAP_PROP_FPS)
    # 获取窗口大小
    cap.set(3, 640)
    cap.set(4, 480)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print('FPS=%d' % videofps)
    cv.resizeWindow('video', width, height)
    process_cap(cap, 1)


# 从本地导入视频检测
def detect_video(videopath):
    # 创建窗口
    global width
    global height
    global videofps
    cv.namedWindow('video', cv.WINDOW_NORMAL)
    # 从本地读取视频
    cap = cv.VideoCapture(videopath)
    # MediaPlayer(path)
    # 获取原视频帧率
    videofps = cap.get(cv.CAP_PROP_FPS)
    # 获取原视频窗口大小
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    print('FPS=%d' % videofps)
    # 视频两帧之间的播放间隔，单位为毫秒
    # gap = int(1000 / videofps)
    gap = 10
    cv.resizeWindow('video', width, height)
    process_cap(cap, gap)


if __name__ == '__main__':
    while True:
        if menu == 1:
            detect_camera()
            break
        elif menu == 2:
            path = video_path + 'falldown.mp4'
            # path = video_path + 'standing.mp4'
            # path = video_path + 'sitting.mp4'
            # input('请输入视频路径（例如：D:\\download\\abc.mp4）：\n')
            detect_video(path)
            break
        elif menu == 3:
            break
        else:
            print("输入错误，请重新输入！")
            continue
