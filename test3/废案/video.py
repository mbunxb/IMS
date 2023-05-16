import cv2 as cv
import mediapipe as mp
import time
import numpy as np

# from ffpyplayer.player import MediaPlayer

# 导入solution
mp_pose = mp.solutions.pose
# 导入绘图函数
mp_drawing = mp.solutions.drawing_utils

# 导入模型
pose = mp_pose.Pose(static_image_mode=False,  # 是静态图片还是连续视频帧
                    model_complexity=1,  # 选择人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于两者之间
                    smooth_landmarks=True,  # 是否平滑关键点
                    enable_segmentation=True,  # 是否人体抠图
                    min_detection_confidence=0.8,  # 置信度阈值
                    min_tracking_confidence=0.3  # 追踪阈值
                    )
global occupied
global pTime
global cTime
global str_pose
str_pose = ""
global width
global height
global menu


# 计算夹角的函数
def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180
    cross = v2[0] * v1[1] - v2[1] * v1[0]
    if cross < 0:
        angle = - angle
    return angle


# 举双手				左手矢量小于0右手矢量夹角大于0
# 举左手				左手矢量小于0右手矢量小于0
# 举右手				左手矢量大于0右手矢量大于0
# 比三角形				举双手的同时，大臂与小臂的夹角小于120度
# 正常				左手矢量大于0右手矢量夹角小于0
# 叉腰				正常情况下，左手肘夹角小于120度，右手肘夹角也小于0


def get_pos(keypoints):
    str_pose = ""
    # 计算左臂与水平方向的夹角
    keypoints = np.array(keypoints)
    v1 = keypoints[12] - keypoints[11]
    v2 = keypoints[13] - keypoints[11]
    angle_left_arm = get_angle(v1, v2)
    # 计算右臂与水平方向的夹角
    v1 = keypoints[11] - keypoints[12]
    v2 = keypoints[14] - keypoints[12]
    angle_right_arm = get_angle(v1, v2)
    # 计算左肘的夹角
    v1 = keypoints[11] - keypoints[13]
    v2 = keypoints[15] - keypoints[13]
    angle_left_elow = get_angle(v1, v2)
    # 计算右肘的夹角
    v1 = keypoints[12] - keypoints[14]
    v2 = keypoints[16] - keypoints[14]
    angle_right_elow = get_angle(v1, v2)
    # 计算左手与肩膀水平方向的夹角
    v1 = keypoints[12] - keypoints[11]
    v2 = keypoints[15] - keypoints[11]
    angle_left_hand = get_angle(v1, v2)
    # 计算右手与肩膀水平方向的夹角
    v1 = keypoints[11] - keypoints[12]
    v2 = keypoints[16] - keypoints[12]
    angle_right_hand = get_angle(v1, v2)

    if angle_left_hand < 0 and angle_right_hand < 0:
        str_pose = "Left_up"
    elif angle_left_hand > 0 and angle_right_hand > 0:
        str_pose = "Right_up"
    elif angle_left_hand < 0 and angle_right_hand > 0:
        str_pose = "All_hands_up"
        # if abs(angle_left_elow) < 120 and abs(angle_right_elow) < 120:
        #     str_pose = "TRIANGLE"
    elif angle_left_arm > 0 and angle_right_arm < 0 and angle_left_hand > 0 and angle_right_hand < 0:
        str_pose = "Normal"
        # if abs(angle_left_elow) < 120 and abs(angle_right_elow) < 120:
        #     str_pose = "AKIMBO"
    return str_pose


# 处理单帧的函数
def process_frame(img):
    global str_pose
    global occupied
    global width
    global height
    keypoints = ['' for i in range(33)]
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        occupied = 1
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for i in range(33):
            cx = int(results.pose_landmarks.landmark[i].x * width)
            cy = int(results.pose_landmarks.landmark[i].y * height)
            keypoints[i] = (cx, cy)
            if keypoints[32]:
                str_pose = get_pos(keypoints)
    else:
        occupied = 0
    return img


# 读取视频帧的函数
def process_cap(cap, gap):
    global pTime
    global cTime
    global occupied
    global menu
    pTime = 0
    cTime = 0
    occupied = 0
    H = 36
    StartTime = time.time()
    while cap.isOpened():
        # 读视频帧
        ret, frame = cap.read()
        if ret:  # 判断是否读取成功
            # 如果读取成功则处理该帧
            # 摄像头模式左右翻转
            if menu == 1:
                frame = cv.flip(frame, 1)
            frame = process_frame(frame)
            fps = 1 / (time.time() - StartTime)
            StartTime = time.time()
            cv.rectangle(frame, (10, 70 - H), (10 + 60, 70), (0, 0, 255), -1, 8);
            cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3,
                       (255, 255, 255), 2)
            if occupied:
                cv.rectangle(frame, (10, 30 - H), (10 + 290, 30), (0, 0, 255), -1, 8);
                cv.putText(frame, 'Detected', (10, 30), cv.FONT_HERSHEY_PLAIN, 3,
                           (255, 255, 255), 2)
                cv.rectangle(frame, (10 + 300, 30 - H), (10 + 290 + 330, 30), (0, 0, 255), -1, 8);
                cv.putText(frame, str_pose, (10 + 300, 30), cv.FONT_HERSHEY_PLAIN, 3,
                           (255, 255, 255), 2)
            else:
                cv.rectangle(frame, (10, 30 - H), (10 + 290, 30), (0, 255, 0), -1, 8);
                cv.putText(frame, 'Undetected', (10, 30), cv.FONT_HERSHEY_PLAIN, 3,
                           (255, 255, 255), 2)
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
    cv.namedWindow('video', cv.WINDOW_NORMAL)
    # 调用摄像头获取画面  0是windows系统下默认的摄像头，1是Mac系统
    print('Opening the camera')
    cap = cv.VideoCapture(1)
    # cap.set(cv.CAP_PROP_FPS, 60)
    print('Successfully opened')
    fps = cap.get(cv.CAP_PROP_FPS)
    # 获取窗口大小
    cap.set(3, 640)
    cap.set(4, 480)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print('FPS=%d' % fps)
    cv.resizeWindow('video', width, height)
    process_cap(cap, 1)


# 从本地导入视频检测
def detect_video(path):
    # 创建窗口
    global width
    global height
    cv.namedWindow('video', cv.WINDOW_NORMAL)
    # 从本地读取视频
    cap = cv.VideoCapture(path)
    # MediaPlayer(path)
    # 获取原视频帧率
    fps = cap.get(cv.CAP_PROP_FPS)
    # 获取原视频窗口大小
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    print('FPS=%d' % fps)
    # 视频两帧之间的播放间隔，单位为毫秒
    # gap = int(1000 / fps)
    gap = 10
    cv.resizeWindow('video', width, height)
    process_cap(cap, gap)


if __name__ == '__main__':
    global menu
    while True:
        # menu = int(input('请选择检测模式：1. 打开摄像头检测\t2. 从本地导入视频检测\t3. 退出\n'))
        menu = 1  # 预设模式
        if menu == 1:
            detect_camera()
            break
        elif menu == 2:
            # path = 'C:\\Users\\85781\\Desktop\\文件夹\\毕设\\amagi.mp4'
            path = 'C:\\Users\\85781\\Desktop\\文件夹\\毕设\\dataset\\chute02\\cam1.avi'
            # input('请输入视频路径（例如：D:\\download\\abc.mp4）：\n')
            detect_video(path)
            break
        elif menu == 3:
            break
        else:
            print("输入错误，请重新输入！")
            continue
