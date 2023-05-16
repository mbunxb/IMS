import os
import cv2
import numpy as np
import mediapipe as mp
import pylab as pl  # 绘图功能
import matplotlib.pyplot as plt
import joblib
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D


# 计算夹角的函数
def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180
    # cross = v2[0] * v1[1] - v2[1] * v1[0]
    # if cross < 0:
    #     angle = - angle
    return angle


# 非线性SVM分类,当degree为0表示线性
def PolynomialSVC(degree, C=1.0):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),  # 生成多项式
        ("std_scaler", StandardScaler()),  # 标准化
        ("linearSVC", LinearSVC(C=C, multi_class='ovr'))  # 最后生成svm
    ])


# 核函数
def PolynomialKernelSVC(degree, C=1.0):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("kernelSVC", SVC(kernel="poly"))  # poly代表多项式特征
    ])


# 高斯核函数
def RBFKernelSVC(gamma=1.0):
    return Pipeline([
        ('std_scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', gamma=gamma, decision_function_shape='ovo'))
    ])


path = "D:\\opencv\\perseonaldataset\\mixed"

# 导入solution
mp_pose = mp.solutions.pose
# 导入绘图函数
mp_drawing = mp.solutions.drawing_utils

# 导入模型
pose = mp_pose.Pose(static_image_mode=True,  # 是静态图片还是连续视频帧
                    model_complexity=2,  # 选择人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于两者之间
                    smooth_landmarks=True,  # 是否平滑关键点
                    enable_segmentation=True,  # 是否人体抠图
                    min_detection_confidence=0.6,  # 置信度阈值
                    min_tracking_confidence=0.9  # 追踪阈值
                    )

image_paths = [os.path.join(path, f) for f in os.listdir(path)]
face_samples = []
ids = []
aspect_ratio = []
Relative_H_ofHead = []
Relative_W_ofHead = []
leg_angle = []
width = 0
height = 0
for image_path in image_paths:
    if os.path.split(image_path)[-1].split(".")[-1] != 'jpg':
        continue
    img = cv2.imread(image_path)
    width = img.shape[1]
    height = img.shape[0]
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_RGB)
    keypoints = ['' for i in range(33)]
    keypoints_world = ['' for i in range(33)]
    deepth = ['' for i in range(33)]
    x_max = 0
    y_max = 0
    x_min = width
    y_min = height
    if results.pose_landmarks:
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
        print(keypoints)
        aspect_ratio.append((y_max - y_min) / (x_max - x_min))
        id = int(os.path.split(image_path)[-1].split(".")[0])
        ids.append(id)
        Relative_H_ofHead.append(((keypoints[23][1] + keypoints[24][1]) / 2 - keypoints[0][1]))
        Relative_W_ofHead.append(((keypoints[23][0] + keypoints[24][0]) / 2 - keypoints[0][0]))
        leg_angle.append(
            (get_angle(np.subtract(keypoints_world[25], keypoints_world[23]),
                       np.subtract(keypoints_world[11], keypoints_world[23])) +
             get_angle(np.subtract(keypoints_world[26], keypoints_world[24]),
                       np.subtract(keypoints_world[12], keypoints_world[24]))) / 2
        )
    else:
        print("NO PERSON")
        continue
print(ids)
print(aspect_ratio)
print(Relative_H_ofHead)
print(Relative_W_ofHead)
print(leg_angle)
y = ids
X = np.c_[aspect_ratio, Relative_H_ofHead, Relative_W_ofHead, leg_angle]
# X = np.vstack((X, [[3, 32], [3.4, 30], [3.2, 28], [2.7, 40]]))
# y.extend([2, 2, 2, 2])
print(X)
print(y)
svc = PolynomialSVC(degree=1, C=1)
svc.fit(X, y)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Set1)
# plt.show()
joblib.dump(svc, 'D:\\opencv\\trainner\\clfLinearSVCov4.model')
