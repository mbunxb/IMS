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


# 非线性SVM分类,当degree为0表示线性
def PolynomialSVC(degree, C=1.0):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),  # 生成多项式
        ("std_scaler", StandardScaler()),  # 标准化
        ("linearSVC", LinearSVC(C=C))  # 最后生成svm
    ])


# 绘制决策边界
def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1)
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)


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
        ('svc', SVC(kernel='rbf', gamma=gamma))
    ])


path = "D:\\opencv\\perseonaldataset\\"

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
width = 852
height = 480
for image_path in image_paths:
    if os.path.split(image_path)[-1].split(".")[-1] != 'jpg':
        continue
    img = cv2.imread(image_path)
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
    else:
        print("NO PERSON")
        continue
print(ids)
print(aspect_ratio)
print(Relative_H_ofHead)
y = ids
X = np.c_[aspect_ratio, Relative_H_ofHead]
print(X)

svc = RBFKernelSVC(1)
svc.fit(X, y)
plot_decision_boundary(svc, axis=[0, 5, -100, 200])
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.show()
joblib.dump(svc, 'D:\\opencv\\trainner\\clfRBFKernelSVC.model')
