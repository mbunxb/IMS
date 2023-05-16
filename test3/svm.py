import numpy as np
import mediapipe as mp
import pylab as pl  # 绘图功能
from sklearn import svm
import joblib

clf = joblib.load('D:\\opencv\\trainner\\clf.model')
print(clf.predict([[2, 0]]))
