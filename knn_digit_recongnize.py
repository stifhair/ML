#-*- coding: utf-8 -*-
import cv2
import numpy as np

img  = cv2.imread('digits.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 세로로 50줄, 가로로 100줄로 사진을 나눈다.
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
x     = np.array(cells)

# 각 (20 X 20) 크기의 사진을 한 줄(1 X 400)으로 변환한다.
train =  x[:, :].reshape(-1, 400).astype(np.float32)

# 0이 500개, 10| 500개, ...로 총 5,000개가 들어가는 ( 1 X 5000 )배열을 만듭니다.
k = np.arange(10)
train_labels = np.repeat(k,500)[:, np.newaxis]
np.savez("trained.npz", train=train, train_labels=train_labels)