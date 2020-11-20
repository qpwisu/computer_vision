#computer vision 실습
#opencv lib을 이용해 영상을 입출력하고 인공지능에서 많이 사용되는 numpy와 matplotlib를 이용
#colab 코드 가져옴 
import cv2
import matplotlib.pyplot as plt
import numpy as np

# MNIST(Modified National Institute of Stardard and Technology)의
# 필기체 숫자 데이터와 CIFAR를 주로 활용하여 실습 진행

#keras로부터 mnist 숫자 데이터를 읽어 오기
from keras.datasets import mnist
(x_train, y_train), (x_test,y_test) = mnist.load_data()

print('Number of original training examples:', len(x_train))
print('Number of original test examples:', len(x_test))

print('Number of original training examples:', len(x_train))
print('Number of original test examples:', len(x_test))

#학습 이미지 하나 가져오기
test_image = x_train[0]
print(test_image)
test_image.shape
#mlt로 이미지 출력력
print(y_train[0])
plt.imshow(test_image)
#gray로 출력
plt.imshow(test_image, cmap='gray')
plt.show()

#opencvlib로 이미지 출력 200,200 사이즈
cv2.imshow(test_image)
cv2.imshow(cv2.resize(test_image,(200,200)))

cv2.imwrite('test_image.jpg', test_image)
cv2.image = cv2.imread('test_image.jpg')
plt_image = plt.imread('test_image.jpg')

#CIFAR10 사물인식 data 활용 Keras.dataset에 저장되어 있는 데이터 읽어 오기
from keras.datasets import cifar10
(cx_train, cy_ftrain), (cx_test, cy_test) = cifar10.load_data()
print('Number of original cifar training examples:', len(cx_train))
print('Number of original cifar test examples:', len(cx_test))
print(cx_train[40])
cx_train[40].shape
height, width, channel = cx_train[40].shape
print(height, width, channel)
plt.imshow(cx_train[40])

R_channel = cx_train[40][:, :, 0]
G_channel = cx_train[40][:, :, 1]
B_channel = cx_train[40][:, :, 2]

plt.imshow(G_channel, cmap='gray')
plt.show()
from google.colab.patches import cv2_imshow
cv2_imshow(cx_train[40])
cv2.imwrite('train_40.jpg', cx_train[40])
reload_image = cv2.imread('train_40.jpg')
plt_colorimage = plt.imread('train_40.jpg')