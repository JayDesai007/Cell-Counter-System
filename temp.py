import numpy as np #pip install numpy
import cv2  #pip install python-opencv
import matplotlib.pyplot as plt #pip install matplotlib

class ImageOperation:
    def __init__(self, filePath):
        self.filePath = filePath

    def loadImage(self):
        self.frame = cv2.imread(self.filePath)
        self.ansFrame = np.copy(self.frame)
        self.height = self.frame.shape[0]
        self.width = self.frame.shape[1]

    def cannyConvert(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        self.canny = cv2.Canny(blur, 7, 7)
        plt.imshow(self.canny)
        plt.show()

    def findHoughLines(self):
        self.lines = cv2.HoughLines(self.canny, 1, np.pi / 90, 250, np.array([]), 0, 0)

    def find_avg_hline(self, temp):
        t = 0
        sum = 0.0
        for i in temp:
            sum += i[1]
            t += 1
        avg = int(sum / t)
        return [temp[0][0], avg, temp[0][2], temp[0][3], avg, temp[0][5], avg]

    def find_avg_vline(self, temp):
        t = 0
        sum = 0.0
        for i in temp:
            sum += i[1]
            t += 1
        avg = int(sum / t)
        return [temp[0][0], avg, temp[0][2], avg, temp[0][4], avg, temp[0][6]]

    def sort_lines(self):
        temp1 = []
        temp2 = []
        if self.lines is not None:
            for i in self.lines:
                rho = i[0][0]
                if rho >= 0:
                    theta = i[0][1]
                    a = int(round(np.cos(theta)))
                    b = int(round(np.sin(theta)))
                    if a == 0:
                        temp1.append([True, rho, theta, 0, rho, self.width, rho])
                    else:
                        temp2.append([True, rho, theta, rho, 0, rho, self.height])

        temp = [[False, 0, 0, 0, 0, self.width, 0], [False, 0, 0, 0, self.height, self.width, self.height]]
        temp_ = [[False, 0, 0, 0, 0, 0, self.height], [False, 0, 0, self.width, 0, self.width, self.height]]
        for j in range(0, len(temp1)):
            t = []
            for i in range(j, len(temp1)):
                if temp1[i][0] and temp1[i][2] == temp1[j][2] and np.abs(temp1[j][1] - temp1[i][1]) < 300:
                    t.append(temp1[i])
                    temp1[i][0] = False
            if t:
                temp.append(self.find_avg_hline(t))
        for j in range(0, len(temp2)):
            t = []
            for i in range(j, len(temp2)):
                if temp2[i][0] and temp2[i][2] == temp2[j][2] and np.abs(temp2[j][1] - temp2[i][1]) < 50:
                    t.append(temp2[i])
                    temp2[i][0] = False
            if t:
                temp_.append(self.find_avg_vline(t))
        intersection_point = []
        for j in temp:
            for i in temp_:
                intersection_point.append(self.lineIntersection([j[3], j[4]], [j[5], j[6]], [i[3], i[4]], [i[5], i[6]]))
        intersection_point_ = []
        intersection_point = sorted(intersection_point)
        for i in range(len(temp_)):
            intersection_point_.append(intersection_point[i * len(temp):i * len(temp) + len(temp)])
        for i in temp_:
            temp.append(i)
        self.lines = temp
        self.intersection_point = intersection_point_

    def lineIntersection(self, A, B, C, D):
        a1 = B[1] - A[1]
        b1 = A[0] - B[0]
        c1 = a1 * A[0] + b1 * A[1]
        a2 = D[1] - C[1]
        b2 = C[0] - D[0]
        c2 = a2 * C[0] + b2 * C[1]
        determinant = a1 * b2 - a2 * b1
        if determinant == 0:
            return [np.Inf, np.Inf]
        else:
            x = (b2 * c1 - b1 * c2) / determinant
            y = (a1 * c2 - a2 * c1) / determinant
            return (int(x), int(y))

    def find_area(self, A, D):
        return (D[0]-A[0])*(D[1]-A[1])

    def saveImage(self):
        cv2.imwrite("test_images/temp.jpg", self.frame)

    def display_lines(self):
        for i in self.lines:
            rho = i[1]
            theta = i[2]
            a = int(round(np.cos(theta)))
            if a == 0:
                cv2.line(self.frame, (0, rho), (self.width, rho), (255, 0, 0), 2)
            else:
                cv2.line(self.frame, (rho, 0), (rho, self.height), (255, 0, 0), 2)
        plt.imshow(self.frame)
        plt.show()

    def display_lines_(self):
        for i in self.lines:
            rho = i[0][0]
            theta = i[0][1]
            a = int(round(np.cos(theta)))
            if a == 0:
                cv2.line(self.ansFrame, (0, rho), (self.width, rho), (255, 0, 0), 2)
            else:
                cv2.line(self.ansFrame, (rho, 0), (rho, self.height), (255, 0, 0), 2)
        plt.imshow(self.ansFrame)
        plt.show()

# image = ImageOperation("test_images/img.jpg")
image = ImageOperation("5min191001_010_ovl.jpg")
image.loadImage()
image.cannyConvert()
image.findHoughLines()
image.display_lines_()
print(image.lines.shape)
image.sort_lines()
image.display_lines()

#
# sporeCount = Spore("inference_graph/frozen_inference_graph.pb", image)
# sporeCount.loadModel()
# sporeCount.find_valid_area()







# # import subprocess
# # MyOut = subprocess.Popen('set PYTHONPATH=C:/tensorflow1/project/Project 2.0/models/research/slim && python models/research/object_detection/train.py --logtostderr --train_dir=models/research/object_detection/training/ --pipeline_config_path=models/research/object_detection/training/faster_rcnn_inception_v2_pets.config',
# #             stdout=subprocess.PIPE,
# #             stderr=subprocess.STDOUT, shell=True)
# # stdout,stderr = MyOut.communicate()
# # # flag = input()
# # print("Hii")
# # MyOut.terminate()
# # print(stdout)
# # print(stderr
# # test = subprocess.Popen('set PYTHONPATH=C:\\tensorflow1\\models\\research\\slim && python models/research/object_detection/train.py --logtostderr --train_dir=models/research/object_detection/training/ --pipeline_config_path=models/research/object_detection/training/faster_rcnn_inception_v2_pets.config').read()
# # print(test)
# # import os
# # os.popen('start')
# # import random
# #sampling with replacement
# # list = [20, 30, 40, 50 ,60, 70, 80]
# # sampling = random.sample(list, k=4)
# # print("sampling with choices() ", sampling)
# # import math
# # while True:
# #     n = int(input())
# #     print(math.ceil(n*0.3))
# import cv2
# import numpy as np
#
# image = cv2.imread('5min191001_010_ovl.jpg')
#
# # Grayscale and Canny Edges extracted
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 100, 170, apertureSize = 3)
#
# # Run HoughLines using a rho accuracy of 1 pixel
# # theta accuracy of np.pi / 180 which is 1 degree
# # Our line threshold is set to 190 (number of points on line)
# lines = cv2.HoughLines(edges, 1, np.pi / 180, 190)
#
# # We iterate through each line and convert it to the format
# # required by cv.lines (i.e. requiring end points)
# for rho, theta in lines[0]:
#     a = int(round(np.cos(theta)))
#     if a == 0:
#         cv2.line(image, (0, rho), (self.width, rho), (255, 0, 0), 2)
#         # temp1.append([True, rho, theta, 0, rho, width, rho])
#     else:
#         cv2.line(image, (rho, 0), (rho, self.height), (255, 0, 0), 2)
#         # temp2.append([True, rho, theta, rho, 0, rho, height])
#     # cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#
# cv2.imshow('Hough Lines', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()