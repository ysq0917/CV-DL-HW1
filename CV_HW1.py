from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import torch
import torchsummary
import torchvision
import torchvision.transforms as transforms
import os


image_L, image_R,allfiles, img_path = None, None, None, None
intrinsic_matrix = None
distortion_coefficients = None
spin_value = None
clicked_points, disparity_value = None, None

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Main Window')
        self.resize(1000, 600)
        
        self.load_image = LoadImage()
        self.calibration = Calibration()
        self.augmented_reality = AugmentedReality()
        self.stereo_disparity_map = StereoDisparityMap()
        self.sift = SIFT()
        self.vgg = VGG19()

        self.layout = QtWidgets.QVBoxLayout(self) #Q1-Q5

        self.layout1 = QtWidgets.QHBoxLayout(self) #Q1-Q3
        self.layout1.setAlignment(QtCore.Qt.AlignTop)
        self.layout1.addWidget(self.load_image)
        self.layout1.addWidget(self.calibration)
        self.layout1.addWidget(self.augmented_reality)
        self.layout1.addWidget(self.stereo_disparity_map)
        self.layout1.addStretch(1)

        self.layout2 = QtWidgets.QHBoxLayout(self) #Q4-Q5
        self.layout2.addWidget(self.sift)
        self.layout2.addWidget(self.vgg)
        self.layout2.addStretch(1)

        self.layout.addLayout(self.layout1)
        self.layout.addLayout(self.layout2)



class LoadImage(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.ui_LoadImage()
        self.setFixedWidth(250)
        
    def ui_LoadImage(self):
        self.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setLineWidth(3)
        self.setObjectName("Load_Image")
        
        self.label = QtWidgets.QLabel()
        self.label.setText('Load Image')

        self.vlayout = QtWidgets.QVBoxLayout(self)
        self.vlayout.setAlignment(QtCore.Qt.AlignTop)
        self.vlayout.addWidget(self.label)

        self.btn1 = QtWidgets.QPushButton(self)
        self.btn1.setText('Load folder')
        self.btn1.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn1) 
        self.vlayout.addSpacing(20)
        self.btn1.clicked.connect(open_folder)

        self.btn2 = QtWidgets.QPushButton(self)
        self.btn2.setText('Load_Image_L') 
        self.btn2.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn2)
        self.vlayout.addSpacing(20)
        self.btn2.clicked.connect(open_imageL)

        self.btn3 = QtWidgets.QPushButton(self)
        self.btn3.setText('Load_Image_R')
        self.btn3.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn3) 
        self.btn3.clicked.connect(open_imageR)

def open_folder():
    folderPath = QtWidgets.QFileDialog.getExistingDirectory()  # 選取資料夾
    global allfiles
    allfiles = folderPath
    print(folderPath)
    global imgs, PIL_imgs, filenames
    imgs = []
    PIL_imgs = []
    filenames = []
    for filename in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath, filename)) #Q1
        PIL_img = Image.open(os.path.join(folderPath, filename)) #Q5-1
        PIL_imgs.append(PIL_img)
        imgs.append(img)
        filenames.append(filename)


def open_imageL():
    filePath, filterType = QtWidgets.QFileDialog.getOpenFileName() # 選取一個檔案
    global image_L
    image_L = cv2.imread(filePath)
    
def open_imageR():
    filePath, filterType = QtWidgets.QFileDialog.getOpenFileName()
    global image_R
    image_R = cv2.imread(filePath)
    
class Calibration(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.ui_Calibration()
        self.setFixedWidth(250)
        
    def ui_Calibration(self):
        self.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setLineWidth(3)
        self.setObjectName("Calibration")

        self.label = QtWidgets.QLabel(self)
        self.label.setText('1. Calibration')

        self.vlayout = QtWidgets.QVBoxLayout(self)
        self.vlayout.setAlignment(QtCore.Qt.AlignTop)
        self.vlayout.addWidget(self.label)

        #two buttons
        self.btn1 = QtWidgets.QPushButton(self)
        self.btn1.setText('1.1 Find Corners')
        self.btn1.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn1) 
        self.vlayout.addSpacing(10)
        self.btn1.clicked.connect(CornerDetection)
        

        self.btn2 = QtWidgets.QPushButton(self)
        self.btn2.setText('1.2 Find Intrinsic')
        self.btn2.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn2)
        self.vlayout.addSpacing(10)
        self.btn2.clicked.connect(FindIntrinsicMatrix)
        

        #frame
        self.frame = QtWidgets.QFrame(self)
        self.frame.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setLineWidth(2)  
        self.frame.setObjectName("1.3 Find Extrinsic")
        self.vlayout.addWidget(self.frame)
        self.vlayout.addSpacing(5)

        self.label = QtWidgets.QLabel(self.frame)
        self.label.setText('1.3 Find Extrinsic')
        self.vlayout.addWidget(self.label)
        self.vlayout.addSpacing(5)

        #spinbox
        self.spinBox = QtWidgets.QSpinBox(self.frame)
        self.spinBox.setRange(1, 15)
        self.spinBox.setObjectName("spinBox")
        self.vlayout.addWidget(self.spinBox)
        self.spinBox.valueChanged.connect(self.change_spinBox)


        #three buttons
        self.btn3 = QtWidgets.QPushButton(self.frame)
        self.btn3.setText('1.3 Find Extrinsic')
        self.btn3.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn3)
        self.vlayout.addSpacing(10)
        self.btn3.clicked.connect(findExtrinsicMatrix)

        self.btn4 = QtWidgets.QPushButton(self.frame)
        self.btn4.setText('1.4 Find Distortion')
        self.btn4.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn4)
        self.vlayout.addSpacing(10)
        self.btn4.clicked.connect(FindDistortionMatrix)

        self.btn5 = QtWidgets.QPushButton(self.frame)
        self.btn5.setText('1.5 Show Result')
        self.btn5.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn5)
        self.btn5.clicked.connect(ShowResult)
    
    def change_spinBox(self):
        global spin_value
        spin_value = self.spinBox.value()
        print(spin_value)

#Q1.1 Corner Detection
# Find and draw the corners on the chessboard for each image
def CornerDetection():
    all_image_files = glob.glob(allfiles + '/*.bmp')
    corners_image = [] # save all images with corners
    for file in all_image_files:
        image = cv2.imread(file)
        img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        find, corners = cv2.findChessboardCorners(img, (11,8), None)
        if find:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(img, corners, (5,5), (-1,-1), criteria)
            img_corner = img.copy()
            cv2.drawChessboardCorners(img_corner, (11,8), corners, find)
            corners_image.append(img_corner)
        
    for i, img in enumerate(corners_image): #show all images
        cv2.imshow(f"Image {i}", img)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()


#Q1.2 Find Intrinsic Matrix
def FindIntrinsicMatrix():
    all_image_files = glob.glob(allfiles + '/*.bmp')
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    p_size= (11,8)
    objp = np.zeros((11*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    for file in all_image_files:
        image = cv2.imread(file)
        img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        find, corners = cv2.findChessboardCorners(img, p_size, None)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(img, corners, (5,5), (-1,-1), criteria)
        if find:
            objpoints.append(objp)
            imgpoints.append(corners)
    if len(objpoints) > 0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, p_size, None, None)
        print("Intrinsic:")
        print(mtx)
        global intrinsic_matrix
        intrinsic_matrix = mtx
        global distortion_coefficients
        distortion_coefficients = dist
    else:
        print("No calibration pattern found in the images.")


# Q1.3 Find Extrinsic Matrix
# Given : Intrinsic Matrix, Distortion Coefficients, and the list of 15 images
def findExtrinsicMatrix():
    global spin_value
    global intrinsic_matrix, distortion_coefficients
    # Load the list of 15 images
    all_image_files = glob.glob(allfiles+"/*.bmp") 
    objp = np.zeros((11*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    extrinsic_matrices = []
    for file in all_image_files:
        image = cv2.imread(file)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        psize = (11, 8)  
        find, corners = cv2.findChessboardCorners(img, psize, None)
        if find:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(img, corners, (5,5), (-1,-1), criteria)
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners, intrinsic_matrix, distortion_coefficients)
            R = cv2.Rodrigues(rvecs)[0]  # Extract rotation matrix
            ext = np.hstack((R, tvecs))  # Concatenate rotation and translation
            extrinsic_matrices.append(ext)
            
    for i, em in enumerate(extrinsic_matrices):
        if i == spin_value:
            print("Extrinsic:")
            print(em)

# Q1.4 Find Distortion Matrix
def FindDistortionMatrix():
    global distortion_coefficients
    print("Distortion:")
    print(distortion_coefficients)

# Q1.5 Show Result
# Undistort the chessboard images.
# Show disctorted and undistorted images.
def ShowResult():
    all_image_files = glob.glob(allfiles + "/*.bmp")
    for img_file in all_image_files:
        image = cv2.imread(img_file)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coefficients, (w, h), 1, (w, h))
        # Undistort the image
        dst = cv2.undistort(img, intrinsic_matrix, distortion_coefficients, None, newcameramtx)
        # Display the original and undistorted images side by side
        imgs = np.hstack([img, dst])
        cv2.imshow('Distorted Image', imgs)
        cv2.waitKey(1)  # Display the distorted image for a brief moment
        cv2.destroyAllWindows()
        

class AugmentedReality(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.ui_AugmentedReality()
        self.setFixedWidth(250)
        self.disparity_map = None 
        self.intrinisic_matrix = None
        self.distorion_coefficients = None
        self.rvecs = None
        self.tvecs = None
        self.text = ""
        
    def ui_AugmentedReality(self):
        
        self.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setLineWidth(3)
        self.setObjectName("AugmentedReality")
        
        self.label = QtWidgets.QLabel(self)
        self.label.setText('2. Augmented Reality')

        self.vlayout = QtWidgets.QVBoxLayout(self)
        self.vlayout.setAlignment(QtCore.Qt.AlignTop)
        self.vlayout.addWidget(self.label)

        #line edit
        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setObjectName("lineEdit")
        self.vlayout.addWidget(self.lineEdit)
        self.vlayout.addSpacing(20)
        self.lineEdit.textChanged.connect(self.showMsg)

        #two buttons
        self.btn1 = QtWidgets.QPushButton(self)
        self.btn1.setText('2.1 Show Word on Board')
        self.btn1.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn1) 
        self.vlayout.addSpacing(20)
        self.btn1.clicked.connect(self.show_text_on_board)
        
        self.btn2 = QtWidgets.QPushButton(self)
        self.btn2.setText('2.2 Show Words Vertically')
        self.btn2.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn2)
        self.vlayout.addSpacing(20)
        self.btn2.clicked.connect(self.show_verticaltext_on_board)

    def showMsg(self, text):
        print(text)
        self.text = text  #store the text in the line edit

# Q2.1 Show Word on Board
    def find_intrinsic(self):
        global imgs
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
        self.p_size= (11,8)
        objp = np.zeros((11*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        for f in imgs:
            img = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
            find, corners = cv2.findChessboardCorners(img, self.p_size, None)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(img, corners, (5,5), (-1,-1), criteria)
            if find:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
        if len(self.objpoints) > 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.p_size, None, None)
            print("Intrinsic:")
            print(mtx)
            self.intrinsic_matrix = mtx
            self.distortion_coefficients = dist
            self.rvecs = rvecs
            self.tvecs = tvecs  
        else:
            print("No calibration pattern found in the images.")
    
    def show_text_on_board(self):
        self.find_intrinsic()
        path = 'Dataset_CvDl_Hw1/Q2_lib/alphabet_lib_onboard.txt'
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)  # Dictionary
        text_coordinates = {}  # Dictionary to store text and corresponding pixel coordinates
        offset = {
        0: [7, 5, 0],
        1: [4, 5, 0],
        2: [1, 5, 0],
        3: [7, 2, 0],
        4: [4, 2, 0],
        5: [1, 2, 0]
        }
        for idx,letter in enumerate(self.text):
            character_coordinate = fs.getNode(letter).mat()  # Get the coordinate of each letter  
            character_coordinate = character_coordinate.reshape(-1, 3)
            # print(character_coordinate)

            if letter not in text_coordinates:
                text_coordinates[letter] = []  # Initialize the list for the letter
            
            for i,c in enumerate(character_coordinate):
                c[0]+= offset[idx][0]
                c[1]+= offset[idx][1]
                c[2]+= offset[idx][2]
                character_coordinate[i] = c    
           
            text_coordinates[letter].append(character_coordinate)
        
        # print(text_coordinates)
        for i, img in enumerate(imgs):
            img = img.copy()
            rvec = self.rvecs[i]
            tvec = self.tvecs[i]
            # Project the coordinates to the image
            for letter, coordinates_list in text_coordinates.items():
            # Gather all coordinates for the letter 'letter'
                coordinates = np.array(coordinates_list, dtype=np.float32)
                # coordinates = np.array(coordinates_list, dtype=np.float32)
                coordinates = coordinates.reshape(-1, 3)
                if coordinates.size > 0:  # Checking if there are coordinates
                    try:
                        image_points, _ = cv2.projectPoints(coordinates, rvec, tvec, self.intrinsic_matrix, self.distortion_coefficients)
                    except cv2.error as e:
                        print(f"Error projecting points for letter '{letter}' in image {i}: {e}")
                        continue  # Move to the next image in case of an error
                # print(image_points)
                # print(image_points.shape)
                # print(type(image_points))
                # Reshaping the image points to a 2D array
                image_points = image_points.reshape(-1, 2)
                # Draw lines using cv2.line
                for i in range(0, len(image_points) - 1, 2):
                    pt1 = tuple(map(int, image_points[i]))
                    pt2 = tuple(map(int, image_points[i + 1]))
                    cv2.line(img, pt1, pt2, (0, 0, 255), 2)
                        
                # Display or save the image
            cv2.imshow(f"Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
#Q2.2 Show Words Vertically
    def show_verticaltext_on_board(self):
        self.find_intrinsic()
        path = 'Dataset_CvDl_Hw1/Q2_lib/alphabet_lib_vertical.txt'
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        text_coordinates = {}  # Dictionary to store text and corresponding pixel coordinates
        offset = {
        0: [7, 5, 0],
        1: [4, 5, 0],
        2: [1, 5, 0],
        3: [7, 2, 0],
        4: [4, 2, 0],
        5: [1, 2, 0]
        }
        for idx,letter in enumerate(self.text):
            character_coordinate = fs.getNode(letter).mat()  # Get the coordinate of each letter  
            character_coordinate = character_coordinate.reshape(-1, 3)
            print(character_coordinate)

            if letter not in text_coordinates:
                text_coordinates[letter] = []  # Initialize the list for the letter
            for i,c in enumerate(character_coordinate):
                c[0]+= offset[idx][0]
                c[1]+= offset[idx][1]
                c[2]+= offset[idx][2]
                character_coordinate[i] = c    
            text_coordinates[letter].append(character_coordinate)
        
        print(text_coordinates)
        for i, img in enumerate(imgs):
            img = img.copy()
            rvec = self.rvecs[i]
            tvec = self.tvecs[i]
            for letter, coordinates_list in text_coordinates.items():
            # Gather all coordinates for the letter 'letter'
                coordinates = np.array(coordinates_list, dtype=np.float32)
                # coordinates = np.array(coordinates_list, dtype=np.float32)
                coordinates = coordinates.reshape(-1, 3)
                if coordinates.size > 0:  
                    try:
                        image_points, _ = cv2.projectPoints(coordinates, rvec, tvec, self.intrinsic_matrix, self.distortion_coefficients)
                    except cv2.error as e:
                        print(f"Error projecting points for letter '{letter}' in image {i}: {e}")
                        continue  # Move to the next image in case of an error
                image_points = image_points.reshape(-1, 2)
                # Draw lines using cv2.line
                for i in range(0, len(image_points) - 1, 2):
                    pt1 = tuple(map(int, image_points[i]))
                    pt2 = tuple(map(int, image_points[i + 1]))
                    cv2.line(img, pt1, pt2, (0, 0, 255), 2)
            cv2.imshow(f"Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


class StereoDisparityMap(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.ui_StereoDisparityMap()
        self.setFixedWidth(250)
        
    def ui_StereoDisparityMap(self):
        self.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setLineWidth(3)
        self.setObjectName("StereoDisparityMap")

        self.label = QtWidgets.QLabel(self)
        self.label.setText('3. Stereo Disparity Map')
        
        self.vlayout = QtWidgets.QVBoxLayout(self)
        
        self.btn1 = QtWidgets.QPushButton(self)
        self.btn1.setText('3.1 Stereo Disparity Map')
        self.btn1.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn1)
        self.vlayout.addSpacing(20)
        self.vlayout.setAlignment(QtCore.Qt.AlignVCenter)
        self.vlayout.setAlignment(QtCore.Qt.AlignHCenter)
        self.btn1.clicked.connect(self.Stereo_map)
        
# Q3.1 Stereo Disparity Map
    def Stereo_map(self):
        global image_L, image_R
        imgL = cv2.cvtColor(image_L, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(image_R, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        self.disparity_map = stereo.compute(imgL, imgR)
        # Scale disparity map to the 0-255 range for visualization
        self.disparity_visual = cv2.normalize(self.disparity_map, None, alpha=0, 
                                          beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow('disparity', self.disparity_visual)
        self.show_imgL_R()
        cv2.setMouseCallback('Left Image', self.check_disparity_value)
         
    def show_imgL_R(self):
        cv2.imshow('Left Image', image_L)
        cv2.imshow('Right Image', image_R)
    #Q3.2 Checking the Disparity Value
    def check_disparity_value(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.disparity_map is not None:
                disparity_value = self.disparity_visual[y, x]
                #compute right corresponding point
                if disparity_value != 0:
                    right_x = x - disparity_value
                    print(f"({x}, {y}), Dis: {disparity_value}")
                    image_R_copy = image_R.copy()
                     #draw circle on the right image
                    cv2.circle(image_R_copy, (int(right_x), y), 20, (0, 255, 0), 10)
                    cv2.imshow("Right Image", image_R_copy)
                else:
                    print("failure case")
                          
        if event == cv2.EVENT_RBUTTONDOWN:
            cv2.destroyAllWindows()

    
class SIFT(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.ui_SIFT()
        self.setFixedWidth(250)
        
    def ui_SIFT(self):
        self.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setLineWidth(3)
        self.setObjectName("SIFT")

        self.label = QtWidgets.QLabel(self)
        self.label.setText('4. SIFT')
        
        self.vlayout = QtWidgets.QVBoxLayout(self)
        self.vlayout.setAlignment(QtCore.Qt.AlignTop)
        self.vlayout.addWidget(self.label)

        # four buttons
        self.btn1 = QtWidgets.QPushButton(self)
        self.btn1.setText('Load Image 1')
        self.btn1.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn1)
        self.vlayout.addSpacing(20)
        self.btn1.clicked.connect(load_image1)

        self.btn2 = QtWidgets.QPushButton(self)
        self.btn2.setText('Load Image 2')
        self.btn2.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn2)
        self.vlayout.addSpacing(20)
        self.btn2.clicked.connect(load_image2)

        self.btn3 = QtWidgets.QPushButton(self)
        self.btn3.setText('4.1 Keypoints')
        self.btn3.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn3)
        self.vlayout.addSpacing(20)
        self.btn3.clicked.connect(SIFT_KeyPoint)
        
        self.btn4 = QtWidgets.QPushButton(self)
        self.btn4.setText('4.2 Matched Keypoints')
        self.btn4.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn4)
        self.btn4.clicked.connect(SIFT_MatchedKeyPoint)

def load_image1():
    filePath, filterType = QtWidgets.QFileDialog.getOpenFileName()
    global img1_path
    img1_path = filePath
    
#Q4.1 Keypoints
def SIFT_KeyPoint():
    img1 = cv2.imread(img1_path,0)          
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    #Based on SIFT algorithm, find  key points on Left.jpg
    img1 = cv2.drawKeypoints(img1,kp1,None,color=(0,255,0))
    plt.imshow(img1)
    plt.show()

def load_image2():
    filePath, filterType = QtWidgets.QFileDialog.getOpenFileName()
    global img2_path
    img2_path = filePath

#Q4.2 Matched Keypoints
def SIFT_MatchedKeyPoint():
    img1 = cv2.imread(img1_path,0)          
    img2 = cv2.imread(img2_path,0) 
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    #Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()


class VGG19(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.ui_VGG19()
        self.setFixedWidth(250)
        
        
    def ui_VGG19(self):    
        self.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setLineWidth(3)
        self.setObjectName("VGG19")
    
        self.label = QtWidgets.QLabel(self)
        self.label.setText('5. VGG19')
            
        self.vlayout = QtWidgets.QVBoxLayout(self)
        self.vlayout.setAlignment(QtCore.Qt.AlignCenter)
        self.vlayout.addWidget(self.label)
    
        # four buttons
        self.btn1 = QtWidgets.QPushButton(self)
        self.btn1.setText('Load Image')
        self.btn1.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn1)
        self.vlayout.addSpacing(20)
        self.btn1.clicked.connect(self.load_inference_img)
            
    
        self.btn2 = QtWidgets.QPushButton(self)
        self.btn2.setText('5.1 show aumented image')
        self.btn2.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn2)
        self.vlayout.addSpacing(20)
        self.btn2.clicked.connect(show_augmentedimg)
    
        self.btn3 = QtWidgets.QPushButton(self)
        self.btn3.setText('5.2 Show Model structure')
        self.btn3.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn3)
        self.vlayout.addSpacing(20)
        self.btn3.clicked.connect(show_model)
            
        self.btn4 = QtWidgets.QPushButton(self)
        self.btn4.setText('5.3 Show Acc&Loss')
        self.btn4.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn4)
        self.vlayout.addSpacing(20)
        self.btn4.clicked.connect(show_acc_loss)

        self.btn5 = QtWidgets.QPushButton(self)
        self.btn5.setText('5.4 Inference')
        self.btn5.setFixedSize(200, 40)
        self.vlayout.addWidget(self.btn5)
        self.vlayout.addSpacing(20)
        self.btn5.clicked.connect(self.Inference)

        self.label_2 = QtWidgets.QLabel("Predict=")
        self.graphics = QtWidgets.QGraphicsView()
        self.graphics.setScene(QtWidgets.QGraphicsScene())
        self.graphics.scene().addText("Inference Image")
        self.graphics.setAlignment(QtCore.Qt.AlignCenter) 
        self.vlayout.addWidget(self.label_2)  
        self.vlayout.addWidget(self.graphics)
        self.vlayout.addSpacing(20)
            
#Q5.1 show aumented image with labels
    def load_inference_img(self):
        filePath, filetype = QtWidgets.QFileDialog.getOpenFileName()
        self.inference_img = Image.open(filePath)
        # Show the inference image
        pixmap = QtGui.QPixmap(filePath)
        pixmap = pixmap.scaled(128,128, QtCore.Qt.KeepAspectRatio)
        self.graphics.scene().clear()
        self.graphics.setFixedSize(150,150)
        self.graphics.scene().addPixmap(pixmap)      

    #Q5.4 Inference
    def Inference(self):
        # Load the img & preprocess
        inference_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))
        ])
        inference_img = inference_transforms(self.inference_img)

        #inference
        with torch.no_grad():
            model = torchvision.models.vgg19_bn(num_classes=10)
            model.load_state_dict(torch.load('CIFAR_VGG19.pth',map_location=torch.device('cpu')))
            model.eval()
            output = model(inference_img.unsqueeze(0))
            
            classes = ['airplane', 'cars', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] #CIFAR10 classes
            prob = torch.nn.functional.softmax(output, dim=1)[0]
            _, predicted = torch.max(output.data, 1)

            self.label_2.setText("Predict={}".format(classes[predicted.item()]))
            plt.figure(figsize=(10, 10))
            plt.bar(classes, prob)
            plt.title('Probability Distribution')
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.show()


def show_augmentedimg():
    augmented_img = []
    Dtransforms = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30)
    ])
    for i in PIL_imgs:
        augmented = Dtransforms(i)
        augmented_img.append(augmented)
    # Labels with image
    labels = [os.path.splitext(fn)[0] for fn in filenames]
    # Show the augmented images
    plt.figure(figsize=(10, 10))
    for i, (img, label) in enumerate(zip(augmented_img, labels)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(label)    
    plt.show()

#Q5.2 Show Model structure
def show_model():
    model = torchvision.models.vgg19_bn(num_classes=10)
    model.load_state_dict(torch.load('CIFAR_VGG19.pth',map_location=torch.device('cpu')))
    # Show the model structure
    torchsummary.summary(model, (3, 32, 32))

#Q5.3 Show Acc&Loss
def show_acc_loss():
    img= Image.open('VGG19.png')
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    Form = MainWindow()
    Form.show()
    sys.exit(app.exec_())