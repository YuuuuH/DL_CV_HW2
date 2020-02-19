import numpy as np
import cv2
from matplotlib import pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
import os

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(436, 292)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.btn1_1 = QtWidgets.QPushButton(self.groupBox)
        self.btn1_1.setMinimumSize(QtCore.QSize(0, 50))
        self.btn1_1.setObjectName("btn1_1")
        self.verticalLayout_3.addWidget(self.btn1_1)
        self.verticalLayout.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.btn2_1 = QtWidgets.QPushButton(self.groupBox_2)
        self.btn2_1.setMinimumSize(QtCore.QSize(0, 50))
        self.btn2_1.setObjectName("btn2_1")
        self.verticalLayout_4.addWidget(self.btn2_1)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.btn3_1 = QtWidgets.QPushButton(self.groupBox_3)
        self.btn3_1.setMinimumSize(QtCore.QSize(0, 50))
        self.btn3_1.setObjectName("btn3_1")
        self.verticalLayout_5.addWidget(self.btn3_1)
        self.btn3_2 = QtWidgets.QPushButton(self.groupBox_3)
        self.btn3_2.setMinimumSize(QtCore.QSize(0, 50))
        self.btn3_2.setObjectName("btn3_2")
        self.verticalLayout_5.addWidget(self.btn3_2)
        self.verticalLayout_2.addWidget(self.groupBox_3)
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.btn4_1 = QtWidgets.QPushButton(self.groupBox_4)
        self.btn4_1.setMinimumSize(QtCore.QSize(0, 50))
        self.btn4_1.setObjectName("btn4_1")
        self.verticalLayout_6.addWidget(self.btn4_1)
        self.verticalLayout_2.addWidget(self.groupBox_4)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.btn1_1.clicked.connect(self.question1)
        self.btn2_1.clicked.connect(self.question2)
        self.btn3_1.clicked.connect(self.question3_1)
        self.btn3_2.clicked.connect(self.question3_2)
        self.btn4_1.clicked.connect(self.showq4)
        

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "1. Stereo"))
        self.btn1_1.setText(_translate("MainWindow", "1.1 Display"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2. Background Subtraction"))
        self.btn2_1.setText(_translate("MainWindow", "2.1 Background Subtraction"))
        self.groupBox_3.setTitle(_translate("MainWindow", "3. Feature Tracking"))
        self.btn3_1.setText(_translate("MainWindow", "3.1 Preprocessing"))
        self.btn3_2.setText(_translate("MainWindow", "3.2 Video Tracking"))
        self.groupBox_4.setTitle(_translate("MainWindow", "4. Augmented Reality"))
        self.btn4_1.setText(_translate("MainWindow", "4.1 Augmented Reality"))
      
    def question1(self): 
        imgL = cv2.imread('imL.png',0)
        imgR = cv2.imread('imR.png',0)
        stereo = cv2.StereoSGBM_create(minDisparity=0,
                               numDisparities=64,
                               blockSize=9,
                               uniquenessRatio=5,
                               speckleRange=1,
                               speckleWindowSize=190,
                               disp12MaxDiff=1,
                               P1=120,
                               P2=1500)
        disparity = stereo.compute(imgL,imgR)
        disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow('gray',disparity)
        cv2.waitKey(0)
        cv2.destroyAllwindows()
    def question2(self):
        cap = cv2.VideoCapture('bgSub.mp4')
        knn = cv2.createBackgroundSubtractorKNN(history=50,dist2Threshold=5000.0,detectShadows=True)
        while True :
            ret, frame = cap.read()
            fgmask = knn.apply(frame)
            cv2.imshow('frame', fgmask) 
            cv2.imshow('original_frame', frame)
            if cv2.waitKey(30) & 0xff == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    def question3_1(self):
        cap = cv2.VideoCapture("featureTracking.mp4") 
        _, first_frame = cap.read()
        #a = cv2.imread("a.jpeg",0)
        im = cv2.cvtColor(first_frame , cv2.COLOR_BGR2GRAY)
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 145


        # Filter by Area.
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 100

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.83
        params.maxCircularity = 1.0

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.8
        params.maxConvexity = 1.0

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.3
        params.maxInertiaRatio = 1.0

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)


        # Detect blobs.
        keypoints = detector.detect(im)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob
        #for keyPoint in keypoints:
        #    x = keyPoint.pt[0]
        #    y = keyPoint.pt[1]
        #    s = keyPoint.size
        #    print("point",keyPoint,"_x is ",x)
        #    print("point",keyPoint,"_y is ",y)
        #    print("point size is",s)

        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        im_with_rec = cv2.rectangle(first_frame,(124,254),(135,265),(0,0,255),1)
        im_with_rec = cv2.rectangle(first_frame,(130,235),(141,246),(0,0,255),1)
        im_with_rec = cv2.rectangle(first_frame,(112,66),(123,77),(0,0,255),1)
        im_with_rec = cv2.rectangle(first_frame,(113,163),(124,174),(0,0,255),1)
        im_with_rec = cv2.rectangle(first_frame,(106,90),(117,101),(0,0,255),1)
        im_with_rec = cv2.rectangle(first_frame,(188,252),(198,262),(0,0,255),1)
        im_with_rec = cv2.rectangle(first_frame,(170,263),(180,273),(0,0,255),1)
        # Show blobs
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.imshow("im_with_rec",im_with_rec)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def question3_2(selt):
        cap = cv2.VideoCapture('featureTracking.mp4')
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 7,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )

        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (14,14),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        #color = np.random.randint(0,0,255)

        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        #p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        #print(type(p0))
        #print(p0)
        #a=p0.ndim
        #print(a)
        p0 = np.array([[[129.74978637695312,259.7417907714844]],[[135.10911560058594,240.959716796875]],[[117.49764251708984,72.00044250488281]],
                       [[175.75177001953125,268.46044921875]],[[118.66619873046875,169.08338928222656]],[[111.38301849365234,95.51689147949219]],
                       [[193.0832061767578,257.392822265625]]],dtype=np.float32)
        print(type(p0))
        print(p0)
        c=p0.ndim
        print(c)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        while(1):
            ret,frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray,p0, None, **lk_params)

            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]

            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), (0,0,255), 2)
                frame = cv2.circle(frame,(a,b),5,(0,0,255),-1)
            img = cv2.add(frame,mask)

            cv2.imshow('frame',img)
            k = cv2.waitKey(50) & 0xff
            if k == 27:
                break

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)

        cv2.destroyAllWindows()
        cap.release()

    
    def question4(self,index):
        root_dir = os.getcwd()
        print(root_dir)
        #pic_dir = os.path.join(root_dir)
        Intrinsic = np.array([[2225.49585482, 0, 1025.5459589], [0, 2225.18414074, 1038.58578846], [0, 0, 1]])
        distortion = np.array([[-0.12874225, 0.09057782, -0.00099125, 0.00000278, 0.0022925]])
        bmp1 = np.array(
            [[-0.97157425, -0.01827487, 0.23602862, 6.81253889], [0.07148055, -0.97312723, 0.2188925, 3.37330384],
             [0.22568565, 0.22954177, 0.94677165, 16.74572319]])
        bmp2 = np.array(
            [[-0.8884799,-0.14530922,-0.4353030,3.3925504], [0.07148066,-0.98078915,0.18150248,4.36149229],
             [-0.45331444,0.13014556,0.88179825,22.15957429]])
        bmp3 = np.array(
            [[-0.52390938,0.22312793,0.82202974,2.68774801], [0.00530458,-0.96420621,0.26510049,4.70990021],
             [0.85175749,0.14324914,0.50397308,12.98147662]])
        bmp4 = np.array(
            [[-0.63108673,0.53013053,0.566296,1.22781875], [0.13263301,-0.64553994,0.75212145,3.48023006],
             [0.76428923,0.54976341,0.33707888,10.9840538]])
        bmp5 = np.array(
            [[-0.87676843,-0.23020567,0.42223508,4.43641198], [0.19708207,-0.97286949,-0.12117596,0.67177428],
             [0.43867502,-0.02302829,0.89835067,16.24069227]])
        point = np.array([[1, 1, 0], [1, 5, 0], [5, 5, 0], [5, 1, 0],[3, 3, -4]],dtype=np.float32)
        dict = {1: '1.bmp', 2: '2.bmp', 3: '3.bmp', 4: '4.bmp', 5: '5.bmp'}
        dict_extrinsic = {1: bmp1,2: bmp2, 3: bmp3, 4: bmp4, 5: bmp5}
        AR_result = []
        #file = os.path.join(pic_dir,dict[index])
        trans = dict_extrinsic[index][:,3:]
        rot = dict_extrinsic[index][:,0:3]
        img = cv2.imread(dict[index])
        point2d, _ = cv2.projectPoints(point, rot, trans, Intrinsic, distortion)
        for i in range(0,len(point2d)-1):
            if i == 3 :
                y = 0
            else :
                y = i+1
            cv2.line(img,tuple(point2d[i][0]),tuple(point2d[y][0]),(0,0,255),10)
            cv2.line(img, tuple(point2d[i][0]), tuple(point2d[4][0]),(0, 0, 255),10)
        return  img
    def showq4(self):
        img = []
        AR_result = []
        for i in range(1,6):
            img = self.question4(i)
            AR_result.append(img)
        
        cv2.namedWindow("result",cv2.WINDOW_NORMAL)
        for i in range(0,5):
            cv2.imshow("result",AR_result[i])
            if cv2.waitKey(500) == 27 :
                    break
        cv2.destroyAllWindows()        

if(__name__ == '__main__'):
     import sys
     app = QtWidgets.QApplication(sys.argv)
     MainWindow = QtWidgets.QMainWindow()
     ui = Ui_MainWindow()
     ui.setupUi(MainWindow)
     MainWindow.show()
     sys.exit(app.exec_())