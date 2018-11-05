import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QHBoxLayout, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, \
    QPushButton, QGroupBox, QAction, QFileDialog, qApp
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import cv2

##########################################
## Do not forget to delete "return NotImplementedError"
## while implementing a function
########################################

class App(QMainWindow):
    def __init__(self):
        super(App, self).__init__()
        self.title = 'Filtering & Geometric Transformation'
        self.left = 10
        self.top = 10
        self.width = 1000
        self.height = 600

        self.count = 0

        self.initUI()

    def openImage(self):
        # ******** place image into qlabel object *********************
        imagePath, _ = QFileDialog.getOpenFileName()
        self.inputImg = cv2.imread(imagePath)


        pixmap_label = self.qlabel1

        height, width, channel = self.inputImg.shape
        bytesPerLine = 3 * width
        qImg = QImage(self.inputImg.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(qImg)

        #pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio)
        pixmap_label.setPixmap(pixmap)

        self.count = self.count + 1
        # **************************************************************

    def saveImage(self):
        # This function is called when the user clicks File->Target Image.
        return NotImplementedError

    def padding(self, padding_size, img):  # fill the outer border pixels according to kernel size

        height, width, channel = img.shape

        new_img = np.zeros((height + padding_size*2, width + padding_size*2, 3), dtype=np.uint8)
        new_img[padding_size:(height+padding_size), padding_size:(width+padding_size), :] = img

        return new_img

    def avg_filter(self, kernel_size): # create average kernel

        kernel = (1 / (kernel_size*kernel_size))*np.ones((kernel_size, kernel_size), np.float32)
        return kernel

    def conv(self, kernel, img, height, width, padding):

        height2, width2, channel = self.inputImg.shape
        temp_image = np.zeros((height2, width2, 3), dtype=np.uint8)

        for i in range(padding, height+padding):
            for j in range(padding, width+padding):

                roi = img[i - padding:i + padding + 1, j - padding:j + padding + 1, 0] # take the part of image for convolution operation
                conv_value = (roi * kernel).sum()
                temp_image[i - padding, j - padding, 0] = int(conv_value)

                roi = img[i - padding:i + padding + 1, j - padding:j + padding + 1, 1]  # take the part of image for convolution operation
                conv_value = (roi * kernel).sum()
                temp_image[i - padding, j - padding, 1] = int(conv_value)

                roi = img[i - padding:i + padding + 1, j - padding:j + padding + 1, 2]  # take the part of image for convolution operation
                conv_value = (roi * kernel).sum()
                temp_image[i - padding, j - padding, 2] = int(conv_value)

        # place the image to the qlabel
        pixmap_label = self.qlabel1
        bytesPerLine = 3 * width2
        qImg = QImage(temp_image.data, width2, height2, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(qImg)
        pixmap_label.setPixmap(pixmap)


    def average_3(self):  # average filter 3x3
        if (self.count != 1):
            return QMessageBox.question(self, 'Error Message', "Please, load the image", QMessageBox.Ok, QMessageBox.Ok)

        height, width, channel = self.inputImg.shape

        image = self.padding(1, self.inputImg)  # add padding to the image
        avg_kernel = self.avg_filter(3)  # create average kernel
        self.conv(avg_kernel, image, height, width, padding=1)  # apply convolution


    def average_5(self):  # average filter 5x5
        if (self.count != 1):
            return QMessageBox.question(self, 'Error Message', "Please, load the image", QMessageBox.Ok, QMessageBox.Ok)

        height, width, channel = self.inputImg.shape

        image = self.padding(2, self.inputImg)  # add padding to the image
        avg_kernel = self.avg_filter(5)  # create average kernel
        self.conv(avg_kernel, image, height, width, padding=2)  # apply convolution


    def average_7(self):  # average filter 7x7
        if (self.count != 1):
            return QMessageBox.question(self, 'Error Message', "Please, load the image", QMessageBox.Ok, QMessageBox.Ok)

        height, width, channel = self.inputImg.shape

        image = self.padding(3, self.inputImg)  # add padding to the image
        avg_kernel = self.avg_filter(7)  # create average kernel
        self.conv(avg_kernel, image, height, width, padding=3)  # apply convolution


    def average_9(self):  # average filter 9x9
        if (self.count != 1):
            return QMessageBox.question(self, 'Error Message', "Please, load the image", QMessageBox.Ok, QMessageBox.Ok)

        height, width, channel = self.inputImg.shape

        image = self.padding(4, self.inputImg)  # add padding to the image
        avg_kernel = self.avg_filter(9)  # create average kernel
        self.conv(avg_kernel, image, height, width, padding=4)  # apply convolution


    def average_11(self):  # average filter 11x11
        if (self.count != 1):
            return QMessageBox.question(self, 'Error Message', "Please, load the image", QMessageBox.Ok, QMessageBox.Ok)

        height, width, channel = self.inputImg.shape

        image = self.padding(5, self.inputImg)  # add padding to the image
        avg_kernel = self.avg_filter(11)  # create average kernel
        self.conv(avg_kernel, image, height, width, padding=5)  # apply convolution


    def average_13(self):  # average filter 13x13
        if (self.count != 1):
            return QMessageBox.question(self, 'Error Message', "Please, load the image", QMessageBox.Ok, QMessageBox.Ok)

        height, width, channel = self.inputImg.shape

        image = self.padding(6, self.inputImg)  # add padding to the image
        avg_kernel = self.avg_filter(13)  # create average kernel
        self.conv(avg_kernel, image, height, width, padding=6)  # apply convolution


    def average_15(self):  # average filter 15x15
        if (self.count != 1):
            return QMessageBox.question(self, 'Error Message', "Please, load the image", QMessageBox.Ok, QMessageBox.Ok)

        height, width, channel = self.inputImg.shape

        image = self.padding(7, self.inputImg)  # add padding to the image
        avg_kernel = self.avg_filter(15)  # create average kernel
        self.conv(avg_kernel, image, height, width, padding=7)  # apply convolution


    def rotate_left(self):
        if (self.count != 1):
            return QMessageBox.question(self, 'Error Message', "Please, load the image", QMessageBox.Ok, QMessageBox.Ok)

        height, width, channel = self.inputImg.shape
        centerx = width // 2
        centery = height // 2

        result_image = np.zeros((height, width, 3))

        angle = np.pi/6

        for j in range(0, height):
            for k in range(0, width):
                rot_mat = np.asarray([[np.cos(angle), np.sin(angle)], [-1 * np.sin(angle), np.cos(angle)]])
                coord = np.matmul(rot_mat, np.asarray([k - centerx, j - centery]))
                coord += [centerx, centery]

                pixel_value = self.inputImg[j, k, :]  # take original image pixel value BGR
                result_image[int(coord[0]), int(coord[1]), :] = pixel_value  # place to the new coord

        pixmap_label = self.qlabel1
        bytesPerLine = 3 * width
        qImg = QImage(result_image.data, (width), (height), bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(qImg)
        pixmap_label.setPixmap(pixmap)


    def rotate_right(self):
        return NotImplementedError

    def scale_twox(self):

        if(self.count != 1):
            return QMessageBox.question(self, 'Error Message', "Please, load the image", QMessageBox.Ok, QMessageBox.Ok)


        height, width, channel = self.inputImg.shape
        centerx = width // 2
        centery = height // 2

        print('centerx: ', centerx)
        print('centery: ', centery)

        scale = 2

        result_image = np.zeros((height*scale, width*scale, 3))


        for j in range(0, height):
            for k in range(0, width):
                scale_mat = np.eye(2)*2
                scale_mat = np.linalg.inv(scale_mat)
                coord = np.matmul(scale_mat, np.asarray([k - centerx, j - centery]))
                #print('coord: ', coord)
                #coord += [centerx, centery]

                pixel_value = self.inputImg[j, k, :]  # take original image pixel value BGR
                result_image[int(coord[0]), int(coord[1]), :] = pixel_value  # place to the new coord


        pixmap_label = self.qlabel1
        bytesPerLine = 3 * (width*scale)
        qImg = QImage(result_image.data, (width*scale), (height*scale), bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(qImg)

        pixmap_label.setPixmap(pixmap)

    def scale_oneovertwox(self):
        return NotImplementedError

    def trans_right(self):
        return NotImplementedError

    def trans_left(self):
        return NotImplementedError



    def initUI(self):
        # Write GUI initialization code

        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowTitle(self.title)

        # ****************add the label for image*********************
        wid = QWidget(self)
        self.setCentralWidget(wid)

        self.groupBox = QGroupBox()
        self.hBoxlayout = QHBoxLayout()

        self.qlabel1 = QLabel('Image', self)
        self.qlabel1.setStyleSheet("border: 1px inset grey; min-height: 200px; ")
        self.qlabel1.setAlignment(Qt.AlignCenter)
        self.hBoxlayout.addWidget(self.qlabel1)

        self.groupBox.setLayout(self.hBoxlayout)

        vBox = QVBoxLayout()
        vBox.addWidget(self.groupBox)
        wid.setLayout(vBox)


        # ****************menu bar***********
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File')
        filters = menubar.addMenu('Filters')
        geometric_transform = menubar.addMenu('Geometric Transform')

        openAction = QAction('Open Image', self)
        openAction.triggered.connect(self.openImage)
        fileMenu.addAction(openAction)

        openAction2 = QAction('Save Image', self)
        openAction2.triggered.connect(self.saveImage)
        fileMenu.addAction(openAction2)

        exitAct = QAction(QIcon('exit.png'), '&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)
        fileMenu.addAction(exitAct)

        average_filters = QMenu('Average Filters', self)
        three_avg = QAction('3x3', self)
        five_avg = QAction('5x5', self)
        seven_avg = QAction('7x7', self)
        nine_avg = QAction('9x9', self)
        eleven_avg = QAction('11x11', self)
        thirteen_avg = QAction('13x13', self)
        fifteen_avg = QAction('15x15', self)

        three_avg.triggered.connect(self.average_3)
        five_avg.triggered.connect(self.average_5)
        seven_avg.triggered.connect(self.average_7)
        nine_avg.triggered.connect(self.average_9)
        eleven_avg.triggered.connect(self.average_11)
        thirteen_avg.triggered.connect(self.average_13)
        fifteen_avg.triggered.connect(self.average_15)

        average_filters.addAction(three_avg)
        average_filters.addAction(five_avg)
        average_filters.addAction(seven_avg)
        average_filters.addAction(nine_avg)
        average_filters.addAction(eleven_avg)
        average_filters.addAction(thirteen_avg)
        average_filters.addAction(fifteen_avg)

        filters.addMenu(average_filters)

        gaussian_filters = QMenu('Gaussian Filters', self)
        three_gaus = QAction('3x3', self)
        five_gaus = QAction('5x5', self)
        seven_gaus = QAction('7x7', self)
        nine_gaus = QAction('9x9', self)
        eleven_gaus = QAction('11x11', self)
        thirteen_gaus = QAction('13x13', self)
        fifteen_gaus = QAction('15x15', self)

        gaussian_filters.addAction(three_gaus)
        gaussian_filters.addAction(five_gaus)
        gaussian_filters.addAction(seven_gaus)
        gaussian_filters.addAction(nine_gaus)
        gaussian_filters.addAction(eleven_gaus)
        gaussian_filters.addAction(thirteen_gaus)
        gaussian_filters.addAction(fifteen_gaus)

        filters.addMenu(gaussian_filters)

        median_filters = QMenu('Median Filters', self)
        three_med = QAction('3x3', self)
        five_med = QAction('5x5', self)
        seven_med = QAction('7x7', self)
        nine_med = QAction('9x9', self)
        eleven_med = QAction('11x11', self)
        thirteen_med = QAction('13x13', self)
        fifteen_med = QAction('15x15', self)

        median_filters.addAction(three_med)
        median_filters.addAction(five_med)
        median_filters.addAction(seven_med)
        median_filters.addAction(nine_med)
        median_filters.addAction(eleven_med)
        median_filters.addAction(thirteen_med)
        median_filters.addAction(fifteen_med)

        filters.addMenu(median_filters)



        ### 5. create rotate, scale and translate menu ###
        rotate = QMenu('Rotate', self)
        scale = QMenu('Scale', self)
        translate = QMenu('Translate', self)

        rotate_right = QAction('Rotate 10 Degree Right', self)
        rotate_left = QAction('Rotate 10 Degree Left', self)
        twox = QAction('2x', self)
        oneovertwox = QAction('1/2x', self)
        right = QAction('Right', self)
        left = QAction('Left', self)

        ### add function when the action is triggered ###
        rotate_right.triggered.connect(self.rotate_right)
        rotate_left.triggered.connect(self.rotate_left)
        twox.triggered.connect(self.scale_twox)
        oneovertwox.triggered.connect(self.scale_oneovertwox)
        right.triggered.connect(self.trans_right)
        left.triggered.connect(self.trans_left)

        ###  add action ###
        rotate.addAction(rotate_right)
        rotate.addAction(rotate_left)
        scale.addAction(twox)
        scale.addAction(oneovertwox)
        translate.addAction(right)
        translate.addAction(left)

        geometric_transform.addMenu(rotate)
        geometric_transform.addMenu(scale)
        geometric_transform.addMenu(translate)

        # ------------------------------------

        self.show()

    def histogramButtonClicked(self):
        if not self.inputLoaded and not self.targetLoaded:
            # Error: "First load input and target images" in MessageBox
            return NotImplementedError
        if not self.inputLoaded:
            # Error: "Load input image" in MessageBox
            return NotImplementedError
        elif not self.targetLoaded:
            # Error: "Load target image" in MessageBox
            return NotImplementedError

    def calcHistogram(self, I):
        # Calculate histogram
        return NotImplementedError

class PlotCanvas(FigureCanvas):
    def __init__(self, hist, parent=None, width=5, height=4, dpi=100):
        return NotImplementedError
        # Init Canvas
        self.plotHistogram(hist)

    def plotHistogram(self, hist):
        return NotImplementedError
        # Plot histogram

        self.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())