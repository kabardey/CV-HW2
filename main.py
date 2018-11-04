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

        #pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio)
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
        three = QAction('3x3', self)
        five = QAction('5x5', self)
        seven = QAction('7x7', self)
        nine = QAction('9x9', self)
        eleven = QAction('11x11', self)
        thirteen = QAction('13x13', self)
        fifteen = QAction('15x15', self)

        average_filters.addAction(three)
        average_filters.addAction(five)
        average_filters.addAction(seven)
        average_filters.addAction(nine)
        average_filters.addAction(eleven)
        average_filters.addAction(thirteen)
        average_filters.addAction(fifteen)

        filters.addMenu(average_filters)

        gaussian_filters = QMenu('Gaussian Filters', self)

        gaussian_filters.addAction(three)
        gaussian_filters.addAction(five)
        gaussian_filters.addAction(seven)
        gaussian_filters.addAction(nine)
        gaussian_filters.addAction(eleven)
        gaussian_filters.addAction(thirteen)
        gaussian_filters.addAction(fifteen)

        filters.addMenu(gaussian_filters)

        median_filters = QMenu('Median Filters', self)

        median_filters.addAction(three)
        median_filters.addAction(five)
        median_filters.addAction(seven)
        median_filters.addAction(nine)
        median_filters.addAction(eleven)
        median_filters.addAction(thirteen)
        median_filters.addAction(fifteen)

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