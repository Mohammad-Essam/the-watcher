from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QRadioButton, QComboBox,QAction, qApp, QSlider, QListWidget, QCheckBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QPoint
import sys, time
import cv2 as cv
import numpy as np
from mobilenet import *
from playsound import playsound
import threading

# GLOBALS
accuracy = 0.5
CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'monitor']
#thread for playing the alerting sound
soundThread = threading.Thread(target=playsound, args=('alertSound.mp3',), daemon=True)
# Dictionary of all objects of interest, that the user added to the watchlist
watchedObjects = {}

# Dictionary of all detected objects on screen
detectedObjects = {}

# A flag decides whether to draw bounding boxes over all detected objects
showAllObjects = True

# Control the threshold of the accuracy, in which the object confidence must exceed.
@pyqtSlot(int)
def setAccuracy(value):
    global accuracy
    accuracy = value / 10
    print(accuracy)

# Axis-Aligned Bounding Boxes Collision Detection algorithm, used to test if the intended object
# is in a forbidden zone or not.
def checkCollision(rect1, rect2):
    r1 = {'x': rect1[0], 'y': rect1[1], 'width': rect1[2] - rect1[0], 'height': rect1[3] - rect1[1]}
    r2 = {'x': rect2[0], 'y': rect2[1], 'width': rect2[2] - rect2[0], 'height': rect2[3] - rect2[1]}

    return (r1['x'] < r2['x'] + r2['width'] and
            r1['x'] + r1['width'] > r2['x'] and
            r1['y'] < r2['y'] + r2['height'] and
            r1['y'] + r1['height'] > r2['y'])


# A class that encapsulates an object and its needed information, such as 
# its specified forbidden zones and danger objects 
class WatchedObject:
    def __init__(self, name):
        self.name = name
        self.dangerZones = []
        self.dangerObjectsNames = []
        self.objectRange = 0

    # Test if the intended object is in a forbidden zone or not.
    def checkDanger(self):
        flag = False
        message = ""
        #check if the specified (this) object is detected
        if self.name in detectedObjects.keys():
            # iterate over all instances of detected object who have the same name 
            for detectedObj in detectedObjects[self.name]:
                # iterate over all the danger zones that the object has
                for dangerZone in self.dangerZones:
                    # Check if the object collide with its danger zones
                    if checkCollision(detectedObj['box'], dangerZone):
                        # Add coordinates of the object in danger zone
                        message += (f"Danger Detected for {self.name} at {detectedObj['box']} at zone {dangerZone}\n")
                        flag = True
                # Iterate over all the names of danger objects that the object has
                for name in self.dangerObjectsNames:
                    if name not in detectedObjects:
                        continue
                    # Iterate over all instances of detected objects assigned as dangerous
                    for dangerObj in detectedObjects[name]:
                        # Resolve the name of the dangerous object to its coordinates 
                        # and check collision with our object of interest
                        result = checkCollision(detectedObj['box'], dangerObj['box'])
                        if result:
                            message += (f"Danger Detected for {self.name} at {name}'s region\n")
                            flag = True
        return flag, message

# Convert image from cv image (np.ndarray) to QImage
def cvtoqt(image, width=0, height=0, noScale=True):
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    result = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    if not noScale:
        result = result.scaled(width, height, Qt.KeepAspectRatio)
    return result

def cartesianPoint(point: QPoint):
    return (point.x(), point.y())

# A separate thread for managing reading from a video or a webcam
# To prevent the main thread from blocking
class VideoThread(QThread):
    frameChange = pyqtSignal(np.ndarray)
    frameIncrease = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.counter = 0            # A counter for the frames

    def run(self):
        video = 0
        cap = cv.VideoCapture(video)
        while self._run_flag:
            ret, img = cap.read()
            if ret:
                # Resize the captured image, keeping aspect ratio
                resize_width = 640
                ratio = resize_width / img.shape[1]
                resize_height = int(ratio * img.shape[0])
                img = cv.resize(img, (resize_width, resize_height))

                # Detect objects from the image and fill detectedObjects with data
                # i.e. name, confidence, bounding box coordinates
                # Imported from the module mobilenet
                getDetectedObjects(img, accuracy, detectedObjects)
                self.counter+=1

                # Sending the captured image to the main thread to show it
                self.frameChange.emit(img)
                self.frameIncrease.emit(str(self.counter))
            else:
                cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class Point:
    def __init__(self, x, y):
        pass

# The container in which the image/video is displayed
class ImageLabel(QLabel):
    messageSignal = pyqtSignal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.current =  None   # TODO FIRST
        self.startPoint = None          # it contains (x, y) 
        self.endPoint = None
        self.isMouseDown = False
        self.image = None
        self.disableDrawing = False

    @pyqtSlot(np.ndarray)
    def drawCVImage(self, img):
        self.image = img
        self.updateImage()

    def updateImage(self):
        danger = False
        message = ""
        for obj in watchedObjects.values():
            xdanger, xmessage = obj.checkDanger()
            danger = xdanger or danger
            message += xmessage
        if danger:
            self.messageSignal.emit(message)
            global soundThread
            if not soundThread.is_alive():
                soundThread = threading.Thread(target=playsound, args=('alertSound.mp3',), daemon=True)
                soundThread.start()
            
            cv.putText(self.image, f'Danger Exists', (0, 20), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 150))
        if showAllObjects:
            copyOfDetections = detectedObjects.copy()
            for name in list(copyOfDetections):
                for det in copyOfDetections[name]:
                    cv.rectangle(self.image, (det['box'][0], det['box'][1]), (det['box'][2], det['box'][3]), (255,255,255),2)
                    self.drawLabel(name, det)
                    
        if self.current and not self.disableDrawing:
            if self.current.name in detectedObjects.keys():
                for det in detectedObjects[self.current.name]:
                    cv.rectangle(self.image, (det['box'][0], det['box'][1]), (det['box'][2], det['box'][3]), (255,255,0),3)
                    self.drawLabel(self.current.name, det)
            # Loop over danger objects by name
            for name in self.current.dangerObjectsNames:
                if name in detectedObjects.keys():
                    for obj in detectedObjects[name]:
                        bbox = obj['box']
                        cv.rectangle(self.image, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,150), 2)
                        cv.putText(self.image, name, (bbox[0],bbox[1] - 10), cv.FONT_HERSHEY_PLAIN, 1, (0,0,150), 2)
            for sq in self.current.dangerZones:
                cv.rectangle(self.image, (sq[0], sq[1]), (sq[2], sq[3]), (55, 55, 255), 2)
            if self.isMouseDown:
                cv.rectangle(self.image, self.startPoint, self.endPoint, (200, 100, 100), 5)
            # self.current.checkDanger()
        qt_image = cvtoqt(self.image, self.size().width(), self.size().height())
        self.setPixmap(QPixmap.fromImage(qt_image))


    def mousePressEvent(self, event):
        self.isMouseDown = True
        self.startPoint = self.endPoint = cartesianPoint(event.pos())

    def mouseMoveEvent(self, event):
        self.endPoint = cartesianPoint(event.pos())
        if self.isMouseDown and not self.disableDrawing:
            cv.circle(self.image, cartesianPoint(event.pos()), 5, (200, 30, 255))
            self.updateImage()

    def mouseReleaseEvent(self, event):
        self.isMouseDown = False
        if self.current and not self.disableDrawing:
            self.endPoint = cartesianPoint(event.pos())
            x1, x2 = min(self.startPoint[0], self.endPoint[0]), max(self.startPoint[0], self.endPoint[0])
            y1, y2 = min(self.startPoint[1], self.endPoint[1]), max(self.startPoint[1], self.endPoint[1])
            self.updateImage()
            self.current.dangerZones.append((x1, y1, x2, y2))

    def setCurrent(self, name):
        if name == "None":
            self.disableDrawing = True
        else:
            self.current = watchedObjects[name]
            self.disableDrawing = False

    def drawLabel(self, name, obj):
        label = name + ": " + str(obj['confidence'])
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x1, y1, x2, y2 = obj['box']

        yLeftBottom = max(y1, labelSize[1])
        cv2.rectangle(self.image, (x1, yLeftBottom - labelSize[1]),
                                (x1 + labelSize[0], yLeftBottom + baseLine),
                                (255, 255, 255), cv2.FILLED)
        cv2.putText(self.image, label, (x1, yLeftBottom),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    def toggleDrawing(self, event):
        self.disableDrawing = not self.disableDrawing


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("The Watcher")
        self.img_width, self.img_height = 480, 270
        self.imgLabel = ImageLabel(self)
        self.imgLabel.resize(self.img_width, self.img_height)

        textLabel = QLabel("Demo")
        countLabel = QLabel("Counter")
        textLabel.count = 0

        vbox = QVBoxLayout()
        vbox.addWidget(self.imgLabel)
        vbox.addWidget(textLabel)
        vbox.addWidget(countLabel)

        hbox = QHBoxLayout()
        #accuracy slider.
        lblAcc = QLabel('Accuracy treshold: ')
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(1,9)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(1)
        slider.setSliderPosition(5)
        slider.valueChanged.connect(setAccuracy)
        hbox.addWidget(lblAcc)
        hbox.addWidget(slider)
        vbox.addLayout(hbox)

        def toggleShowAll(val):
            global showAllObjects
            showAllObjects = not showAllObjects
            
        chkBox = QCheckBox("Show all objects bounding boxes")
        chkBox.setChecked(True)
        chkBox.clicked.connect(toggleShowAll)
        vbox.addWidget(chkBox)
        

        container = QHBoxLayout()
        controller = QVBoxLayout()

        # a list that contain all the object that the model can detect.
        self.listWidget = QListWidget()
        self.listWidget.addItems(CLASSES[1:])
        self.listWidget.setCurrentRow(0)
        controller.addWidget(self.listWidget)
        btnWatch = QPushButton('Watch')
        controller.addWidget(btnWatch)
        #comboBox contains the object under WATCHING
        self.combo = QComboBox()
        btnWatch.clicked.connect(self.watchNewObject)
        self.combo.addItem('None')


        self.combo.activated[str].connect(self.imgLabel.setCurrent)
        self.imgLabel.messageSignal.connect(textLabel.setText)
        controller.addWidget(self.combo)
        staticLabel = QLabel('To add static zones:')
        sLabel = QLabel('Draw on the screen using the mouse')
        sLabel.setWordWrap(True)
        controller.addWidget(staticLabel)
        controller.addWidget(sLabel)
        dangerBtn = QCheckBox('Allow drawing')
        dangerBtn.setChecked(True)
        dangerBtn.clicked.connect(self.imgLabel.toggleDrawing)
        controller.addWidget(dangerBtn)

        dngObjLbl = QLabel('To add objects as danger zone:')
        dangerObjectBtn = QPushButton('Add Danger Object')
        dangerObjectBtn.clicked.connect(self.addNewDangerObject)

        controller.addWidget(dngObjLbl)
        controller.addWidget(dangerObjectBtn)
        controller.addStretch(1)
        controllerWidget = QWidget()
        controllerWidget.setLayout(controller)
        controllerWidget.setMinimumWidth(200)
        controllerWidget.setMaximumWidth(220)
        container.addWidget(controllerWidget)
        container.addLayout(vbox)
        self.setLayout(container)

        self.thread = VideoThread()
        self.thread.frameChange.connect(self.imgLabel.drawCVImage)
        self.thread.frameIncrease.connect(countLabel.setText)
        self.thread.start()
        self.initActions()
        self.show()

    def initActions(self):
        exitAct = QAction('Quit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.triggered.connect(qApp.quit)
        self.addAction(exitAct)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def watchNewObject(self):
        newObjectName = self.listWidget.currentItem().text()
        if  not newObjectName in [self.combo.itemText(i) for i in range(self.combo.count())]:
            self.combo.addItem(newObjectName)
            self.combo.setCurrentText(newObjectName)
            watchedObjects[newObjectName] = WatchedObject(newObjectName)
            self.imgLabel.setCurrent(newObjectName)

    def addNewDangerObject(self):
        currentObjectName = self.combo.currentText()
        if currentObjectName != "None":
            popup = dangerObjectPopup(currentObjectName, self)
            popup.show()

class dangerObjectPopup(QDialog):
    def __init__(self,name, parent):
        super().__init__(parent)
        self.name = name
        self.setWindowTitle('select the object you want to treat as a danger object')
        self.setGeometry(300,300,200,200)
        self.vbox = QVBoxLayout()
        self.list = QListWidget(self)
        self.classes = [i for i in CLASSES  if i != name]
        self.list.addItems(self.classes)
        self.btn = QPushButton('Add',self)
        self.btn.clicked.connect(self.addDangerObject)
        self.vbox.addWidget(self.list)
        self.vbox.addWidget(self.btn)
        self.setLayout(self.vbox)

    def addDangerObject(self, e):
        if watchedObjects[self.name] :
            newObj = self.list.currentItem().text()
            if newObj not in watchedObjects[self.name].dangerObjectsNames:
                watchedObjects[self.name].dangerObjectsNames.append(newObj)
                print(f'a new danger object {newObj} zone has been added to {self.name}')
                self.close()


def main():
    app = QApplication(sys.argv)
    a = App()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
