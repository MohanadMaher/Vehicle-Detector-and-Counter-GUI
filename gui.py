import sys
from PyQt5.uic import loadUi
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from vehicle_count import detectOnImage

class WelcomeScreen(QDialog):
    def __init__(self):
        super(WelcomeScreen, self).__init__()
        loadUi("gui.ui", self)
        self.uploadButton.clicked.connect(self.loadImage)
        self.cancelButton.clicked.connect(self.cancel)

    def loadImage(self, filename):
        if not filename:
            filename, _ = QFileDialog.getOpenFileName(self, 'Select Photo', QDir.currentPath(), 'Images (*.png *.jpg)')
            if not filename:
                return
        detectOnImage(filename)
        print("hello")
    def cancel(self):
        widget.close()


app = QApplication(sys.argv)
welcome = WelcomeScreen()
widget = QStackedWidget()
widget.setWindowTitle("Vehicle Detector and Counter")
widget.setWindowIcon(QIcon('icon.jpg'))
widget.addWidget(welcome)
widget.setFixedWidth(657)
widget.setFixedHeight(535)
widget.show()
try:
    sys.exit(app.exec_())
except:
    print("Exiting")

