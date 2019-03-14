import sys

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication,QDialog, QMainWindow, QFileDialog, QSpinBox, QStylePainter,QWidget
from PyQt5.uic import loadUi
from PyQt5 import QtGui, QtCore
import numpy as np
from PIL import Image
import logging
from splitter import Solver

class MyMainWindow(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        loadUi('guiv2.ui', self)
        self.loadImage.clicked.connect(self.loadClicked)
        self.run.clicked.connect(self.runInit)
        self.saveSolution.clicked.connect(self.saveClicked)
        logTextBox = QPlainTextEditLogger(self)
        logTextBox.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(logTextBox)

    #sets the level of the logger
    def handle_log_level(self):
        if str(self.logLevel.currentText()) == 'Debug':
            return 10
        if str(self.logLevel.currentText()) == 'Info':
            return 20
        if str(self.logLevel.currentText()) == 'Warning':
            return 30
        if str(self.logLevel.currentText()) == 'Error':
            return 40
        if str(self.logLevel.currentText()) == 'Critical':
            return 50

    def loadClicked(self):
        self.loadpath = QFileDialog.getOpenFileName(self,directory="/Users/jeanmalod/PycharmProjects/fyp-jigsaw/images",filter=('*.jpg'))[0]
        self.run.setEnabled(True)

    def saveClicked(self):
        self.savepath = QFileDialog.getSaveFileName(self, directory="/Users/jeanmalod/PycharmProjects/fyp-jigsaw/images")[0]
        try:
            self.original_image.save(self.savepath + 'shuffled.jpg')
            self.solution_image.save(self.savepath + 'sol.jpg')
        except:
            self.original_image.save('default'+ 'shuffled.jpg')
            self.solution_image.save('default' + 'sol.jpg')


    #Initiates a solver when run is clicked
    def runInit(self):
        logging.getLogger().setLevel(self.handle_log_level())
        piece_w = self.pieceNumW.value()
        piece_h =   self.pieceNumH.value()
        self.color_space = self.colorSpace.currentText()
        solver = Solver(self,logging.getLogger(),self.color_space, piece_w,piece_h,self.loadpath,self.shuffle.isChecked(), self.displayInitial.isChecked(),self.displaySol.isChecked(), False)
        self.solution_image = solver.result_image
        self.original_image  =solver.shuffled_og_image

class QPlainTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = parent.logger
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)
        QApplication.processEvents()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    mainapplication = MyMainWindow()
    mainapplication.show()
    app.exec_()

