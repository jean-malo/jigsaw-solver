import logging

from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
from PyQt5.uic import loadUi

# from splitter import Puzzle
from puzzle import Puzzle


class MyMainWindow(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        loadUi('guiv2.ui', self)
        logging.getLogger().setLevel(self.set_log_level())
        self.puzzle = None
        self.loadImage.clicked.connect(self.load_clicked)
        self.run.clicked.connect(self.puzzle_pipeline)
        self.saveSolution.clicked.connect(self.save_clicked)
        logTextBox = QPlainTextEditLogger(self)
        logTextBox.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(logTextBox)

    def set_log_level(self):
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

    def load_clicked(self):
        self.loadpath = QFileDialog.getOpenFileName(self, directory=f"./images", filter=('*.jpg'))[0]
        self.run.setEnabled(True)

    def save_clicked(self):
        formats = "Portable Network Graphic (*.jpg);;"
        try:
            path = QFileDialog.getSaveFileName(self, directory="./images")[0]
            self.puzzle.og_image.save(path + "_shuffled.jpg")
            self.puzzle.result_image.save(path + "_solved.jpg")
        except Exception as e:
            print(e)
            self.puzzle.og_image.save('./images/' + '_shuffled.jpg')
            self.puzzle.result_image.save('./images/' + '_solved.jpg')

    # Initiates, solve and evaluate a jigsaw-puzzle when run is clicked
    def puzzle_pipeline(self):
        self.saveSolution.setEnabled(False)
        self.puzzle = self.init_puzzle()
        self.solve_puzzzle()
        self.saveSolution.setEnabled(True)

    def init_puzzle(self):
        piece_w = self.pieceNumW.value()
        piece_h = self.pieceNumH.value()
        print(piece_w, piece_h)
        self.color_space = self.colorSpace.currentText()
        puzzle = Puzzle(self, logging.getLogger(), self.color_space, piece_w, piece_h, self.loadpath,
                        self.shuffle.isChecked(), self.displayInitial.isChecked(), self.displaySol.isChecked())
        return puzzle

    def solve_puzzzle(self):
        pieces, og_images = self.puzzle.build_pieces()
        if self.puzzle.displayInit:
            self.puzzle.og_image.show()
        result_edge = self.puzzle.edges(pieces, 4)
        self.puzzle.results = self.puzzle.compatibility(result_edge)
        self.puzzle.treat_results(self.puzzle.results)
        graphres, graph_not_opt = self.puzzle.populate_graph(self.puzzle.results)
        res_arr_pos = self.puzzle.reconstruct(graphres, pieces, self.puzzle.results)
        coord_arr1 = self.puzzle.insert_failure(self.puzzle.clean_results(res_arr_pos), self.puzzle.results)
        coord_arr2 = self.puzzle.trim_coords(coord_arr1, self.puzzle.piece_w * self.puzzle.piece_h)
        self.puzzle.neighbor_comparison(pieces, og_images, coord_arr1)
        self.puzzle.direct_comparison(pieces, og_images, coord_arr1)
        self.puzzle.build_end_image(pieces, self.puzzle.clean_results(coord_arr2))
        if self.puzzle.displayEnd:
            self.puzzle.result_image.show()


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
