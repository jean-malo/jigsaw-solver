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
        try:
            path = QFileDialog.getSaveFileName(self, directory="./images")[0]
            self.original_image.save(path + 'og.jpg')
            self.solution_image.save(path + 'sol.jpg')
        except Exception as e:
            self.original_image.save('./images/' + 'shuffled.jpg')
            self.solution_image.save('./images/' + 'sol.jpg')

    # Initiates, solve and evaluate a jigsaw-puzzle when run is clicked
    def puzzle_pipeline(self):
        puzzle = self.init_puzzle()
        puzzle = self.solve_puzzzle(puzzle)
        return puzzle

    def init_puzzle(self):
        piece_w = self.pieceNumW.value()
        piece_h = self.pieceNumH.value()
        self.color_space = self.colorSpace.currentText()
        puzzle = Puzzle(self, logging.getLogger(), self.color_space, piece_w, piece_h, self.loadpath,
                        self.shuffle.isChecked(), self.displayInitial.isChecked(), self.displaySol.isChecked())
        return puzzle

    def solve_puzzzle(self, puzzle):
        pieces, og_images = puzzle.build_pieces()
        if puzzle.displayInit:
            puzzle.shuffled_og_image = puzzle.show_original_pieces(pieces)
        result_edge = puzzle.edges(pieces, 4)
        puzzle.results = puzzle.compatibility(result_edge)
        puzzle.treat_results(puzzle.results)
        graphres, graph_not_opt = puzzle.populate_graph(puzzle.results)
        res_arr_pos = puzzle.reconstruct(graphres, pieces, puzzle.results)
        coord_arr1 = puzzle.insert_failure(puzzle.clean_results(res_arr_pos), puzzle.results)
        coord_arr2 = puzzle.trim_coords(coord_arr1, puzzle.piece_w * puzzle.piece_h)
        puzzle.neighbor_comparison(pieces, og_images, coord_arr1)
        puzzle.direct_comparison(pieces, og_images, coord_arr1)
        if puzzle.displayEnd:
            puzzle.result_image = puzzle.show_end_pieces(pieces, puzzle.clean_results(coord_arr2), puzzle.displayEnd)
        return puzzle


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
