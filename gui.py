import logging

from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
from PyQt5.uic import loadUi

# from splitter import Puzzle
from puzzle import Puzzle


class MyMainWindow(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        loadUi('guiv2.ui', self)
        self.loadImage.clicked.connect(self.load_clicked)
        self.run.clicked.connect(self.run_init)
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

    # Initiates a puzzle when run is clicked
    def puzzle_init(self):
        logging.getLogger().setLevel(self.set_log_level())
        piece_w = self.pieceNumW.value()
        piece_h = self.pieceNumH.value()
        self.color_space = self.colorSpace.currentText()
        solver = Puzzle(self, logging.getLogger(), self.color_space, piece_w, piece_h, self.loadpath,
                        self.shuffle.isChecked(), self.displayInitial.isChecked(), self.displaySol.isChecked())
        pieces, og_images = solver.build_pieces()
        if solver.displayInit:
            solver.shuffled_og_image = solver.show_original_pieces(pieces)
        result_edge = solver.edges(pieces, 4)
        solver.results = solver.compatibility(result_edge)
        solver.treat_results(solver.results)
        graphres, graph_not_opt = solver.populate_graph(solver.results)
        res_arr_pos = solver.reconstruct(graphres, pieces, solver.results)
        coord_arr1 = solver.insert_failure(solver.clean_results(res_arr_pos), solver.results)
        coord_arr2 = solver.trim_coords(coord_arr1, solver.piece_w * solver.piece_h)
        solver.neighbor_comparison(pieces, og_images, coord_arr1)
        solver.direct_comparison(pieces, og_images, coord_arr1)
        if solver.displayEnd:
            solver.result_image = solver.show_end_pieces(pieces, solver.clean_results(coord_arr2), solver.displayEnd)


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
