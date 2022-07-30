from PyQt5.QtGui import QCursor, QPixmap


class QCursorDB:
    cross_red = None
    cross_green = None
    cross_blue = None

    @staticmethod
    def initialize(cursor_path):
        QCursorDB.cross_red = QCursor(QPixmap(str(cursor_path / 'cross_red.png')))
        QCursorDB.cross_green = QCursor(QPixmap(str(cursor_path / 'cross_green.png')))
        QCursorDB.cross_blue = QCursor(QPixmap(str(cursor_path / 'cross_blue.png')))
