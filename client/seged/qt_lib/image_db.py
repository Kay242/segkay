from PyQt5.QtGui import QImage


class QImageDB:
    intro = None

    @staticmethod
    def initialize(image_path):
        QImageDB.intro = QImage(str(image_path / 'intro.png'))
