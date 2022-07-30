from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor


class QDarkPalette(QPalette):
    def __init__(self):
        super().__init__()
        text_color = QColor(200, 200, 200)
        self.setColor(QPalette.Window, QColor(53, 53, 53))
        self.setColor(QPalette.WindowText, text_color)
        self.setColor(QPalette.Base, QColor(25, 25, 25))
        self.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        self.setColor(QPalette.ToolTipBase, text_color)
        self.setColor(QPalette.ToolTipText, text_color)
        self.setColor(QPalette.Text, text_color)
        self.setColor(QPalette.Button, QColor(53, 53, 53))
        self.setColor(QPalette.ButtonText, Qt.white)
        self.setColor(QPalette.BrightText, Qt.red)
        self.setColor(QPalette.Link, QColor(42, 130, 218))
        self.setColor(QPalette.Highlight, QColor(42, 130, 218))
        self.setColor(QPalette.HighlightedText, Qt.black)

