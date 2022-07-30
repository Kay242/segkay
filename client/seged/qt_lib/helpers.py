import numpy as np
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QPixmap, QImage


def QImage_from_np(img):
    if img.dtype != np.uint8:
        raise ValueError("img should be in np.uint8 format")

    h, w, c = img.shape
    if c == 1:
        fmt = QImage.Format_Grayscale8
    elif c == 3:
        fmt = QImage.Format_BGR888
    elif c == 4:
        fmt = QImage.Format_ARGB32
    else:
        raise ValueError("unsupported channel count")

    return QImage(img.data, w, h, c * w, fmt)


def QImage_to_np(q_img, fmt=QImage.Format_BGR888):
    q_img = q_img.convertToFormat(fmt)

    width = q_img.width()
    height = q_img.height()

    b = q_img.constBits()
    b.setsize(height * width * 3)
    arr = np.frombuffer(b, np.uint8).reshape((height, width, 3))
    return arr  # [::-1]


def QPixmap_from_np(img):
    return QPixmap.fromImage(QImage_from_np(img))


def QPoint_from_np(n):
    return QPoint(*n.astype(np.int))


def QPoint_to_np(q):
    return np.int32([q.x(), q.y()])


def QSize_to_np(q):
    return np.int32([q.width(), q.height()])
