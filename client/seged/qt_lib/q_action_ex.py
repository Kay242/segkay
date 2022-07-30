from PyQt5.QtWidgets import QAction
from client.seged.localization import StringsDB


class QActionEx(QAction):
    def __init__(self, icon, text, shortcut=None, trigger_func=None, shortcut_in_tooltip=False, is_checkable=False,
                 is_auto_repeat=False):
        super().__init__(icon, text)
        if shortcut is not None:
            self.setShortcut(shortcut)
            if shortcut_in_tooltip:
                self.setToolTip(f"{text} ( {StringsDB['S_HOT_KEY']}: {shortcut} )")

        if trigger_func is not None:
            self.triggered.connect(trigger_func)
        if is_checkable:
            self.setCheckable(True)
        self.setAutoRepeat(is_auto_repeat)