from PySide2.QtWidgets import QLayout


def clearLayout(layout: QLayout):
    for i in reversed(range(layout.count())):
        layout.itemAt(i).widget().deleteLater()
