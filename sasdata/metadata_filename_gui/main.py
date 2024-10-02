from PySide6.QtWidgets import QWidget, QApplication

class MetadataFilenameDialog(QWidget):
    def __init__(filename: str):
        self.filename = filename


if __name__ == "__main__":
    app = QApplication([])

    filename = input('Input filename to test: ')
    widget = MetadataFilenameDialog(filename)
    widget.show()


    exit(app.exec())
