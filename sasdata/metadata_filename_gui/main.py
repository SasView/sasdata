from PySide6.QtWidgets import QWidget, QApplication, QVBoxLayout, QLineEdit, QHBoxLayout, QLabel
from sys import argv

class MetadataFilenameDialog(QWidget):
    def __init__(self, filename: str):
        super().__init__()

        self.filename = filename

        self.filename_line_label = QLabel(f'Filename: <b>{filename}</b>')

        self.seperator_chars_label = QLabel('Seperators')
        self.separator_chars = QLineEdit()

        self.filename_separator_layout = QHBoxLayout()
        self.filename_separator_layout.addWidget(self.filename_line_label)
        self.filename_separator_layout.addWidget(self.seperator_chars_label)
        self.filename_separator_layout.addWidget(self.separator_chars)

        self.layout = QVBoxLayout(self)
        self.layout.addLayout(self.filename_separator_layout)


if __name__ == "__main__":
    app = QApplication([])
    if len(argv) < 2:
        filename = input('Input filename to test: ')
    else:
        filename = argv[1]
    widget = MetadataFilenameDialog(argv[1])
    widget.show()


    exit(app.exec())
