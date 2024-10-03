from PySide6.QtWidgets import QWidget, QApplication, QVBoxLayout, QLineEdit, QHBoxLayout, QLabel
from sys import argv
import re

def build_font(text: str, classname: str = '') -> str:
    match classname:
        case 'token':
            return f"<font color='red'>{text}</font>"
        case 'separator':
            return f"<font color='grey'>{text}</font>"
        case _:
            return text
    return f'<span class="{classname}">{text}</span>'

class MetadataFilenameDialog(QWidget):
    def __init__(self, filename: str):
        super().__init__()

        self.filename = filename

        self.filename_line_label = QLabel()
        self.seperator_chars_label = QLabel('Seperators')
        self.separator_chars = QLineEdit()
        self.separator_chars.textChanged.connect(self.update_filename_separation)

        # Have to update this now because it relies on the value of the separator.
        self.update_filename_separation()

        self.filename_separator_layout = QHBoxLayout()
        self.filename_separator_layout.addWidget(self.filename_line_label)
        self.filename_separator_layout.addWidget(self.seperator_chars_label)
        self.filename_separator_layout.addWidget(self.separator_chars)

        self.layout = QVBoxLayout(self)
        self.layout.addLayout(self.filename_separator_layout)

    def formatted_filename(self) -> str:
        sep_str = self.separator_chars.text()
        if sep_str == '':
            return f'<span>{filename}</span>'
        # Won't escape characters; I'll handle that later.
        separated = re.split(f'([{sep_str}])', self.filename)
        font_elements = ''
        for i, token in enumerate(separated):
            classname = 'token' if i % 2 == 0 else 'separator'
            font_elements += build_font(token, classname)
        return font_elements

    def update_filename_separation(self):
        self.filename_line_label.setText(f'Filename: {self.formatted_filename()}')



if __name__ == "__main__":
    app = QApplication([])
    if len(argv) < 2:
        filename = input('Input filename to test: ')
    else:
        filename = argv[1]
    widget = MetadataFilenameDialog(filename)
    widget.show()


    exit(app.exec())
