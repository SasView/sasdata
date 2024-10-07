from PySide6.QtWidgets import QWidget, QPushButton, QHBoxLayout

class MetadataComponentSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.options: list[str]
        self.option_buttons: list[QPushButton]
        self.layout = QHBoxLayout(self)

    def clear_options(self):
        for i in reversed(range(self.layout.count() - 1)):
            self.layout.takeAt(i).widget().deleteLater()

    def draw_options(self, new_options: list[str]):
        self.clear_options()
        self.options = new_options
        self.option_buttons = []
        for option in self.options:
            option_button = QPushButton(option)
            option_button.setCheckable(True)
            self.layout.addWidget(option_button)
            self.option_buttons.append(option_button)
