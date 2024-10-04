from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem
from sasdata.metadata_filename_gui.metadata_component_selector import MetadataComponentSelector

class MetadataTreeWidget(QTreeWidget):
    def __init__(self):
        super().__init__()
        self.setColumnCount(2)
        self.setHeaderLabels(['Name', 'Filename Components'])

        # TODO: This is placeholder data that'll need to be replaced by the real metadata.

    def draw_tree(self):
        self.clear()
        metadata = {'Instrument': ['Slit width', 'Other']}
        for top_level, items in metadata.items():
            top_level_item = QTreeWidgetItem([top_level])
            for metadatum in items:
                selector = MetadataComponentSelector()
                metadatum_item = QTreeWidgetItem([metadatum, selector])
                top_level_item.addChild(metadatum_item)
            self.insertTopLevelItem(top_level_item)
