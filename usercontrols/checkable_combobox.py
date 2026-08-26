"""
Checkable combo box
===================

Simple `QComboBox` variant whose items are checkable. Useful for quick
multi-select inputs when a full tree is not required.

Public API
----------
- `CheckableComboBox` — add items via `addItem(s)` as usual; clicking toggles check state.

Notes
-----
- The model is a `QStandardItemModel` with checkable items. You can access
  it via `model()` for advanced customization.
"""

from qtpy.QtWidgets import * 
from qtpy.QtGui import QStandardItemModel
from qtpy.QtCore import Qt

class CheckableComboBox(QComboBox):
    """Combo box with checkable items.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    width : int, default 140
        Minimum width in pixels.
    """

    # constructor
    def __init__(self, parent=None, width=140):
        super(CheckableComboBox, self).__init__(parent)
        self.setModel(QStandardItemModel(self))
        self.count = 0
        self.setMinimumWidth(width) # pixels

    # action called when item get checked
    def do_action(self):
        """Hook called after an item becomes checked.

        Notes
        -----
        Override to react to item checks (e.g., update UI). Default prints
        a debug message.
        """
        print("Checked number : " +str(self.count))

    # when any item get pressed
    def handleItemPressed(self, index):
        """Toggle item check state when pressed.

        Parameters
        ----------
        index : QModelIndex
            Index of the pressed item in the combo's model.
        """
        # getting the item
        item = self.model().itemFromIndex(index)

        # checking if item is checked
        if item.checkState() == Qt.Checked:

            # making it unchecked
            item.setCheckState(Qt.Unchecked)

        # if not checked
        else:
            # making the item checked
            item.setCheckState(Qt.Checked)

            self.count += 1

            # call the action
            self.do_action()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = QWidget()
    layout = QVBoxLayout(window)
    combo = CheckableComboBox(window)
    combo.addItem("Item 1")
    combo.addItem("Item 2")
    combo.addItem("Item 3")
    combo.addItem("Item 4")
    layout.addWidget(combo)
    window.show()
    sys.exit(app.exec_())
