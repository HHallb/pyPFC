from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QDoubleValidator
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QSizePolicy,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure
from functools import partial
import sys
import numpy as np
from typing import Any

import pypfc


# Main window subclass
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.sim: pypfc.setup_simulation = pypfc.setup_simulation(domain_size=[1.0, 1.0, 1.0], ndiv=[2, 2, 2], device_type="cpu")
        self.selected_file_path: str | None = None
        self.loaded_data: dict[str, Any] | None = None
        self.domain_size: np.ndarray | None = None
        self.ndiv: np.ndarray | None = None
        self.ddiv: np.ndarray | None = None
        self.atom_coords: np.ndarray | None = None
        self.atom_data: np.ndarray | None = None
        self.atom_data_labels: list[str] | None = None
        self._colorbar: Any | None = None
        self._selected_atom_data_index: int | None = None
        self._selected_atom_data_label: str | None = None
        self._active_plot_kind: str | None = None
        self._selected_field_index: int | None = None
        self._selected_field_label: str | None = None
        self._current_view: tuple[int, int, str, str] = (0, 1, "x", "y")
        self._initial_view_limits: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._coordinate_labels = ("x", "y")
        self._marker_size = 4.0
        self._left_vertical_sizes_initialized = False
        self.fields: np.ndarray | None = None
        self.field_labels: list[str] | None = None
        self.den: np.ndarray | None = None
        self.ene: np.ndarray | None = None
        self.dtime: float | None = None
        self._pan_start: tuple[float, float, tuple[float, float], tuple[float, float]] | None = None

        self.setWindowTitle("pyPFC GUI")
        self.setMinimumSize(QSize(800, 600))

        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Data", "Value"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tree.setStyleSheet(
            "QTreeWidget { background-color: white; color: black; }"
            "QHeaderView::section { color: black; background-color: white; }"
        )
        self.left_splitter = QSplitter(Qt.Orientation.Vertical)
        self.left_splitter.addWidget(self.tree)

        self.plot_settings_panel = QWidget()
        self.plot_settings_panel.setStyleSheet(
            "QWidget { background-color: white; color: black; }"
        )
        settings_layout = QVBoxLayout(self.plot_settings_panel)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        settings_layout.setSpacing(6)
        settings_heading = QLabel("Plot settings")
        settings_heading.setStyleSheet("font-weight: bold;")
        settings_layout.addWidget(settings_heading)

        settings_form = QFormLayout()
        settings_layout.addLayout(settings_form)
        checkbox_row = QHBoxLayout()
        checkbox_row.setSpacing(16)
        self.show_axes_checkbox = QCheckBox("Show axes")
        self.show_axes_checkbox.setChecked(True)
        self.show_axes_checkbox.toggled.connect(self._show_axes_toggled)
        checkbox_row.addWidget(self.show_axes_checkbox)

        self.show_legend_checkbox = QCheckBox("Show legend")
        self.show_legend_checkbox.setChecked(True)
        self.show_legend_checkbox.toggled.connect(self._show_legend_toggled)
        checkbox_row.addWidget(self.show_legend_checkbox)
        checkbox_row.addStretch(1)
        settings_form.addRow(checkbox_row)

        self.marker_size_edit = QLineEdit(str(self._marker_size))
        self.marker_size_edit.setValidator(QDoubleValidator(0.01, 1000.0, 2, self))
        self.marker_size_edit.setFixedWidth(80)
        self.marker_size_edit.editingFinished.connect(self._marker_size_changed)
        settings_form.addRow("Marker size", self.marker_size_edit)

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(
            [
                "viridis",
                "plasma",
                "inferno",
                "magma",
                "cividis",
                "coolwarm",
                "RdBu",
                "jet",
            ]
        )
        longest_colormap = max(
            (self.colormap_combo.itemText(index) for index in range(self.colormap_combo.count())),
            key=len,
        )
        colormap_width = (
            self.colormap_combo.fontMetrics().horizontalAdvance(longest_colormap)
            + self.colormap_combo.sizeHint().width()
            - self.colormap_combo.fontMetrics().horizontalAdvance(
                self.colormap_combo.currentText()
            )
        )
        self.colormap_combo.setFixedWidth(colormap_width)
        self.colormap_combo.currentTextChanged.connect(self._colormap_changed)
        settings_form.addRow("Colormap", self.colormap_combo)

        settings_layout.addStretch(1)
        self.plot_settings_panel.hide()
        self.left_splitter.addWidget(self.plot_settings_panel)
        self.left_splitter.setStretchFactor(0, 2)
        self.left_splitter.setStretchFactor(1, 1)
        self.splitter.addWidget(self.left_splitter)

        self.plot_panel = QWidget()
        self.plot_panel.setStyleSheet("background-color: white;")
        self.plot_layout = QVBoxLayout(self.plot_panel)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.view_controls = QWidget()
        self.view_controls.setFixedSize(168, 26)
        self.view_controls_layout = QHBoxLayout(self.view_controls)
        self.view_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.view_controls_layout.setSpacing(2)
        self.view_buttons: dict[str, QPushButton] = {}
        for label in ("O", "xy", "xz", "yz"):
            button = QPushButton(label)
            button.setFixedSize(36 if label == "O" else 42, 26)
            button.setStyleSheet(
                "QPushButton { color: #111; background-color: #f0f0f0; "
                "border: 1px solid #888; padding: 2px; }"
                "QPushButton:disabled { color: #777; }"
            )
            button.clicked.connect(partial(self._view_button_clicked, label))
            self.view_controls_layout.addWidget(button)
            self.view_buttons[label] = button
        self.view_controls.hide()
        self.figure = Figure(facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.axes = self.figure.add_subplot(111)
        self.axes.set_axis_off()
        self.plot_layout.addWidget(self.canvas)
        self.splitter.addWidget(self.plot_panel)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)
        self.tree.itemSelectionChanged.connect(self._tree_item_selected)
        self.canvas.mpl_connect("motion_notify_event", self._plot_mouse_moved)
        self.canvas.mpl_connect("scroll_event", self._plot_mouse_wheel)
        self.canvas.mpl_connect("button_press_event", self._plot_mouse_pressed)
        self.canvas.mpl_connect("button_release_event", self._plot_mouse_released)
        self.canvas.mpl_connect("motion_notify_event", self._plot_mouse_dragged)

        # Add main toolbar
        toolbar_main = QToolBar("My main toolbar")
        toolbar_main.setIconSize(QSize(16, 16))
        toolbar_main.setMovable(False)
        toolbar_main.setFloatable(False)

        # Add buttons to the main toolbar
        button_open_file_action = QAction("Open", self)
        button_open_file_action.setStatusTip("Open *.h5 file")
        button_open_file_action.triggered.connect(self.toolbar_main_button_open_file_clicked)
        toolbar_main.addAction(button_open_file_action)

        toolbar_row = QWidget()
        toolbar_row.setFixedHeight(32)
        toolbar_row_layout = QHBoxLayout(toolbar_row)
        toolbar_row_layout.setContentsMargins(0, 0, 4, 0)
        toolbar_row_layout.setSpacing(0)
        toolbar_main.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        toolbar_row_layout.addWidget(toolbar_main)
        toolbar_row_layout.addStretch(1)
        toolbar_row_layout.addWidget(self.view_controls)

        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(toolbar_row)
        central_layout.addWidget(self.splitter)
        self.setCentralWidget(central_widget)
        QTimer.singleShot(0, self._set_initial_panel_sizes)

        # Add status bar
        self.setStatusBar(QStatusBar(self))

            
    def toolbar_main_button_open_file_clicked(self, _: bool = False) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open pyPFC HDF5 data file",
            "",
            "HDF5 files (*.h5)",
        )
        if not file_path:
            return

        try:
            loaded_data = self.sim.load_hdf5(file_path)
        except (ImportError, OSError, ValueError) as error:
            QMessageBox.critical(
                self,
                "Unable to open HDF5 file",
                f"Could not read '{file_path}':\n{error}",
            )
            self.statusBar().showMessage(f"Failed to open: {file_path}")
            return

        grid_data = loaded_data.get("grid", {})
        atom_data = loaded_data.get("atoms", {})
        field_group = loaded_data.get("fields", {})
        simulation_data = loaded_data.get("simulation", {})

        self.domain_size = grid_data.get("domain_size")
        self.ndiv = grid_data.get("ndiv")
        self.ddiv = grid_data.get("ddiv")
        self.atom_coords = atom_data.get("coords")
        self.atom_data = atom_data.get("atom_data")
        self.atom_data_labels = atom_data.get("atom_data_labels")
        self.fields = field_group.get("field_data")
        self.field_labels = field_group.get("field_labels")
        self.dtime = simulation_data.get("dtime")

        fields_by_label = {}
        if self.fields is not None and self.field_labels is not None:
            fields_by_label = dict(zip(self.field_labels, self.fields, strict=True))
        self.den = fields_by_label.get("density")
        self.ene = fields_by_label.get("energy")

        self.selected_file_path = file_path
        self.loaded_data = loaded_data
        self._populate_hdf5_tree(loaded_data)
        self._clear_plot()
        self.statusBar().showMessage(f"Loaded: {file_path}")

    def _populate_hdf5_tree(self, data: dict[str, Any]) -> None:
        self.tree.clear()
        for name, value in data.items():
            if not self._has_content(value):
                continue
            display_name = "Root attributes" if name == "root_attrs" else name
            group_item = QTreeWidgetItem([display_name, ""])
            group_item.setData(0, Qt.ItemDataRole.UserRole, (name,))
            self.tree.addTopLevelItem(group_item)
            if isinstance(value, dict):
                self._add_mapping_items(group_item, value)
            else:
                group_item.setText(1, self._describe_dataset(value))
        self.tree.expandToDepth(1)

    def _add_mapping_items(
        self,
        parent: QTreeWidgetItem,
        mapping: dict[str, Any],
    ) -> None:
        if not mapping:
            parent.setText(1, "(empty)")
            return

        for name, value in mapping.items():
            if not self._has_content(value):
                continue
            item = QTreeWidgetItem([name, ""])
            parent_path = parent.data(0, Qt.ItemDataRole.UserRole) or ()
            item.setData(0, Qt.ItemDataRole.UserRole, (*parent_path, name))
            parent.addChild(item)
            if parent_path == ("atoms",) and name == "atom_data_labels":
                self._add_atom_data_label_items(item, value)
            elif parent_path == ("fields",) and name == "field_labels":
                self._add_field_label_items(item, value)
            elif isinstance(value, dict):
                self._add_mapping_items(item, value)
            else:
                item.setText(1, self._describe_dataset(value))

    def _add_atom_data_label_items(
        self,
        parent: QTreeWidgetItem,
        labels: Any,
    ) -> None:
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        if not isinstance(labels, (list, tuple)):
            return

        parent_path = parent.data(0, Qt.ItemDataRole.UserRole) or ()
        for index, label in enumerate(labels):
            if not isinstance(label, str):
                continue
            item = QTreeWidgetItem([label, f"column {index}"])
            item.setData(
                0,
                Qt.ItemDataRole.UserRole,
                (*parent_path, label),
            )
            item.setData(0, Qt.ItemDataRole.UserRole + 1, index)
            parent.addChild(item)
        parent.setExpanded(True)

    def _add_field_label_items(self, parent: QTreeWidgetItem, labels: Any) -> None:
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        if not isinstance(labels, (list, tuple)):
            return

        parent_path = parent.data(0, Qt.ItemDataRole.UserRole) or ()
        for index, label in enumerate(labels):
            if not isinstance(label, str):
                continue
            item = QTreeWidgetItem([label, f"field {index}"])
            item.setData(0, Qt.ItemDataRole.UserRole, (*parent_path, label))
            item.setData(0, Qt.ItemDataRole.UserRole + 1, index)
            parent.addChild(item)
        parent.setExpanded(True)

    @staticmethod
    def _has_content(value: Any) -> bool:
        if isinstance(value, dict):
            return any(MainWindow._has_content(item) for item in value.values())
        if isinstance(value, np.ndarray):
            return value.size > 0
        if isinstance(value, (list, tuple, str, bytes)):
            return len(value) > 0
        return True

    def _tree_item_selected(self) -> None:
        item = self.tree.currentItem()
        if item is None:
            self._clear_plot()
            return
        item_path = item.data(0, Qt.ItemDataRole.UserRole)
        if item_path in (("atoms",), ("atoms", "coords"), ("atoms", "coord")):
            self._plot_atoms()
        elif (
            isinstance(item_path, tuple)
            and len(item_path) == 3
            and item_path[:2] == ("atoms", "atom_data_labels")
        ):
            atom_data_index = int(item.data(0, Qt.ItemDataRole.UserRole + 1))
            atom_data_label = str(item_path[2])
            self._plot_atoms(atom_data_index, atom_data_label)
        elif (
            isinstance(item_path, tuple)
            and len(item_path) == 3
            and item_path[:2] == ("fields", "field_labels")
        ):
            field_index = int(item.data(0, Qt.ItemDataRole.UserRole + 1))
            field_label = str(item_path[2])
            self._plot_field(field_index, field_label)
        else:
            self._clear_plot()
            self.statusBar().showMessage(
                f"Loaded: {self.selected_file_path}" if self.selected_file_path else ""
            )

    def _plot_atoms(
        self,
        atom_data_index: int | None = None,
        atom_data_label: str | None = None,
        view: tuple[int, int, str, str] = (0, 1, "x", "y"),
        preserve_limits: bool = False,
    ) -> None:
        self._active_plot_kind = "atoms"
        if self.atom_coords is None:
            self._clear_plot()
            self.statusBar().showMessage("This file does not contain /atoms/coords")
            return
        if self.atom_coords.ndim != 2 or self.atom_coords.shape[1] < 2:
            QMessageBox.warning(
                self,
                "Invalid atom coordinates",
                "The /atoms/coords dataset must have at least two columns for an XY plot.",
            )
            return

        x_index, y_index, x_label, y_label = view
        if self.atom_coords.shape[1] <= max(x_index, y_index):
            self._clear_plot()
            QMessageBox.warning(
                self,
                "Unavailable coordinate plane",
                f"The /atoms/coords dataset must have at least "
                f"{max(x_index, y_index) + 1} columns to show the {x_label}{y_label} plane.",
            )
            return

        color_values = None
        if atom_data_index is not None:
            if (
                self.atom_data is None
                or self.atom_data.ndim != 2
                or self.atom_data.shape[0] != self.atom_coords.shape[0]
                or atom_data_index >= self.atom_data.shape[1]
            ):
                self._clear_plot()
                QMessageBox.warning(
                    self,
                    "Invalid atom data",
                    f"The /atoms/atom_data dataset has no valid column for '{atom_data_label}'.",
                )
                return
            if not np.issubdtype(self.atom_data.dtype, np.number):
                self._clear_plot()
                QMessageBox.warning(
                    self,
                    "Invalid atom data",
                    f"The /atoms/atom_data column '{atom_data_label}' must be numeric.",
                )
                return
            color_values = np.asarray(self.atom_data[:, atom_data_index], dtype=float)
            if not np.isfinite(color_values).any():
                self._clear_plot()
                QMessageBox.warning(
                    self,
                    "Invalid atom data",
                    f"The /atoms/atom_data column '{atom_data_label}' contains no finite values.",
                )
                return

        previous_limits = (
                (self.axes.get_xlim(), self.axes.get_ylim())
                if preserve_limits
                else None
        )
        self._remove_colorbar()
        self.axes.clear()
        self.figure.subplots_adjust(left=0.08, right=0.985, bottom=0.10, top=0.97)
        if color_values is None:
            self.axes.scatter(
                self.atom_coords[:, x_index],
                self.atom_coords[:, y_index],
                color="gray",
                s=self._marker_size,
            )
        else:
            scatter = self.axes.scatter(
                self.atom_coords[:, x_index],
                self.atom_coords[:, y_index],
                c=color_values,
                cmap=self.colormap_combo.currentText(),
                s=self._marker_size,
            )
            if self.show_legend_checkbox.isChecked():
                self._colorbar = self.figure.colorbar(
                    scatter,
                    ax=self.axes,
                    pad=0.015,
                    fraction=0.045,
                    label=atom_data_label,
                )
        if self.show_axes_checkbox.isChecked():
            self.axes.set_axis_on()
        else:
            self.axes.set_axis_off()
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.margins(0.01)
        self.axes.autoscale(tight=True)
        if previous_limits is not None:
            self.axes.set_xlim(previous_limits[0])
            self.axes.set_ylim(previous_limits[1])
        self._selected_atom_data_index = atom_data_index
        self._selected_atom_data_label = atom_data_label
        self._current_view = view
        self._coordinate_labels = (x_label, y_label)
        self.plot_settings_panel.show()
        if not self._left_vertical_sizes_initialized:
            QTimer.singleShot(0, self._set_initial_left_panel_sizes)
        self.view_buttons["xz"].setEnabled(self.atom_coords.shape[1] >= 3)
        self.view_buttons["yz"].setEnabled(self.atom_coords.shape[1] >= 3)
        self.view_controls.show()
        self.canvas.draw()
        if not preserve_limits:
            self._initial_view_limits = (self.axes.get_xlim(), self.axes.get_ylim())
        if atom_data_label is None:
            self.statusBar().showMessage("Atom positions in the XY plane")
        else:
            self.statusBar().showMessage(
                f"Atom positions colored by {atom_data_label}"
            )

    def _plot_field(
        self,
        field_index: int,
        field_label: str,
        view: tuple[int, int, str, str] = (0, 1, "x", "y"),
        preserve_limits: bool = False,
    ) -> None:
        self._active_plot_kind = "field"
        if self.fields is None or self.fields.ndim < 3 or field_index >= self.fields.shape[0]:
            self._clear_plot()
            self.statusBar().showMessage(f"Field data for '{field_label}' is unavailable")
            return

        field = np.asarray(self.fields[field_index])
        if not np.issubdtype(field.dtype, np.number) or field.ndim not in (2, 3):
            self._clear_plot()
            QMessageBox.warning(
                self,
                "Invalid field data",
                f"Field '{field_label}' must be a 2D or 3D numeric array.",
            )
            return

        x_index, y_index, x_label, y_label = view
        if field.ndim == 2:
            if view != (0, 1, "x", "y"):
                self.statusBar().showMessage("2D fields are available in the XY plane only")
                return
            plane = field
            x_index, y_index = 0, 1
        elif view == (0, 1, "x", "y"):
            plane = field[:, :, field.shape[2] // 2]
        elif view == (0, 2, "x", "z"):
            plane = field[:, field.shape[1] // 2, :]
        else:
            plane = field[field.shape[0] // 2, :, :]

        if min(plane.shape) < 2:
            self._clear_plot()
            QMessageBox.warning(
                self,
                "Invalid field dimensions",
                f"Field '{field_label}' needs at least two grid points along both "
                "plotted axes for a contour plot.",
            )
            return
        if not np.isfinite(plane).any():
            self._clear_plot()
            QMessageBox.warning(
                self,
                "Invalid field values",
                f"Field '{field_label}' contains no finite values in the selected slice.",
            )
            return

        previous_limits = (
            (self.axes.get_xlim(), self.axes.get_ylim())
            if preserve_limits
            else None
        )
        spacings = np.asarray(self.ddiv, dtype=float).reshape(-1) if self.ddiv is not None else np.ones(3)
        x_values = np.arange(plane.shape[0]) * (
            spacings[x_index] if x_index < len(spacings) else 1.0
        )
        y_values = np.arange(plane.shape[1]) * (
            spacings[y_index] if y_index < len(spacings) else 1.0
        )

        self._remove_colorbar()
        self.axes.clear()
        self.figure.subplots_adjust(left=0.08, right=0.985, bottom=0.10, top=0.97)
        contour = self.axes.contourf(
            x_values,
            y_values,
            np.ma.masked_invalid(plane.T),
            levels=50,
            cmap=self.colormap_combo.currentText(),
        )
        if self.show_legend_checkbox.isChecked():
            self._colorbar = self.figure.colorbar(
                contour,
                ax=self.axes,
                pad=0.015,
                fraction=0.045,
                label=field_label,
            )
        if self.show_axes_checkbox.isChecked():
            self.axes.set_axis_on()
        else:
            self.axes.set_axis_off()
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)
        self.axes.set_aspect("equal", adjustable="box")
        if previous_limits is not None:
            self.axes.set_xlim(previous_limits[0])
            self.axes.set_ylim(previous_limits[1])

        self._selected_field_index = field_index
        self._selected_field_label = field_label
        self._selected_atom_data_index = None
        self._selected_atom_data_label = None
        self._current_view = view
        self._coordinate_labels = (x_label, y_label)
        self.plot_settings_panel.show()
        if not self._left_vertical_sizes_initialized:
            QTimer.singleShot(0, self._set_initial_left_panel_sizes)
        can_show_3d = field.ndim == 3
        self.view_buttons["xz"].setEnabled(can_show_3d)
        self.view_buttons["yz"].setEnabled(can_show_3d)
        self.view_controls.show()
        self.canvas.draw()
        if not preserve_limits:
            self._initial_view_limits = (self.axes.get_xlim(), self.axes.get_ylim())
        self.statusBar().showMessage(
            f"Field: {field_label} ({x_label}{y_label} plane)"
        )

    def _clear_plot(self) -> None:
        self._remove_colorbar()
        self.axes.clear()
        self.axes.set_axis_off()
        self._initial_view_limits = None
        self.view_controls.hide()
        self.plot_settings_panel.hide()
        self._active_plot_kind = None
        self.canvas.draw()

    def _show_axes_toggled(self, show: bool) -> None:
        self.axes.set_axis_on() if show else self.axes.set_axis_off()
        self.canvas.draw_idle()

    def _marker_size_changed(self) -> None:
        try:
            marker_size = float(self.marker_size_edit.text())
        except ValueError:
            marker_size = 0.0
        if not np.isfinite(marker_size) or marker_size <= 0:
            self.marker_size_edit.setStyleSheet("QLineEdit { border: 1px solid red; }")
            self.statusBar().showMessage("Marker size must be a positive number")
            return

        self.marker_size_edit.setStyleSheet("")
        self._marker_size = marker_size
        self._refresh_current_plot()

    def _colormap_changed(self, _: str) -> None:
        self._refresh_current_plot()

    def _show_legend_toggled(self, _: bool) -> None:
        self._refresh_current_plot()

    def _refresh_current_plot(self) -> None:
        if self._active_plot_kind == "atoms":
            self._plot_atoms(
                self._selected_atom_data_index,
                self._selected_atom_data_label,
                self._current_view,
                preserve_limits=True,
            )
        elif (
            self._active_plot_kind == "field"
            and self._selected_field_index is not None
            and self._selected_field_label is not None
        ):
            self._plot_field(
                self._selected_field_index,
                self._selected_field_label,
                self._current_view,
                preserve_limits=True,
            )

    def _view_button_clicked(self, view_name: str, checked: bool = False) -> None:
        if checked:
            return
        if view_name == "O":
            if self._initial_view_limits is not None:
                self.axes.set_xlim(self._initial_view_limits[0])
                self.axes.set_ylim(self._initial_view_limits[1])
                self.canvas.draw_idle()
            return

        views = {
            "xy": (0, 1, "x", "y"),
            "xz": (0, 2, "x", "z"),
            "yz": (1, 2, "y", "z"),
        }
        view = views.get(view_name)
        if view is not None:
            if self._active_plot_kind == "atoms":
                self._plot_atoms(
                    self._selected_atom_data_index,
                    self._selected_atom_data_label,
                    view,
                )
            elif (
                self._active_plot_kind == "field"
                and self._selected_field_index is not None
                and self._selected_field_label is not None
            ):
                field = np.asarray(self.fields[self._selected_field_index])
                if field.ndim == 2 and view != (0, 1, "x", "y"):
                    return
                self._plot_field(
                    self._selected_field_index,
                    self._selected_field_label,
                    view,
                )

    def _remove_colorbar(self) -> None:
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None

    def _plot_mouse_moved(self, event: Any) -> None:
        if event.inaxes is self.axes and event.xdata is not None and event.ydata is not None:
            self._show_mouse_coordinates(event)

    def _plot_mouse_wheel(self, event: Any) -> None:
        if event.inaxes is not self.axes or event.xdata is None or event.ydata is None:
            return

        scale = 1 / 1.2 if event.button == "up" else 1.2
        x_limits = self.axes.get_xlim()
        y_limits = self.axes.get_ylim()
        self.axes.set_xlim(
            event.xdata + (x_limits[0] - event.xdata) * scale,
            event.xdata + (x_limits[1] - event.xdata) * scale,
        )
        self.axes.set_ylim(
            event.ydata + (y_limits[0] - event.ydata) * scale,
            event.ydata + (y_limits[1] - event.ydata) * scale,
        )
        self.canvas.draw_idle()
        self._show_mouse_coordinates(event)

    def _plot_mouse_pressed(self, event: Any) -> None:
        if event.inaxes is not self.axes or event.button not in (MouseButton.MIDDLE, 2):
            return
        if event.x is None or event.y is None:
            return

        self._pan_start = (
            float(event.x),
            float(event.y),
            self.axes.get_xlim(),
            self.axes.get_ylim(),
        )

    def _plot_mouse_released(self, event: Any) -> None:
        if event.button in (MouseButton.MIDDLE, 2):
            self._pan_start = None

    def _plot_mouse_dragged(self, event: Any) -> None:
        if self._pan_start is None or event.inaxes is not self.axes:
            return
        if event.x is None or event.y is None:
            return

        start_x, start_y, x_limits, y_limits = self._pan_start
        width = self.axes.bbox.width
        height = self.axes.bbox.height
        if width <= 0 or height <= 0:
            return

        delta_x = (event.x - start_x) * (x_limits[1] - x_limits[0]) / width
        delta_y = (event.y - start_y) * (y_limits[1] - y_limits[0]) / height
        self.axes.set_xlim(x_limits[0] - delta_x, x_limits[1] - delta_x)
        self.axes.set_ylim(y_limits[0] - delta_y, y_limits[1] - delta_y)
        self.canvas.draw_idle()
        self._show_mouse_coordinates(event)

    def _show_mouse_coordinates(self, event: Any) -> None:
        if event.x is not None and event.y is not None:
            x_coord, y_coord = self.axes.transData.inverted().transform((event.x, event.y))
        else:
            x_coord, y_coord = event.xdata, event.ydata
        x_label, y_label = self._coordinate_labels
        self.statusBar().showMessage(
            f"{x_label}: {x_coord:.6g}, {y_label}: {y_coord:.6g}"
        )

    def _set_initial_panel_sizes(self) -> None:
        splitter_width = self.splitter.width()
        if splitter_width > 0:
            left_width = splitter_width // 3
            self.splitter.setSizes([left_width, splitter_width - left_width])

    def _set_initial_left_panel_sizes(self) -> None:
        panel_height = self.left_splitter.height()
        if panel_height > 0 and self.plot_settings_panel.isVisible():
            upper_height = panel_height * 2 // 3
            self.left_splitter.setSizes(
                [upper_height, panel_height - upper_height]
            )
            self._left_vertical_sizes_initialized = True

    @staticmethod
    def _describe_dataset(value: Any) -> str:
        if isinstance(value, np.ndarray):
            return f"shape={value.shape}, dtype={value.dtype}"
        if isinstance(value, list):
            return repr(value)
        return repr(value)


def main():
    # Create one QApplication instance per application
    app = QApplication(sys.argv)

    # Create a Qt widget, which will be our window, centered on the screen
    window = MainWindow()
    window.show()  # Windows are hidden by default
    frame_geometry = window.frameGeometry()
    frame_geometry.moveCenter(window.screen().availableGeometry().center())
    window.move(frame_geometry.topLeft())

    # Start the event loop
    app.exec()

    # Your application won't reach here until you exit and the event
    # loop has stopped.


if __name__ == "__main__":
    main()