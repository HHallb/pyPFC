from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QBrush, QColor, QDoubleValidator, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QInputDialog,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QMenu,
    QPushButton,
    QRadioButton,
    QSplitter,
    QStatusBar,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QSizePolicy,
    QWidget,
)
from functools import partial
import sys
from pathlib import Path
import uuid
import numpy as np
from typing import Any, Callable

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
import pypfc
import torch


class ExportDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        source_path: str | None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setModal(True)
        self.source_paths = [source_path] if source_path else []
        self.destination_path = (
            str(Path(source_path).parent) if source_path else str(Path.cwd())
        )

        layout = QVBoxLayout(self)
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Source"))
        self.source_edit = QLineEdit()
        self.source_edit.setReadOnly(True)
        self._update_source_display()
        source_layout.addWidget(self.source_edit, 1)
        source_button = QPushButton("Browse...")
        source_button.clicked.connect(self._choose_sources)
        source_layout.addWidget(source_button)
        layout.addLayout(source_layout)

        destination_layout = QHBoxLayout()
        destination_layout.addWidget(QLabel("Destination"))
        self.destination_edit = QLineEdit(self.destination_path)
        self.destination_edit.setReadOnly(True)
        destination_layout.addWidget(self.destination_edit, 1)
        destination_button = QPushButton("Browse...")
        destination_button.clicked.connect(self._choose_destination)
        destination_layout.addWidget(destination_button)
        layout.addLayout(destination_layout)

        format_group = QGroupBox("Export format")
        format_layout = QVBoxLayout(format_group)
        self.xyz_radio = QRadioButton("Extended XYZ (atoms)")
        self.vtp_radio = QRadioButton("VTK/VTP (atoms)")
        self.vts_radio = QRadioButton("VTK/VTS (fields)")
        self.xyz_radio.setChecked(True)
        for radio in (self.xyz_radio, self.vtp_radio, self.vts_radio):
            format_layout.addWidget(radio)
        layout.addWidget(format_group)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    @property
    def export_format(self) -> str:
        if self.vtp_radio.isChecked():
            return "vtp"
        if self.vts_radio.isChecked():
            return "vts"
        return "xyz"

    def accept(self) -> None:
        if not self.source_paths:
            QMessageBox.warning(self, "Export", "Select at least one HDF5 source file.")
            return
        super().accept()

    def _choose_sources(self) -> None:
        initial_directory = (
            str(Path(self.source_paths[0]).parent)
            if self.source_paths
            else self.destination_path
        )
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select HDF5 source files",
            initial_directory,
            "HDF5 files (*.h5 *.hdf5)",
        )
        if file_paths:
            self.source_paths = file_paths
            self._update_source_display()

    def _update_source_display(self) -> None:
        if len(self.source_paths) == 1:
            display_text = self.source_paths[0]
        elif self.source_paths:
            display_text = f"{len(self.source_paths)} files selected"
        else:
            display_text = ""
        self.source_edit.setText(display_text)
        self.source_edit.setToolTip("\n".join(self.source_paths))

    def _choose_destination(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select export destination",
            self.destination_path,
        )
        if directory:
            self.destination_path = directory
            self.destination_edit.setText(directory)


class EvaluateDataDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        field_labels: list[str],
        evaluate_callback: Callable[["EvaluateDataDialog"], None],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fields")
        self.setModal(True)
        self._evaluate_callback = evaluate_callback

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.density_combo = QComboBox()
        self.density_combo.addItems(field_labels)
        density_index = self.density_combo.findText("density")
        if density_index >= 0:
            self.density_combo.setCurrentIndex(density_index)
        form.addRow("Select density data", self.density_combo)

        self.evaluation_combo = QComboBox()
        self.evaluation_combo.addItems(
            [
                "Energy density",
                "Chemical potential energy",
                "Grand potential energy",
            ]
        )
        form.addRow("Select field to evaluate", self.evaluation_combo)
        layout.addLayout(form)

        self.interpolate_checkbox = QCheckBox("Do atom interpolation")
        layout.addWidget(self.interpolate_checkbox)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    @property
    def density_label(self) -> str:
        return self.density_combo.currentText()

    @property
    def evaluation_name(self) -> str:
        return self.evaluation_combo.currentText()

    def accept(self) -> None:
        self._evaluate_callback(self)
        super().accept()


class AtomsDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        density_labels: list[str] | None,
        interpolate: bool,
        evaluate_callback: Callable[["AtomsDialog"], None],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Atoms")
        self.setModal(True)
        self.interpolate = interpolate
        self._evaluate_callback = evaluate_callback

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.selection_combo = QComboBox()
        if interpolate:
            self.selection_combo.addItems(density_labels or [])
            form.addRow("Select density data", self.selection_combo)
        else:
            self.selection_combo.addItems(
                ["Centro-Symmetry Parameter (CSP)", "Atom bond data"]
            )
            form.addRow("Select atom data", self.selection_combo)
        layout.addLayout(form)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    @property
    def selection(self) -> str:
        return self.selection_combo.currentText()

    def accept(self) -> None:
        self._evaluate_callback(self)
        super().accept()


class _MiddlePanViewBox(pg.ViewBox):
    def __init__(self) -> None:
        super().__init__()
        self.box_selection_enabled = False
        self.box_selection_callback: Any = None
        self._box_start: QPointF | None = None

    def mouseDragEvent(self, event: Any, axis: int | None = None) -> None:
        if (
            self.box_selection_enabled
            and event.button() == Qt.MouseButton.LeftButton
            and axis is None
        ):
            event.accept()
            current = self.mapToView(event.pos())
            if event.isStart():
                start = self.mapToView(event.buttonDownPos())
                self._box_start = QPointF(start)
                if self.box_selection_callback is not None:
                    self.box_selection_callback("start", self._box_start, current)
            elif self._box_start is not None:
                if self.box_selection_callback is not None:
                    stage = "finish" if event.isFinish() else "move"
                    self.box_selection_callback(stage, self._box_start, current)
                if event.isFinish():
                    self._box_start = None
            return

        if event.button() == Qt.MouseButton.MiddleButton:
            if event.isStart():
                event.accept()
                return
            if not event.isFinish():
                previous = self.mapToView(event.lastPos())
                current = self.mapToView(event.pos())
                self.translateBy(
                    x=previous.x() - current.x(),
                    y=previous.y() - current.y(),
                )
            event.accept()
            return
        super().mouseDragEvent(event, axis=axis)


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
        self._colorbar: pg.ColorBarItem | None = None
        self._colorbar_in_layout = False
        self._selected_atom_data_index: int | None = None
        self._selected_atom_data_label: str | None = None
        self._active_plot_kind: str | None = None
        self._distance_mode_active = False
        self._distance_points: list[tuple[float, float]] = []
        self._distance_line: pg.PlotDataItem | None = None
        self._measured_distance: float | None = None
        self._cursor_coordinates: tuple[float, float] | None = None
        self._box_outline: pg.PlotDataItem | None = None
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
        self._computed_field_labels: set[str] = set()
        self._computed_atom_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._computed_atom_group = False

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
        self.tree.setMouseTracking(True)
        self.tree.viewport().setMouseTracking(True)
        self.tree.viewport().installEventFilter(self)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_tree_context_menu)
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
        self.view_controls.setFixedSize(294, 26)
        self.view_controls_layout = QHBoxLayout(self.view_controls)
        self.view_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.view_controls_layout.setSpacing(2)
        self.view_buttons: dict[str, QToolButton] = {}
        self.distance_button = QToolButton()
        self.distance_button.setText("Dist")
        self.distance_button.setAutoRaise(True)
        self.distance_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self.distance_button.setCheckable(True)
        self.distance_button.setToolTip("Measure the distance between two plot points")
        self.distance_button.toggled.connect(self._distance_mode_toggled)
        self.view_controls_layout.addWidget(self.distance_button)
        self.image_button = QToolButton()
        self.image_button.setText("Img")
        self.image_button.setAutoRaise(True)
        self.image_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self.image_button.setToolTip("Save the current plot as an image")
        self.image_button.clicked.connect(self._save_plot_image)
        self.view_controls_layout.addWidget(self.image_button)
        self.box_button = QToolButton()
        self.box_button.setText("Box")
        self.box_button.setAutoRaise(True)
        self.box_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self.box_button.setCheckable(True)
        self.box_button.setToolTip("Drag a rectangle to zoom into that region")
        self.box_button.toggled.connect(self._box_mode_toggled)
        self.view_controls_layout.addWidget(self.box_button)
        for label in ("Ext", "xy", "xz", "yz"):
            button = QToolButton()
            button.setText(label)
            button.setAutoRaise(True)
            button.setToolButtonStyle(
                Qt.ToolButtonStyle.ToolButtonTextOnly
            )
            button.clicked.connect(partial(self._view_button_clicked, label))
            self.view_controls_layout.addWidget(button)
            self.view_buttons[label] = button
        self.view_controls.hide()
        pg.setConfigOptions(antialias=False, imageAxisOrder="row-major")
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")
        self.graphics.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.view_box = _MiddlePanViewBox()
        self.view_box.box_selection_callback = self._box_selection_changed
        self.plot_item = self.graphics.addPlot(viewBox=self.view_box)
        self.view_box.setBackgroundColor("w")
        self.plot_item.hideAxis("left")
        self.plot_item.hideAxis("bottom")
        self.plot_item.hideAxis("top")
        self.plot_item.hideAxis("right")
        self.plot_layout.addWidget(self.graphics)
        self.graphics.scene().sigMouseMoved.connect(self._plot_mouse_moved)
        self.graphics.scene().sigMouseClicked.connect(self._plot_mouse_clicked)
        self.splitter.addWidget(self.plot_panel)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)
        self.tree.itemSelectionChanged.connect(self._tree_item_selected)

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
        button_export_action = QAction("Export", self)
        button_export_action.setStatusTip("Export data from HDF5 file(s)")
        button_export_action.triggered.connect(self._show_export_dialog)
        toolbar_main.addAction(button_export_action)
        self.field_data_action = QAction("Fields", self)
        self.field_data_action.setStatusTip("Evaluate fields from the selected density data")
        self.field_data_action.triggered.connect(self._show_evaluate_data_dialog)
        self.field_data_action.setVisible(False)
        toolbar_main.addAction(self.field_data_action)
        self.atoms_action = QAction("Atoms", self)
        self.atoms_action.setStatusTip("Interpolate or evaluate atom data")
        self.atoms_action.triggered.connect(self._show_atoms_dialog)
        self.atoms_action.setVisible(False)
        toolbar_main.addAction(self.atoms_action)

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
        self._computed_field_labels.clear()
        self._computed_atom_data.clear()
        self._computed_atom_group = False
        self.field_data_action.setVisible(
            self.fields is not None
            and self.fields.size > 0
            and isinstance(self.field_labels, (list, tuple, np.ndarray))
            and len(self.field_labels) > 0
        )
        self._update_atoms_action_visibility()

        fields_by_label = {}
        if self.fields is not None and self.field_labels is not None:
            fields_by_label = dict(zip(self.field_labels, self.fields, strict=True))
        self.den = fields_by_label.get("density")
        self.ene = fields_by_label.get("energy")

        self.selected_file_path = file_path
        self.loaded_data = loaded_data
        self._populate_hdf5_tree(loaded_data)
        if self._find_tree_item(("atoms",)) is not None:
            self._add_computed_atom_data_items()
        self._clear_plot()
        self.statusBar().showMessage(f"Loaded: {file_path}")

    def _update_atoms_action_visibility(self) -> None:
        has_field_data = (
            self.fields is not None
            and self.fields.ndim == 4
            and self.fields.shape[0] > 0
            and self.field_labels is not None
            and len(self.field_labels) > 0
        )
        self.atoms_action.setVisible(has_field_data or self.atom_coords is not None)

    def _show_export_dialog(self, _: bool = False) -> None:
        dialog = ExportDialog(self, self.selected_file_path)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._export_selected_files(dialog)

    def _show_evaluate_data_dialog(self, _: bool = False) -> None:
        labels = self.field_labels
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        if not labels or not all(isinstance(label, str) for label in labels):
            return

        dialog = EvaluateDataDialog(self, labels, self._evaluate_data)
        dialog.exec()

    def _show_atoms_dialog(self, _: bool = False) -> None:
        if self.loaded_data is None:
            return
        labels = (
            self._string_labels(self.field_labels)
            if self.field_labels is not None
            else []
        )
        atoms_data = self.loaded_data.get("atoms", {})
        interpolation_ready = (
            isinstance(atoms_data, dict)
            and atoms_data.get("coords") is not None
            and atoms_data.get("atom_data") is not None
            and atoms_data.get("atom_data_labels") is not None
        )
        if interpolation_ready:
            dialog = AtomsDialog(self, None, False, self._evaluate_atom_data)
        else:
            if not labels:
                QMessageBox.warning(
                    self,
                    "Atoms",
                    "The open file contains no field data to interpolate.",
                )
                return
            dialog = AtomsDialog(self, labels, True, self._evaluate_atom_data)
        dialog.exec()

    def _evaluate_atom_data(self, dialog: AtomsDialog) -> None:
        if self.loaded_data is None:
            return
        try:
            if dialog.interpolate:
                if self.fields is None:
                    raise ValueError("The open file contains no field data.")
                field_labels = self._string_labels(self.field_labels)
                density_index = field_labels.index(dialog.selection)
                density = np.asarray(self.fields[density_index])
                evaluation_sim = self._create_atom_evaluation_simulation(density)
                atom_coords, interpolated_data = (
                    evaluation_sim.interpolate_density_maxima(density)
                )
                if atom_coords.shape[0] == 0:
                    raise ValueError("No density maxima were found.")
                atom_values = np.ascontiguousarray(interpolated_data[:, :1])
                atom_labels = ["den"]
                self.atom_coords = np.ascontiguousarray(atom_coords)
                self.atom_data = atom_values
                self.atom_data_labels = atom_labels
                atoms_group = self.loaded_data.setdefault("atoms", {})
                atoms_group["coords"] = self.atom_coords
                atoms_group["atom_data"] = self.atom_data
                atoms_group["atom_data_labels"] = self.atom_data_labels
                self._computed_atom_data = {
                    "den": (self.atom_coords, self.atom_data)
                }
                self._computed_atom_group = True
            else:
                if self.atom_coords is None:
                    raise ValueError("The open file contains no /atoms/coords data.")
                field_labels = (
                    self._string_labels(self.field_labels)
                    if self.field_labels is not None
                    else []
                )
                if self.fields is not None and field_labels:
                    source_field = (
                        "density" if "density" in field_labels else field_labels[0]
                    )
                    density = np.asarray(
                        self.fields[field_labels.index(source_field)]
                    )
                else:
                    density = None
                evaluation_sim = self._create_atom_evaluation_simulation(density)
                coords = np.asarray(self.atom_coords)
                if dialog.selection == "Centro-Symmetry Parameter (CSP)":
                    values = evaluation_sim.get_csp(coords)
                    self._store_computed_atom_features(
                        [("csp", np.asarray(values))]
                    )
                elif dialog.selection == "Atom bond data":
                    bond_angles, bond_lengths = evaluation_sim.get_atom_bond_data(
                        coords
                    )
                    self._store_computed_atom_features(
                        [
                            ("min_bond_angle", bond_angles[0]),
                            ("max_bond_angle", bond_angles[1]),
                            ("min_bond_length", bond_lengths[0]),
                            ("max_bond_length", bond_lengths[1]),
                        ]
                    )
                else:
                    raise ValueError(f"Unsupported atom-data choice: {dialog.selection}")
        except (
            ImportError,
            OSError,
            TypeError,
            ValueError,
            KeyError,
            IndexError,
            RuntimeError,
        ) as error:
            QMessageBox.warning(self, "Atoms", f"Could not evaluate atom data:\n{error}")
            return

        self._sync_loaded_data_group("atoms")
        self._populate_hdf5_tree(self.loaded_data)
        self._add_computed_atom_data_items()
        self._color_computed_atom_group()

    def _create_atom_evaluation_simulation(
        self,
        density: np.ndarray | None,
    ) -> pypfc.setup_simulation:
        if density is not None:
            density = np.asarray(density)
        grid = self.loaded_data.get("grid", {}) if self.loaded_data else {}
        ndiv = (
            np.asarray(grid["ndiv"], dtype=int).reshape(-1)
            if grid.get("ndiv") is not None
            else (
                np.asarray(density.shape, dtype=int)
                if density is not None
                else np.empty(0, dtype=int)
            )
        )
        domain_value = grid.get("domain_size")
        if domain_value is None:
            spacing = grid.get("ddiv")
            if spacing is None:
                raise ValueError("The HDF5 file has no grid size or spacing data.")
            domain_size = np.asarray(spacing, dtype=float).reshape(-1) * ndiv
        else:
            domain_size = np.asarray(domain_value, dtype=float).reshape(-1)
        if (
            ndiv.shape != (3,)
            or domain_size.shape != (3,)
            or not np.isfinite(domain_size).all()
            or np.any(domain_size <= 0)
            or (
                density is not None
                and (
                    density.ndim != 3
                    or not np.issubdtype(density.dtype, np.number)
                    or not np.array_equal(ndiv, density.shape)
                )
            )
        ):
            raise ValueError("The density field and grid metadata are incompatible.")

        config = dict(self.loaded_data.get("simulation", {}))
        config.update(
            device_type="cpu",
            verbose=False,
            torch_threads=torch.get_num_threads(),
            torch_threads_interop=torch.get_num_interop_threads(),
        )
        evaluation_sim = pypfc.setup_simulation(
            domain_size=domain_size,
            ndiv=ndiv,
            config=config,
        )
        if density is not None:
            density = np.ascontiguousarray(density, dtype=evaluation_sim._dtype_cpu)
            evaluation_sim.set_density(density)
        return evaluation_sim

    def _store_computed_atom_features(
        self,
        features: list[tuple[str, np.ndarray]],
    ) -> None:
        if self.atom_coords is None:
            raise ValueError("The open file contains no /atoms/coords data.")
        atom_coords = np.asarray(self.atom_coords)
        n_atoms = atom_coords.shape[0]
        if self.atom_data is None:
            atom_data = np.empty((n_atoms, 0), dtype=atom_coords.dtype)
        else:
            atom_data = np.asarray(self.atom_data).copy()
        labels = (
            self._string_labels(self.atom_data_labels)
            if self.atom_data_labels is not None
            else []
        )
        computed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for label, values in features:
            column = np.asarray(values).reshape(-1)
            if column.shape != (n_atoms,):
                raise ValueError(f"Atom data '{label}' has an unexpected shape.")
            if label in labels:
                atom_data[:, labels.index(label)] = column
            else:
                atom_data = np.column_stack((atom_data, column))
                labels.append(label)
            computed[label] = (atom_coords, column[:, np.newaxis])
        self.atom_data = atom_data
        self.atom_data_labels = labels
        self._computed_atom_data.update(computed)
        self._sync_loaded_data_group("atoms")

    def _color_computed_atom_group(self) -> None:
        if not self._computed_atom_group:
            return
        blue = QBrush(QColor("blue"))
        for path in (
            ("atoms",),
            ("atoms", "coords"),
            ("atoms", "atom_data"),
            ("atoms", "atom_data_labels"),
        ):
            item = self._find_tree_item(path)
            if item is not None:
                item.setForeground(0, blue)

    def _evaluate_data(self, dialog: EvaluateDataDialog) -> None:
        if self.fields is None or self.field_labels is None:
            return
        labels = (
            self.field_labels.tolist()
            if isinstance(self.field_labels, np.ndarray)
            else list(self.field_labels)
        )
        try:
            density_index = labels.index(dialog.density_label)
        except ValueError:
            QMessageBox.warning(self, "Evaluate data", "Select a valid density field.")
            return

        density = np.asarray(self.fields[density_index])
        if density.ndim != 3 or not np.issubdtype(density.dtype, np.number):
            QMessageBox.warning(
                self,
                "Evaluate data",
                "The selected density data must be a numeric 3D field.",
            )
            return

        grid = self.loaded_data.get("grid", {}) if self.loaded_data else {}
        ndiv = (
            np.asarray(grid["ndiv"], dtype=int).reshape(-1)
            if grid.get("ndiv") is not None
            else np.asarray(density.shape, dtype=int)
        )
        domain_size_value = grid.get("domain_size")
        if domain_size_value is None:
            ddiv_value = grid.get("ddiv")
            if ddiv_value is None:
                QMessageBox.warning(
                    self,
                    "Evaluate data",
                    "The HDF5 file must contain grid domain size or spacing data.",
                )
                return
            domain_size = np.asarray(ddiv_value, dtype=float).reshape(-1) * ndiv
        else:
            domain_size = np.asarray(domain_size_value, dtype=float).reshape(-1)
        if (
            ndiv.shape != (3,)
            or not np.array_equal(ndiv, density.shape)
            or domain_size.shape != (3,)
            or not np.isfinite(domain_size).all()
            or np.any(domain_size <= 0)
        ):
            QMessageBox.warning(
                self,
                "Evaluate data",
                "The density field and grid metadata are incompatible.",
            )
            return

        simulation_config = dict(
            self.loaded_data.get("simulation", {}) if self.loaded_data else {}
        )
        simulation_config["device_type"] = "cpu"
        simulation_config["verbose"] = False
        simulation_config["torch_threads"] = torch.get_num_threads()
        simulation_config["torch_threads_interop"] = torch.get_num_interop_threads()

        try:
            evaluation_sim = pypfc.setup_simulation(
                domain_size=domain_size,
                ndiv=ndiv,
                config=simulation_config,
            )
            density = np.ascontiguousarray(
                density,
                dtype=evaluation_sim._dtype_cpu,
            )
            evaluation_sim.set_density(density)

            evaluation_names = {
                "Energy density": "energy",
                "Chemical potential energy": "chem_pot",
                "Grand potential energy": "grand_pot_energy",
            }
            label = evaluation_names[dialog.evaluation_name]
            if label == "energy":
                evaluated_field, _ = evaluation_sim.get_energy()
            elif label == "chem_pot":
                evaluated_field, _ = evaluation_sim.get_chemical_potential()
            else:
                evaluated_field, _ = evaluation_sim.get_grand_potential_energy()
            evaluated_field = np.asarray(evaluated_field)
            if evaluated_field.shape != density.shape:
                raise ValueError(
                    f"Evaluated field '{label}' has an unexpected shape."
                )

            interpolated_atoms: tuple[str, np.ndarray, np.ndarray] | None = None
            if dialog.interpolate_checkbox.isChecked():
                atom_coords, atom_data = evaluation_sim.interpolate_density_maxima(
                    density
                )
                if atom_coords.shape[0] == 0:
                    raise ValueError(
                        "No density maxima were found for atom interpolation."
                    )
                atom_label = label
                interpolated_atoms = (
                    atom_label,
                    atom_coords,
                    np.ascontiguousarray(atom_data[:, :1]),
                )
        except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
            QMessageBox.warning(
                self,
                "Evaluate data",
                f"Could not evaluate the selected data:\n{error}",
            )
            return

        self._store_computed_field(label, evaluated_field)
        if interpolated_atoms is not None:
            atom_label, atom_coords, atom_data = interpolated_atoms
            self._computed_atom_data[atom_label] = (atom_coords, atom_data)
        if self.loaded_data is not None:
            self._populate_hdf5_tree(self.loaded_data)
            self._add_computed_atom_data_items()

    def _store_computed_field(self, label: str, field: np.ndarray) -> None:
        if self.fields is None or self.field_labels is None:
            return
        labels = (
            self.field_labels.tolist()
            if isinstance(self.field_labels, np.ndarray)
            else list(self.field_labels)
        )
        field_data = np.asarray(self.fields)
        if field.shape != field_data.shape[1:]:
            raise ValueError(f"Evaluated field '{label}' has an unexpected shape.")
        if label in labels:
            field_data[labels.index(label)] = field
        else:
            field_data = np.concatenate((field_data, field[np.newaxis, ...]), axis=0)
            labels.append(label)
        self.fields = field_data
        self.field_labels = labels
        self._computed_field_labels.add(label)
        if self.loaded_data is not None:
            fields_group = self.loaded_data.setdefault("fields", {})
            fields_group["field_data"] = self.fields
            fields_group["field_labels"] = self.field_labels

    def _add_computed_atom_data_items(self) -> None:
        parent = self._find_tree_item(("atoms", "atom_data_labels"))
        if parent is None:
            atoms_group = self._find_tree_item(("atoms",))
            if atoms_group is None:
                atoms_group = QTreeWidgetItem(["atoms", ""])
                atoms_group.setData(
                    0,
                    Qt.ItemDataRole.UserRole,
                    ("atoms",),
                )
                self.tree.addTopLevelItem(atoms_group)
            parent = QTreeWidgetItem(["atom_data_labels", ""])
            parent.setData(
                0,
                Qt.ItemDataRole.UserRole,
                ("atoms", "atom_data_labels"),
            )
            atoms_group.addChild(parent)
        labels = (
            self._string_labels(self.atom_data_labels)
            if self.atom_data_labels is not None
            else []
        )
        for label in self._computed_atom_data:
            item = next(
                (
                    parent.child(index)
                    for index in range(parent.childCount())
                    if parent.child(index).text(0) == label
                ),
                None,
            )
            if item is None:
                item = QTreeWidgetItem([label, ""])
                item.setData(
                    0,
                    Qt.ItemDataRole.UserRole,
                    ("atoms", "atom_data_labels", label),
                )
                parent.addChild(item)
            item.setData(
                0,
                Qt.ItemDataRole.UserRole + 1,
                labels.index(label) if label in labels else -1,
            )
            item.setData(0, Qt.ItemDataRole.UserRole + 2, "computed")
            item.setForeground(0, QBrush(QColor("blue")))
        parent.setExpanded(True)

    def _show_tree_context_menu(self, position: Any) -> None:
        item = self.tree.itemAt(position)
        if item is None:
            return
        item_path = item.data(0, Qt.ItemDataRole.UserRole)
        if item_path == ("atoms",):
            menu = QMenu(self)
            if self._computed_atom_group:
                add_action = menu.addAction("Add to file")
            delete_action = menu.addAction("Delete")
            action = menu.exec(self.tree.viewport().mapToGlobal(position))
            if self._computed_atom_group and action is add_action:
                self._add_computed_atom_group_to_file()
            if action is delete_action:
                self._delete_atoms_group()
            return
        computed_atom_group_paths = {
            ("atoms", "atom_data"),
            ("atoms", "atom_data_labels"),
            ("atoms", "coords"),
        }
        if (
            self._computed_atom_group
            and item_path in computed_atom_group_paths
            and item.foreground(0).color() == QColor("blue")
        ):
            menu = QMenu(self)
            add_action = menu.addAction("Add to file")
            if menu.exec(self.tree.viewport().mapToGlobal(position)) is add_action:
                self._add_computed_atom_group_to_file()
            return
        if (
            not isinstance(item_path, tuple)
            or len(item_path) != 3
            or item_path[:2]
            not in (("atoms", "atom_data_labels"), ("fields", "field_labels"))
            or item_path == ("fields", "field_labels", "density")
        ):
            return

        self.tree.setCurrentItem(item)
        is_computed = (
            item.data(0, Qt.ItemDataRole.UserRole + 2) == "computed"
            or item_path[2] in self._computed_field_labels
        )
        menu = QMenu(self)
        add_action = menu.addAction("Add to file")
        add_action.setEnabled(is_computed)
        delete_action = menu.addAction("Delete")
        rename_action = menu.addAction("Rename")
        action = menu.exec(self.tree.viewport().mapToGlobal(position))
        if action is add_action:
            self._edit_tree_data_item("add", item_path)
        elif action is delete_action:
            self._edit_tree_data_item("delete", item_path)
        elif action is rename_action:
            self._edit_tree_data_item("rename", item_path)

    def _add_computed_atom_group_to_file(self) -> None:
        if (
            self.selected_file_path is None
            or self.atom_coords is None
            or self.atom_data is None
            or self.atom_data_labels is None
        ):
            return
        try:
            labels = self._string_labels(self.atom_data_labels)
            coords = np.asarray(self.atom_coords)
            atom_data = np.asarray(self.atom_data)
            if (
                coords.ndim != 2
                or coords.shape[1] != 3
                or atom_data.ndim != 2
                or atom_data.shape != (coords.shape[0], len(labels))
            ):
                raise ValueError("Interpolated atom data has an inconsistent shape.")
            self._write_hdf5_label_data(
                "atoms",
                "atom_data",
                "atom_data_labels",
                atom_data,
                labels,
                coords=coords,
                replace_coords=True,
            )
        except (ImportError, OSError, TypeError, ValueError, KeyError, IndexError) as error:
            QMessageBox.critical(
                self,
                "Unable to save atoms",
                f"Could not add interpolated atom data to the HDF5 file:\n{error}",
            )
            return

        self._computed_atom_group = False
        self._computed_atom_data.clear()
        self._sync_loaded_data_group("atoms")
        if self.loaded_data is not None:
            self._populate_hdf5_tree(self.loaded_data)
        self.statusBar().showMessage("Added interpolated atoms to the HDF5 file")

    def _delete_atoms_group(self) -> None:
        if self.selected_file_path is None:
            return
        try:
            import h5py
        except ImportError as error:
            QMessageBox.critical(
                self,
                "Unable to delete atoms",
                f"h5py is required to edit HDF5 data:\n{error}",
            )
            return

        try:
            with h5py.File(self.selected_file_path, "r+") as h5_file:
                if "atoms" in h5_file:
                    del h5_file["atoms"]
        except (OSError, RuntimeError, ValueError) as error:
            QMessageBox.critical(
                self,
                "Unable to delete atoms",
                f"Could not delete '/atoms' from the open file:\n{error}",
            )
            return

        if self.loaded_data is not None:
            self.loaded_data.pop("atoms", None)
        self.atom_coords = None
        self.atom_data = None
        self.atom_data_labels = None
        self._computed_atom_data.clear()
        self._computed_atom_group = False
        self._populate_hdf5_tree(self.loaded_data or {})
        self._update_atoms_action_visibility()
        self._clear_plot()
        self.statusBar().showMessage("Deleted /atoms from the open file")

    def _edit_tree_data_item(
        self,
        operation: str,
        item_path: tuple[str, ...],
    ) -> None:
        group_name, _, label = item_path
        is_field = group_name == "fields"
        data_key = "field_data" if is_field else "atom_data"
        label_key = "field_labels" if is_field else "atom_data_labels"
        if self.selected_file_path is None:
            return

        item = self.tree.currentItem()
        is_computed = (
            item is not None
            and item.data(0, Qt.ItemDataRole.UserRole) == item_path
            and item.data(0, Qt.ItemDataRole.UserRole + 2) == "computed"
        ) or (is_field and label in self._computed_field_labels)

        new_label = label
        if operation == "rename":
            new_label, accepted = QInputDialog.getText(
                self,
                "Rename data",
                "New name:",
                QLineEdit.EchoMode.Normal,
                label,
            )
            new_label = new_label.strip()
            if not accepted:
                return
            if not new_label:
                QMessageBox.warning(self, "Rename data", "The name cannot be empty.")
                return

        previous_state = (
            self.fields,
            self.field_labels,
            self.den,
            self.ene,
            self.atom_data,
            self.atom_data_labels,
            self._computed_field_labels.copy(),
            self._computed_atom_data.copy(),
            self.atom_coords,
            self._computed_atom_group,
        )
        try:
            disk_data, disk_labels, disk_coords = self._read_hdf5_label_data(
                group_name,
                data_key,
                label_key,
            )
            disk_index = disk_labels.index(label) if label in disk_labels else None
            if operation == "add":
                if not is_computed:
                    return
                if is_field:
                    session_labels = self._string_labels(self.field_labels)
                    session_data = np.asarray(self.fields)
                    session_index = session_labels.index(label)
                    new_disk_data, new_disk_labels = self._replace_or_append_column(
                        disk_data,
                        disk_labels,
                        label,
                        session_data[session_index],
                        field=True,
                    )
                else:
                    if label not in self._computed_atom_data:
                        return
                    computed_coords, computed_values = self._computed_atom_data[label]
                    if disk_coords is None:
                        if disk_data.size:
                            raise ValueError(
                                "The atom data in the open file has no matching "
                                "coordinate dataset."
                            )
                        disk_data = np.empty(
                            (computed_coords.shape[0], 0),
                            dtype=computed_values.dtype,
                        )
                    elif (
                        computed_coords.shape != disk_coords.shape
                        or not np.allclose(computed_coords, disk_coords)
                    ):
                        raise ValueError(
                            "Interpolated atom data can only be added when its "
                            "coordinates match the atoms in the open file."
                        )
                    new_disk_data, new_disk_labels = self._replace_or_append_column(
                        disk_data,
                        disk_labels,
                        label,
                        computed_values[:, 0],
                        field=False,
                    )
                self._write_hdf5_label_data(
                    group_name,
                    data_key,
                    label_key,
                    new_disk_data,
                    new_disk_labels,
                    coords=computed_coords if not is_field and disk_coords is None else None,
                )
                if not is_field:
                    if self.atom_coords is None:
                        self.atom_coords = computed_coords
                    session_atom_labels = (
                        self._string_labels(self.atom_data_labels)
                        if self.atom_data_labels is not None
                        else []
                    )
                    self._set_session_atom_data(
                        self.atom_data,
                        session_atom_labels,
                        label,
                        self._computed_atom_data[label][1][:, 0],
                    )
                    self._computed_atom_data.pop(label, None)
                    if disk_coords is None:
                        self._computed_atom_group = False
                if is_field:
                    self._computed_field_labels.discard(label)
            elif operation in ("delete", "rename"):
                if operation == "rename":
                    if new_label == label:
                        return
                    if is_field:
                        session_labels = self._string_labels(self.field_labels)
                    elif is_computed:
                        base_atom_labels = (
                            self._string_labels(self.atom_data_labels)
                            if self.atom_data_labels is not None
                            else []
                        )
                        session_labels = [
                            *base_atom_labels,
                            *self._computed_atom_data.keys(),
                        ]
                    else:
                        session_labels = self._string_labels(self.atom_data_labels)
                    if (
                        (new_label in disk_labels or new_label in session_labels)
                        and new_label != label
                    ):
                        QMessageBox.warning(
                            self,
                            "Rename data",
                            f"An item named '{new_label}' already exists.",
                        )
                        return

                if is_field:
                    session_labels = self._string_labels(self.field_labels)
                    session_data = np.asarray(self.fields)
                    session_index = session_labels.index(label)
                    if operation == "delete":
                        session_data = np.delete(session_data, session_index, axis=0)
                        session_labels.pop(session_index)
                        self._computed_field_labels.discard(label)
                    else:
                        session_labels[session_index] = new_label
                        if label in self._computed_field_labels:
                            self._computed_field_labels.discard(label)
                            self._computed_field_labels.add(new_label)
                    self.fields = session_data
                    self.field_labels = session_labels
                    self.den = (
                        session_data[session_labels.index("density")]
                        if "density" in session_labels
                        else None
                    )
                    self.ene = (
                        session_data[session_labels.index("energy")]
                        if "energy" in session_labels
                        else None
                    )
                elif is_computed:
                    atom_coords, atom_values = self._computed_atom_data[label]
                    session_labels = (
                        self._string_labels(self.atom_data_labels)
                        if self.atom_data_labels is not None
                        else []
                    )
                    if label in session_labels and self.atom_data is not None:
                        session_data = np.asarray(self.atom_data)
                        session_index = session_labels.index(label)
                        if operation == "delete":
                            session_data = np.delete(
                                session_data,
                                session_index,
                                axis=1,
                            )
                            session_labels.pop(session_index)
                        else:
                            session_labels[session_index] = new_label
                        self.atom_data = session_data
                        self.atom_data_labels = session_labels
                    if operation == "delete":
                        self._computed_atom_data.pop(label)
                    else:
                        self._computed_atom_data.pop(label)
                        self._computed_atom_data[new_label] = (
                            atom_coords,
                            atom_values,
                        )
                else:
                    session_labels = self._string_labels(self.atom_data_labels)
                    session_data = np.asarray(self.atom_data)
                    session_index = session_labels.index(label)
                    if operation == "delete":
                        session_data = np.delete(session_data, session_index, axis=1)
                        session_labels.pop(session_index)
                    else:
                        session_labels[session_index] = new_label
                    self.atom_data = session_data
                    self.atom_data_labels = session_labels

                if disk_index is not None:
                    if operation == "delete":
                        axis = 0 if is_field else 1
                        disk_data = np.delete(disk_data, disk_index, axis=axis)
                        disk_labels.pop(disk_index)
                    else:
                        disk_labels[disk_index] = new_label
                    self._write_hdf5_label_data(
                        group_name,
                        data_key,
                        label_key,
                        disk_data,
                        disk_labels,
                    )
            else:
                return
        except (ImportError, OSError, TypeError, ValueError, KeyError, IndexError) as error:
            (
                self.fields,
                self.field_labels,
                self.den,
                self.ene,
                self.atom_data,
                self.atom_data_labels,
                self._computed_field_labels,
                self._computed_atom_data,
                self.atom_coords,
                self._computed_atom_group,
            ) = previous_state
            QMessageBox.critical(
                self,
                "Unable to update HDF5 data",
                f"Could not {operation} '{label}':\n{error}",
            )
            return

        self._sync_loaded_data_group(group_name)
        if self.loaded_data is not None:
            self._populate_hdf5_tree(self.loaded_data)
            self._add_computed_atom_data_items()
            self._color_computed_atom_group()
        self.statusBar().showMessage(f"Updated {group_name} data: {label}")

    @staticmethod
    def _string_labels(labels: Any) -> list[str]:
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        if not isinstance(labels, (list, tuple)):
            raise ValueError("The HDF5 label collection is unavailable.")
        if not all(isinstance(label, str) for label in labels):
            raise ValueError("The HDF5 label collection must contain strings.")
        return list(labels)

    def _read_hdf5_label_data(
        self,
        group_name: str,
        data_key: str,
        label_key: str,
    ) -> tuple[np.ndarray, list[str], np.ndarray | None]:
        try:
            import h5py
        except ImportError as error:
            raise ImportError("h5py is required to edit HDF5 data.") from error

        group_path = f"/{group_name}"
        with h5py.File(self.selected_file_path, "r") as h5_file:
            group = h5_file.get(group_path)
            coords = (
                np.asarray(h5_file["/atoms/coords"])
                if group_name == "atoms"
                and group is not None
                and "coords" in group
                else None
            )
            if group is None and group_name == "atoms":
                data = np.empty((0, 0), dtype=float)
                labels = []
            elif group is None:
                raise KeyError(f"The HDF5 file has no {group_path} group.")
            elif (
                data_key not in group
                and label_key not in group
                and group_name == "atoms"
            ):
                data = np.empty(
                    (coords.shape[0], 0) if coords is not None else (0, 0),
                    dtype=coords.dtype if coords is not None else float,
                )
                labels = []
            elif data_key in group and label_key in group:
                data = np.asarray(group[data_key])
                labels = group[label_key].asstr()[()].tolist()
                if isinstance(labels, str):
                    labels = [labels]
            else:
                raise KeyError(
                    f"{group_path} must contain both '{data_key}' and '{label_key}'."
                )
        if data.shape[0 if group_name == "fields" else 1] != len(labels):
            raise ValueError(
                f"{group_path}/{data_key} and {group_path}/{label_key} do not match."
            )
        return data, self._string_labels(labels), coords

    @staticmethod
    def _replace_or_append_column(
        data: np.ndarray,
        labels: list[str],
        label: str,
        values: np.ndarray,
        *,
        field: bool,
    ) -> tuple[np.ndarray, list[str]]:
        values = np.asarray(values)
        index = labels.index(label) if label in labels else None
        axis = 0 if field else 1
        expected_shape = data.shape[1:] if field else (data.shape[0],)
        if values.shape != expected_shape:
            raise ValueError(
                f"Data for '{label}' has shape {values.shape}; expected {expected_shape}."
            )
        if index is None:
            data = np.concatenate(
                (data, np.expand_dims(values, axis=axis)),
                axis=axis,
            )
            labels.append(label)
        else:
            if field:
                data[index] = values
            else:
                data[:, index] = values
        return data, labels

    def _write_hdf5_label_data(
        self,
        group_name: str,
        data_key: str,
        label_key: str,
        data: np.ndarray,
        labels: list[str],
        coords: np.ndarray | None = None,
        replace_coords: bool = False,
    ) -> None:
        try:
            import h5py
        except ImportError as error:
            raise ImportError("h5py is required to edit HDF5 data.") from error

        temporary_suffix = uuid.uuid4().hex
        data_temp = f"_pypfc_gui_data_{temporary_suffix}"
        labels_temp = f"_pypfc_gui_labels_{temporary_suffix}"
        coords_temp = f"_pypfc_gui_coords_{temporary_suffix}"
        with h5py.File(self.selected_file_path, "r+") as h5_file:
            group = h5_file.require_group(f"/{group_name}")
            options = {}
            data_attrs = {}
            labels_attrs = {}
            if data_key in group and label_key in group:
                old_data = group[data_key]
                old_labels = group[label_key]
                data_attrs = dict(old_data.attrs.items())
                labels_attrs = dict(old_labels.attrs.items())
                if old_data.compression is not None:
                    options["compression"] = old_data.compression
                    if old_data.compression_opts is not None:
                        options["compression_opts"] = old_data.compression_opts
                if old_data.shuffle:
                    options["shuffle"] = True
                if old_data.fletcher32:
                    options["fletcher32"] = True
            elif data_key in group or label_key in group:
                raise KeyError(
                    f"/{group_name} must contain both '{data_key}' and '{label_key}'."
                )

            new_data = group.create_dataset(data_temp, data=data, **options)
            new_labels = group.create_dataset(
                labels_temp,
                data=np.asarray(labels, dtype=object),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            for key, value in data_attrs.items():
                new_data.attrs[key] = value
            for key, value in labels_attrs.items():
                new_labels.attrs[key] = value
            if coords is not None and ("coords" not in group or replace_coords):
                group.create_dataset(coords_temp, data=coords)

            if data_key in group:
                del group[data_key]
                del group[label_key]
            group.move(data_temp, data_key)
            group.move(labels_temp, label_key)
            if coords_temp in group:
                if "coords" in group:
                    del group["coords"]
                group.move(coords_temp, "coords")

    def _set_session_atom_data(
        self,
        atom_data: np.ndarray | None,
        labels: list[str],
        label: str,
        values: np.ndarray,
    ) -> None:
        if self.atom_coords is None:
            raise ValueError("The open file has no atom coordinates.")
        atom_coords = np.asarray(self.atom_coords)
        values = np.asarray(values)
        if values.shape != (atom_coords.shape[0],):
            raise ValueError(f"Interpolated data for '{label}' has an unexpected shape.")
        if atom_data is None:
            atom_data = np.empty((atom_coords.shape[0], 0), dtype=values.dtype)
        else:
            atom_data = np.asarray(atom_data)
        if label in labels:
            atom_data[:, labels.index(label)] = values
        else:
            atom_data = np.column_stack((atom_data, values))
            labels.append(label)
        self.atom_data = atom_data
        self.atom_data_labels = labels
        if self.loaded_data is not None:
            atoms_group = self.loaded_data.setdefault("atoms", {})
            atoms_group["atom_data"] = atom_data
            atoms_group["atom_data_labels"] = labels

    def _sync_loaded_data_group(self, group_name: str) -> None:
        if self.loaded_data is None:
            return
        group = self.loaded_data.setdefault(group_name, {})
        if group_name == "fields":
            group["field_data"] = self.fields
            group["field_labels"] = self.field_labels
        else:
            if self.atom_data is not None:
                group["atom_data"] = self.atom_data
            if self.atom_data_labels is not None:
                group["atom_data_labels"] = self.atom_data_labels

    def _find_tree_item(self, path: tuple[str, ...]) -> QTreeWidgetItem | None:
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            found = self._find_tree_item_in_branch(item, path)
            if found is not None:
                return found
        return None

    @classmethod
    def _find_tree_item_in_branch(
        cls,
        item: QTreeWidgetItem,
        path: tuple[str, ...],
    ) -> QTreeWidgetItem | None:
        if item.data(0, Qt.ItemDataRole.UserRole) == path:
            return item
        for index in range(item.childCount()):
            found = cls._find_tree_item_in_branch(item.child(index), path)
            if found is not None:
                return found
        return None

    def _export_selected_files(self, dialog: ExportDialog) -> None:
        format_names = {
            "xyz": "Extended XYZ",
            "vtp": "VTK/VTP",
            "vts": "VTK/VTS",
        }
        format_name = format_names[dialog.export_format]
        source_count = len(dialog.source_paths)
        exported_count = 0
        for source_path in dialog.source_paths:
            try:
                if self._export_hdf5_file(
                    source_path,
                    dialog.destination_path,
                    dialog.export_format,
                ):
                    exported_count += 1
            except (
                ImportError,
                OSError,
                ValueError,
                TypeError,
                AssertionError,
                RuntimeError,
                IndexError,
            ):
                continue

        summary = QMessageBox(self)
        summary.setWindowTitle("Export summary")
        summary.setText(
            f"{exported_count} of {source_count} selected files exported into\n"
            f"{format_name} format"
        )
        summary.setStandardButtons(QMessageBox.StandardButton.Ok)
        summary.setWindowModality(Qt.WindowModality.ApplicationModal)
        message_label = summary.findChild(QLabel, "qt_msgbox_label")
        layout = summary.layout()
        if message_label is not None:
            message_label.setAlignment(
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
            )
            if layout is not None:
                position = layout.getItemPosition(layout.indexOf(message_label))
                layout.addWidget(
                    message_label,
                    position[0],
                    0,
                    position[2],
                    layout.columnCount(),
                )
        button_box = summary.findChild(QDialogButtonBox)
        if button_box is not None and layout is not None:
            layout.setAlignment(button_box, Qt.AlignmentFlag.AlignHCenter)
        summary.exec()

    def _export_hdf5_file(
        self,
        source_path: str,
        destination_path: str,
        export_format: str,
    ) -> bool:
        data = self.sim.load_hdf5(source_path)
        Path(destination_path).mkdir(parents=True, exist_ok=True)
        output_base = Path(destination_path) / Path(source_path).stem

        if export_format in ("xyz", "vtp"):
            atoms = data.get("atoms", {})
            coords_value = atoms.get("coords")
            atom_data_value = atoms.get("atom_data")
            labels_value = atoms.get("atom_data_labels")
            if coords_value is None or atom_data_value is None or labels_value is None:
                return False

            coords = np.asarray(coords_value)
            atom_data = np.asarray(atom_data_value)
            labels = labels_value.tolist() if isinstance(labels_value, np.ndarray) else labels_value
            if (
                coords.ndim != 2
                or coords.shape[1] != 3
                or coords.shape[0] == 0
                or not np.issubdtype(coords.dtype, np.number)
                or atom_data.ndim != 2
                or atom_data.shape[0] != coords.shape[0]
                or atom_data.shape[1] == 0
                or not np.issubdtype(atom_data.dtype, np.number)
                or not isinstance(labels, list)
                or len(labels) != atom_data.shape[1]
                or not all(isinstance(label, str) for label in labels)
            ):
                return False

            scalar_data = [
                np.ascontiguousarray(atom_data[:, index])
                for index in range(atom_data.shape[1])
            ]
            if export_format == "xyz":
                grid = data.get("grid", {})
                domain_size_value = grid.get("domain_size")
                if domain_size_value is None:
                    ndiv_value = grid.get("ndiv")
                    ddiv_value = grid.get("ddiv")
                    if ndiv_value is None or ddiv_value is None:
                        return False
                    domain_size = (
                        np.asarray(ndiv_value, dtype=float).reshape(-1)
                        * np.asarray(ddiv_value, dtype=float).reshape(-1)
                    )
                else:
                    domain_size = np.asarray(
                        domain_size_value,
                        dtype=float,
                    ).reshape(-1)
                if (
                    domain_size.shape != (3,)
                    or not np.isfinite(domain_size).all()
                    or np.any(domain_size <= 0)
                ):
                    return False

                previous_domain_size = self.sim._domain_size
                try:
                    self.sim._domain_size = domain_size
                    self.sim.write_extended_xyz(
                        str(output_base),
                        np.ascontiguousarray(coords),
                        scalar_data,
                        labels,
                        simulation_time=float(
                            data.get("meta", {}).get("simulation_time", 0.0)
                        ),
                        gz=False,
                    )
                finally:
                    self.sim._domain_size = previous_domain_size
                output_path = Path(f"{output_base}.xyz")
            else:
                self.sim.write_vtk_points(
                    str(output_base),
                    np.ascontiguousarray(coords),
                    scalar_data,
                    labels,
                )
                output_path = Path(f"{output_base}.vtp")
        elif export_format == "vts":
            fields = data.get("fields", {})
            field_data_value = fields.get("field_data")
            field_labels_value = fields.get("field_labels")
            if field_data_value is None or field_labels_value is None:
                return False

            field_data = np.asarray(field_data_value)
            field_labels = (
                field_labels_value.tolist()
                if isinstance(field_labels_value, np.ndarray)
                else field_labels_value
            )
            grid = data.get("grid", {})
            if (
                field_data.ndim != 4
                or field_data.shape[0] == 0
                or not np.issubdtype(field_data.dtype, np.number)
                or not isinstance(field_labels, list)
                or len(field_labels) != field_data.shape[0]
                or not all(isinstance(label, str) for label in field_labels)
            ):
                return False

            ndiv_value = grid.get("ndiv")
            ndiv = (
                np.asarray(ndiv_value, dtype=int).reshape(-1)
                if ndiv_value is not None
                else np.asarray(field_data.shape[1:], dtype=int)
            )
            if ndiv.shape != (3,) or not np.array_equal(ndiv, field_data.shape[1:]):
                return False
            ddiv_value = grid.get("ddiv")
            if ddiv_value is None:
                domain_size = grid.get("domain_size")
                if domain_size is None:
                    return False
                domain_size_array = np.asarray(domain_size, dtype=float).reshape(-1)
                if (
                    domain_size_array.shape != (3,)
                    or not np.isfinite(domain_size_array).all()
                    or np.any(domain_size_array <= 0)
                ):
                    return False
                ddiv = domain_size_array / ndiv
            else:
                ddiv = np.asarray(ddiv_value, dtype=float).reshape(-1)
            if ddiv.shape != (3,) or not np.isfinite(ddiv).all() or np.any(ddiv <= 0):
                return False

            previous_ndiv = self.sim._ndiv
            previous_ddiv = self.sim._ddiv
            try:
                self.sim._ndiv = ndiv
                self.sim._ddiv = ddiv
                self.sim.write_vtk_structured_grid(
                    str(output_base),
                    [np.ascontiguousarray(field) for field in field_data],
                    field_labels,
                )
            finally:
                self.sim._ndiv = previous_ndiv
                self.sim._ddiv = previous_ddiv
            output_path = Path(f"{output_base}.vts")
        else:
            return False

        return output_path.is_file() and output_path.stat().st_size > 0

    def _save_plot_image(self) -> None:
        if self._active_plot_kind is None:
            return

        if self.selected_file_path:
            default_path = str(Path(self.selected_file_path).with_suffix(".png"))
        else:
            default_path = str(Path.cwd() / "plot.png")
        png_filter = "PNG image (*.png)"
        tiff_filter = "TIFF image (*.tif *.tiff)"
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save plot image",
            default_path,
            f"{png_filter};;{tiff_filter}",
            png_filter,
        )
        if not file_path:
            return

        output_path = Path(file_path)
        suffix = output_path.suffix.lower()
        if suffix not in (".png", ".tif", ".tiff"):
            extension = ".tif" if selected_filter == tiff_filter else ".png"
            output_path = output_path.with_suffix(extension)
        elif suffix == ".png" and selected_filter == tiff_filter:
            output_path = output_path.with_suffix(".tif")

        try:
            ImageExporter(self.plot_item).export(fileName=str(output_path))
        except (OSError, ValueError, RuntimeError) as error:
            QMessageBox.critical(
                self,
                "Unable to save plot image",
                f"Could not save '{output_path}':\n{error}",
            )
            self.statusBar().showMessage(f"Failed to save image: {output_path}")
            return

        self.statusBar().showMessage(f"Saved image: {output_path}")

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
            if label in self._computed_field_labels:
                item.setForeground(0, QBrush(QColor("blue")))
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

    def eventFilter(self, watched: Any, event: QEvent) -> bool:
        if watched is self.tree.viewport():
            if event.type() == QEvent.Type.MouseMove and isinstance(event, QMouseEvent):
                item = self.tree.itemAt(event.position().toPoint())
                cursor = (
                    Qt.CursorShape.PointingHandCursor
                    if self._tree_item_is_clickable(item)
                    else Qt.CursorShape.ArrowCursor
                )
                self.tree.viewport().setCursor(cursor)
            elif event.type() == QEvent.Type.Leave:
                self.tree.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        return super().eventFilter(watched, event)

    @staticmethod
    def _tree_item_is_clickable(item: QTreeWidgetItem | None) -> bool:
        if item is None:
            return False
        item_path = item.data(0, Qt.ItemDataRole.UserRole)
        return item_path in (
            ("atoms",),
            ("atoms", "coords"),
            ("atoms", "coord"),
        ) or (
            isinstance(item_path, tuple)
            and len(item_path) == 3
            and item_path[:2]
            in (("atoms", "atom_data_labels"), ("fields", "field_labels"))
        )

    def _plot_atoms(
        self,
        atom_data_index: int | None = None,
        atom_data_label: str | None = None,
        view: tuple[int, int, str, str] = (0, 1, "x", "y"),
        preserve_limits: bool = False,
    ) -> None:
        self._active_plot_kind = "atoms"
        if (
            atom_data_index == -1
            and atom_data_label in self._computed_atom_data
        ):
            atom_coords, atom_data = self._computed_atom_data[atom_data_label]
        else:
            atom_coords, atom_data = self.atom_coords, self.atom_data
        if atom_coords is None:
            self._clear_plot()
            self.statusBar().showMessage("This file does not contain /atoms/coords")
            return
        if atom_coords.ndim != 2 or atom_coords.shape[1] < 2:
            QMessageBox.warning(
                self,
                "Invalid atom coordinates",
                "The /atoms/coords dataset must have at least two columns for an XY plot.",
            )
            return

        x_index, y_index, x_label, y_label = view
        if atom_coords.shape[1] <= max(x_index, y_index):
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
                atom_data is None
                or atom_data.ndim != 2
                or atom_data.shape[0] != atom_coords.shape[0]
                or atom_data_index >= atom_data.shape[1]
            ):
                self._clear_plot()
                QMessageBox.warning(
                    self,
                    "Invalid atom data",
                    f"The /atoms/atom_data dataset has no valid column for '{atom_data_label}'.",
                )
                return
            if not np.issubdtype(atom_data.dtype, np.number):
                self._clear_plot()
                QMessageBox.warning(
                    self,
                    "Invalid atom data",
                    f"The /atoms/atom_data column '{atom_data_label}' must be numeric.",
                )
                return
            color_values = np.asarray(atom_data[:, atom_data_index], dtype=float)
            if not np.isfinite(color_values).any():
                self._clear_plot()
                QMessageBox.warning(
                    self,
                    "Invalid atom data",
                    f"The /atoms/atom_data column '{atom_data_label}' contains no finite values.",
                )
                return

        previous_limits = self._current_view_ranges() if preserve_limits else None
        self._remove_colorbar()
        self._reset_distance_measurement()
        self._reset_box_outline()
        self.plot_item.clear()
        x_values = np.asarray(atom_coords[:, x_index], dtype=float)
        y_values = np.asarray(atom_coords[:, y_index], dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        scatter_items: list[pg.ScatterPlotItem] = []
        if color_values is None:
            scatter_items.append(
                pg.ScatterPlotItem(
                    x=x_values[valid],
                    y=y_values[valid],
                    size=1.0,
                    brush=pg.mkBrush("gray"),
                    pen=None,
                    pxMode=True,
                )
            )
        else:
            color_values = np.asarray(color_values, dtype=float)
            valid &= np.isfinite(color_values)
            if not valid.any():
                self._clear_plot()
                QMessageBox.warning(
                    self,
                    "Invalid atom data",
                    f"No finite coordinates and values are available for '{atom_data_label}'.",
                )
                return
            color_map = self._selected_color_map()
            color_levels = self._color_levels(color_values[valid])
            normalized = (color_values[valid] - color_levels[0]) / (
                color_levels[1] - color_levels[0]
            )
            palette_size = 256
            color_indices = np.minimum(
                (normalized * palette_size).astype(np.int32),
                palette_size - 1,
            )
            order = np.argsort(color_indices)
            sorted_indices = color_indices[order]
            boundaries = np.flatnonzero(np.diff(sorted_indices)) + 1
            color_groups = np.split(order, boundaries)
            x_plot = x_values[valid]
            y_plot = y_values[valid]
            for color_index, point_indices in zip(
                np.unique(sorted_indices),
                color_groups,
                strict=True,
            ):
                scatter_items.append(
                    pg.ScatterPlotItem(
                        x=x_plot[point_indices],
                        y=y_plot[point_indices],
                        size=1.0,
                        brush=pg.mkBrush(
                            color_map.map(
                                (int(color_index) + 0.5) / palette_size,
                                mode="qcolor",
                            )
                        ),
                        pen=None,
                        pxMode=True,
                    )
                )
            self._add_colorbar(color_levels, color_map, atom_data_label)
        if not valid.any():
            self._clear_plot()
            QMessageBox.warning(
                self,
                "Invalid atom coordinates",
                "The atom coordinates contain no finite XY positions for this view.",
            )
            return
        for scatter in scatter_items:
            self.plot_item.addItem(scatter)
        self._configure_axes(x_label, y_label)
        self.plot_item.setAspectLocked(True, ratio=1)
        self.plot_item.autoRange(padding=0.02)
        if previous_limits is not None:
            self._set_view_ranges(previous_limits)
        for scatter in scatter_items:
            self._set_scatter_marker_size(scatter)
        self._selected_atom_data_index = atom_data_index
        self._selected_atom_data_label = atom_data_label
        self._current_view = view
        self._coordinate_labels = (x_label, y_label)
        self.plot_settings_panel.show()
        if not self._left_vertical_sizes_initialized:
            QTimer.singleShot(0, self._set_initial_left_panel_sizes)
        self.view_buttons["xz"].setEnabled(atom_coords.shape[1] >= 3)
        self.view_buttons["yz"].setEnabled(atom_coords.shape[1] >= 3)
        self.view_controls.show()
        if not preserve_limits:
            self._initial_view_limits = (
                self._padded_range(x_values[valid]),
                self._padded_range(y_values[valid]),
            )
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

        previous_limits = self._current_view_ranges() if preserve_limits else None
        spacings = np.asarray(self.ddiv, dtype=float).reshape(-1) if self.ddiv is not None else np.ones(3)
        x_values = np.arange(plane.shape[0]) * (
            spacings[x_index] if x_index < len(spacings) else 1.0
        )
        y_values = np.arange(plane.shape[1]) * (
            spacings[y_index] if y_index < len(spacings) else 1.0
        )

        self._remove_colorbar()
        self._reset_distance_measurement()
        self._reset_box_outline()
        self.plot_item.clear()
        color_levels = self._color_levels(plane)
        image = pg.ImageItem()
        image.setColorMap(self._selected_color_map())
        image.setImage(
            np.asarray(plane.T, dtype=float),
            autoLevels=False,
            levels=color_levels,
        )
        x_spacing = spacings[x_index] if x_index < len(spacings) else 1.0
        y_spacing = spacings[y_index] if y_index < len(spacings) else 1.0
        image.setRect(
            QRectF(
                x_values[0] - x_spacing / 2,
                y_values[0] - y_spacing / 2,
                plane.shape[0] * x_spacing,
                plane.shape[1] * y_spacing,
            )
        )
        self.plot_item.addItem(image)
        if self.show_legend_checkbox.isChecked():
            self._colorbar = pg.ColorBarItem(
                values=color_levels,
                colorMap=self._selected_color_map(),
                label=field_label,
                interactive=False,
            )
            self._colorbar.setImageItem(image, insert_in=self.plot_item)
            self._colorbar_in_layout = True
        self._configure_axes(x_label, y_label)
        self.plot_item.setAspectLocked(True, ratio=1)
        self.plot_item.autoRange(padding=0.02)
        if previous_limits is not None:
            self._set_view_ranges(previous_limits)

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
        if not preserve_limits:
            self._initial_view_limits = (
                (
                    float(x_values[0] - x_spacing / 2),
                    float(x_values[-1] + x_spacing / 2),
                ),
                (
                    float(y_values[0] - y_spacing / 2),
                    float(y_values[-1] + y_spacing / 2),
                ),
            )
        self.statusBar().showMessage(
            f"Field: {field_label} ({x_label}{y_label} plane)"
        )

    def _clear_plot(self) -> None:
        self._reset_distance_measurement()
        self._reset_box_outline()
        if self.distance_button.isChecked():
            self.distance_button.setChecked(False)
        if self.box_button.isChecked():
            self.box_button.setChecked(False)
        self._remove_colorbar()
        self.plot_item.clear()
        for axis in ("left", "bottom", "top", "right"):
            self.plot_item.hideAxis(axis)
        self._initial_view_limits = None
        self.view_controls.hide()
        self.plot_settings_panel.hide()
        self._active_plot_kind = None

    def _show_axes_toggled(self, show: bool) -> None:
        self._configure_axes(*self._coordinate_labels, show=show)

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

    def _show_legend_toggled(self, show: bool) -> None:
        if self._colorbar is None:
            if show:
                self._refresh_current_plot()
            return

        if show:
            if not self._colorbar_in_layout:
                self.plot_item.layout.addItem(self._colorbar, 2, 5)
                self._colorbar_in_layout = True
            self._colorbar.show()
        elif self._colorbar_in_layout:
            self._colorbar.hide()
            self.plot_item.layout.removeItem(self._colorbar)
            self._colorbar_in_layout = False

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
        if view_name == "Ext":
            if self._initial_view_limits is not None:
                self._set_view_ranges(self._initial_view_limits)
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
            colorbar = self._colorbar
            colorbar.hide()
            if self._colorbar_in_layout:
                self.plot_item.layout.removeItem(colorbar)
                self._colorbar_in_layout = False
            self._colorbar = None

    def _selected_color_map(self) -> pg.ColorMap:
        aliases = {
            "coolwarm": "CET-D1",
            "RdBu": "CET-D1",
            "jet": "turbo",
        }
        name = self.colormap_combo.currentText()
        return pg.colormap.get(aliases.get(name, name))

    @staticmethod
    def _color_levels(values: np.ndarray) -> tuple[float, float]:
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        low = float(finite_values.min())
        high = float(finite_values.max())
        if low == high:
            padding = abs(low) * 0.01 or 0.5
            return low - padding, high + padding
        return low, high

    def _add_colorbar(
        self,
        levels: tuple[float, float],
        color_map: pg.ColorMap,
        label: str | None,
    ) -> None:
        if self.show_legend_checkbox.isChecked():
            self._colorbar = pg.ColorBarItem(
                values=levels,
                colorMap=color_map,
                label=label,
                interactive=False,
            )
            self.plot_item.layout.addItem(self._colorbar, 2, 5)
            self._colorbar_in_layout = True

    def _configure_axes(
        self,
        x_label: str,
        y_label: str,
        show: bool | None = None,
    ) -> None:
        if show is None:
            show = self.show_axes_checkbox.isChecked()
        self.plot_item.setLabel("bottom", x_label)
        self.plot_item.setLabel("left", y_label)
        self.plot_item.showAxis("bottom", show)
        self.plot_item.showAxis("left", show)
        self.plot_item.hideAxis("top")
        self.plot_item.hideAxis("right")

    def _current_view_ranges(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        x_range, y_range = self.view_box.viewRange()
        return (
            (float(x_range[0]), float(x_range[1])),
            (float(y_range[0]), float(y_range[1])),
        )

    @staticmethod
    def _padded_range(values: np.ndarray) -> tuple[float, float]:
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        padding = (maximum - minimum) * 0.02
        if padding == 0:
            padding = abs(minimum) * 0.02 or 0.5
        return minimum - padding, maximum + padding

    def _set_view_ranges(
        self,
        ranges: tuple[tuple[float, float], tuple[float, float]],
    ) -> None:
        self.view_box.setRange(
            xRange=ranges[0],
            yRange=ranges[1],
            padding=0,
            disableAutoRange=True,
        )

    def _set_scatter_marker_size(self, scatter: pg.ScatterPlotItem) -> None:
        scatter.setSize(self._marker_size)

    def _distance_mode_toggled(self, enabled: bool) -> None:
        if enabled and self.box_button.isChecked():
            self.box_button.setChecked(False)
        self._distance_mode_active = enabled
        self._reset_distance_measurement()
        if not enabled:
            self._update_plot_status()

    def _box_mode_toggled(self, enabled: bool) -> None:
        if enabled and self.distance_button.isChecked():
            self.distance_button.setChecked(False)
        self.view_box.box_selection_enabled = enabled
        self._reset_box_outline()

    def _reset_box_outline(self) -> None:
        if self._box_outline is not None:
            self.plot_item.removeItem(self._box_outline)
            self._box_outline = None
        self.view_box._box_start = None

    def _box_selection_changed(
        self,
        stage: str,
        start: QPointF,
        current: QPointF,
    ) -> None:
        if stage == "start":
            if self._box_outline is not None:
                self.plot_item.removeItem(self._box_outline)
            self._box_outline = pg.PlotDataItem(
                pen=pg.mkPen("black", width=2),
            )
            self.plot_item.addItem(self._box_outline)

        if self._box_outline is None:
            return

        left, right = sorted((start.x(), current.x()))
        bottom, top = sorted((start.y(), current.y()))
        self._box_outline.setData(
            [left, right, right, left, left],
            [bottom, bottom, top, top, bottom],
        )

        if stage == "finish":
            self._reset_box_outline()
            if left < right and bottom < top:
                self.view_box.setRange(
                    xRange=(left, right),
                    yRange=(bottom, top),
                    padding=0,
                    disableAutoRange=True,
                )

    def _reset_distance_measurement(self) -> None:
        if self._distance_line is not None:
            self.plot_item.removeItem(self._distance_line)
            self._distance_line = None
        self._distance_points.clear()
        self._measured_distance = None

    def _plot_mouse_clicked(self, event: Any) -> None:
        if (
            not self._distance_mode_active
            or self._active_plot_kind is None
            or event.button() != Qt.MouseButton.LeftButton
        ):
            return

        scene_position = event.scenePos()
        if not self.view_box.sceneBoundingRect().contains(scene_position):
            return

        point = self.view_box.mapSceneToView(scene_position)
        coordinates = (float(point.x()), float(point.y()))
        self._cursor_coordinates = coordinates
        if not self._distance_points:
            self._distance_points.append(coordinates)
            return

        first_point = self._distance_points[0]
        self._reset_distance_measurement()
        self._distance_line = pg.PlotDataItem(
            [first_point[0], coordinates[0]],
            [first_point[1], coordinates[1]],
            pen=pg.mkPen("black", width=2),
        )
        self.plot_item.addItem(self._distance_line)
        self._measured_distance = float(
            np.hypot(
                coordinates[0] - first_point[0],
                coordinates[1] - first_point[1],
            )
        )
        self._update_plot_status()

    def _update_plot_status(self) -> None:
        if self._cursor_coordinates is None:
            return
        x_coord, y_coord = self._cursor_coordinates
        x_label, y_label = self._coordinate_labels
        message = f"{x_label}: {x_coord:.6g}, {y_label}: {y_coord:.6g}"
        if self._measured_distance is not None:
            message += f", distance: {self._measured_distance:.6g}"
        self.statusBar().showMessage(message)

    def _plot_mouse_moved(self, position: QPointF) -> None:
        if not self.view_box.sceneBoundingRect().contains(position):
            return
        point = self.view_box.mapSceneToView(position)
        self._cursor_coordinates = (float(point.x()), float(point.y()))
        self._update_plot_status()

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


if __name__ == "__main__":
    main()