from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.Qt import *
from itertools import chain


STYLE = """
    QWidget {
    color: #FFFFFF;
    background-color: #212121;
    border-width: 1px;}
"""


class BaseApplication(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setStyleSheet(STYLE)
        self.setGeometry(60, 60, 920, 800)
        self.viewer = None
        self._imagePressEvent = None
        self.toolBar = None
        self.img = None

    def init_ui(self):
        self.viewer = ImageViewer(self)
        self.viewer.mouseMove.connect(self.viewerMouseMove)
        self.viewer.mousePress.connect(self.viewerMousePress)

        self.toolBar = QToolBar()
        self.toolBar.addAction(QIcon('icons/folder-open.png'), 'Open', self.open_file)
        self.toolBar.addAction(QIcon('icons/pan.png'), 'Drag', self.viewer.toggleDragMode)
        self.toolBar.addAction(QIcon('icons/save.png'), 'Save Image', self.save_file)
        self.addToolBar(self.toolBar)

        self.setCentralWidget(self.viewer)
        self.show()

    def viewerMouseMove(self, x, y):
        pass

    def viewerMousePress(self, x, y):
        pass


class ImageViewer(QGraphicsView):
    mouseMove = pyqtSignal(int, int)
    mousePress = pyqtSignal(int, int)

    def __init__(self, parent):
        super().__init__(parent)
        self._zoom = 0
        self._empty = True
        self._parent = self.parentWidget()
        self.scene = QGraphicsScene(self)
        self.pixmap = None
        self.image = None
        self._h, self._w = None, None
        self.curr_img = None
        self.contour_visible = True
        self.contours = []
        self.seeds = []
        self.live_wire, self.prev_wire, self.selected_item = None, None, None
        self.setScene(self.scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        self.setFrameShape(QGraphicsView.NoFrame)

    def fitInView(self, scale=True):
        rect = QRectF(self.pixmap.rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if not self._empty:
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                view_rect = self.viewport().rect()
                scene_rect = self.transform().mapRect(rect)
                factor = min(view_rect.width() / scene_rect.width(), view_rect.height() / scene_rect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def draw_image(self, img):
        self._h, self._w = img.shape[:2]
        self.image = QImage(img.flatten(), self._w, self._h,  3 * self._w, QImage.Format_RGB888)
        self.update_image()

    def update_image(self):
        self.set_opacity(self.contour_visible)
        self.pixmap = QPixmap(self.image)
        if self.curr_img is None:
            self.curr_img = self.scene.addPixmap(self.pixmap)
            self.curr_img.setZValue(-1)
        else:
            self.curr_img.setPixmap(self.pixmap)
        self.scene.update()
        self._empty = False
        self.fitInView()

    def reset(self):
        self.scene.clear()
        self.pixmap = None
        self.image = None
        self._h, self._w = None, None
        self.curr_img = None
        self.contour_visible = True
        self.contours = []
        self.reset_wire()

    def reset_wire(self):
        for item in self.scene.items():
            if item == self.live_wire or item == self.prev_wire or item in self.seeds:
                self.scene.removeItem(item)
        self.seeds = []
        self.live_wire, self.prev_wire, self.selected_item = None, None, None

    def set_opacity(self, value):
        for seed in self.seeds:
            seed.setOpacity(value)
        for contour in self.contours:
            contour.setOpacity(value)
        if self.selected_item is not None:
            self.selected_item.setOpacity(value)
        if self.prev_wire is not None:
            self.prev_wire.setOpacity(value)
        if self.live_wire is not None:
            self.live_wire.setOpacity(value)

    def draw_seed(self, x, y, prev_paths):
        seed = self.scene.addEllipse(x, y, 1, 1, QPen(Qt.red), QBrush(Qt.red))
        self.seeds.append(seed)
        self.draw_prev_wire(list(chain(*prev_paths)))
        self.scene.update()

    def delete_seed(self):
        seed = self.seeds.pop()
        self.scene.removeItem(seed)
        self.delete_live_wire()

    def create_path(self, path_pts):
        first_pt = True
        path = QPainterPath()
        for x, y in path_pts:
            if first_pt:
                path.moveTo(x, y)
                first_pt = False
            else:
                path.lineTo(x, y)
        return path

    def draw_prev_wire(self, path_pts):
        path = self.create_path(path_pts)
        if self.prev_wire is None:
            self.prev_wire = self.scene.addPath(path, QPen(Qt.red))
        else:
            self.prev_wire.setPath(path)
        self.scene.update()

    def draw_live_wire(self, path_pts):
        path = self.create_path(path_pts)
        if self.live_wire is None:
            self.live_wire = self.scene.addPath(path, QPen(Qt.red))
        else:
            self.live_wire.setPath(path)
        self.scene.update()

    def delete_live_wire(self):
        if self.live_wire is not None:
            self.scene.removeItem(self.live_wire)
            self.live_wire = None

    def draw_contour(self, path_pts):
        self.reset_wire()
        path = self.create_path(path_pts)
        contour = self.scene.addPath(path, QPen(Qt.red))
        self.contours.append(contour)
        self.scene.update()

    def color_change(self, x, y):
        item = self.scene.itemAt(QPoint(x, y), QGraphicsView.transform(self))
        if not isinstance(item, QGraphicsPixmapItem):
            item.setPen(Qt.green)
            self.selected_item = item
        else:
            if self.selected_item is not None:
                self.selected_item.setPen(Qt.red)
        self.scene.update()

    def wheelEvent(self, QWheelEvent):
        if not self._empty:
            if QWheelEvent.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QGraphicsView.NoDrag:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        elif self.dragMode() == QGraphicsView.ScrollHandDrag:
            self.setDragMode(QGraphicsView.NoDrag)

    def mapToImage(self, pos):
        pos = self.mapToScene(pos)
        if (0 <= pos.x() < self._w) and (0 <= pos.y() < self._h):
            return int(pos.x()), int(pos.y())
        return None, None

    def mouseMoveEvent(self, QMouseEvent):
        x, y = self.mapToImage(QMouseEvent.pos())
        if x is not None and y is not None:
            self.mouseMove.emit(x, y)
        super().mouseMoveEvent(QMouseEvent)

    def mousePressEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton:
            x, y = self.mapToImage(QMouseEvent.pos())
            if x is not None and y is not None:
                self.mousePress.emit(x, y)
        super().mousePressEvent(QMouseEvent)
