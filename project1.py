from enum import IntEnum
import sys

from gui import *
from iscissor import *


class ContourMode(IntEnum):
    null = 0
    first = 1
    following = 2
    hold = 3


class WorkerSignals(QObject):
    result = pyqtSignal(object, name='worker')
    progress = pyqtSignal(float, name='progress')


class Worker(QRunnable):
    def __init__(self, func, *args, **kwargs):
        super(Worker, self).__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot(name='worker')
    def run(self):
        outputs = self.func(*self.args, **self.kwargs)
        self.signals.result.emit(outputs)


class IntelligentScissor(BaseApplication, IScissor):

    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('icons/scissor.png'))
        self.threadPool = QThreadPool()
        self.create_dock()
        self.init_ui()
        self.debug_cache = {}
        self.pressed_keys = []
        self.contour_mode = ContourMode.null
        self._reset_gui()
        if self.img is None:
            self.init_img('sample/425.Avatar.Saldana.Worthington.lc.121409.jpg')

    def create_toolbar(self):
        super().create_toolbar()
        self.toolBar.addAction(self.dockView)
        self.toolBar.addWidget(self.statusMsg)

    def create_dock(self):
        panel = QWidget()
        dockLayout = QVBoxLayout(panel)

        snapButton = QPushButton('Seed Snapping')
        snapButton.setCheckable(True)
        snapButton.setChecked(False)
        snapButton.clicked.connect(self.toggle_snapping)
        dockLayout.addWidget(snapButton)

        blurLabel = QLabel('Blurring')
        dockLayout.addWidget(blurLabel)

        blurSlider = QSlider(Qt.Horizontal)
        blurSlider.setMinimum(0)
        blurSlider.setMaximum(4)
        blurSlider.valueChanged.connect(self.blurring)
        dockLayout.addWidget(blurSlider)

        costLabel = QLabel('Cost Function')
        dockLayout.addWidget(costLabel)
        costFunction = QComboBox()
        costFunction.addItems([DistanceMode.lecture.name, DistanceMode.paper.name])
        costFunction.currentTextChanged.connect(self.change_cost_function)
        dockLayout.addWidget(costFunction)

        progressLabel = QLabel('Path Tree Progress')
        dockLayout.addWidget(progressLabel)
        self.progress = QProgressBar(self)
        dockLayout.addWidget(self.progress)

        saveLabel = QLabel('Save Object')
        dockLayout.addWidget(saveLabel)
        self.saveObject = QComboBox()
        self.saveObject.addItems(['Contour', 'Mask', 'Cropped Image'])
        dockLayout.addWidget(self.saveObject)

        self.toolBox = QToolBox()

        self.workBox = QComboBox()
        self.workBox.addItems(['-', 'Contour', 'Image Only'])
        self.workBox.setCurrentIndex(1)
        self.workBox.currentTextChanged.connect(self.change_work_graph)
        self.toolBox.addItem(self.workBox, 'Work Mode')

        self.debugBox = QComboBox()
        self.debugBox.addItems(['-', 'Pixel Node', 'Cost Graph', 'Path Tree'])
        self.debugBox.currentIndexChanged.connect(self.change_debug_graph)
        self.toolBox.addItem(self.debugBox, 'Debug Mode')

        self.toolBox.currentChanged.connect(self.change_graph_mode)
        dockLayout.addWidget(self.toolBox)

        panel.setLayout(dockLayout)

        dock = QDockWidget(panel)
        dock.setMinimumWidth(120)

        self.dockView = dock.toggleViewAction()
        self.dockView.setIcon(QIcon('icons/debug.png'))
        self.dockView.setIconText('Debug')

        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        dock.setWidget(panel)

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open File')
        if file_name:
            self.init_img(file_name)
            self.statusMsg.showMessage('Opened {}'.format(file_name))

    def save_file(self):
        obj = self.saveObject.currentText()
        file_name, _ = QFileDialog.getSaveFileName(self, 'Save File')
        if file_name:
            if obj == 'Contour':
                screenshot = QGraphicsView(self.viewer.scene)
                image = QWidget.grab(screenshot)
            else:
                mask = self.create_mask()
                mask = mask[:, :, np.newaxis]
                white = np.full((self.h, self.w, 3), 255)
                if obj == 'Mask':
                    black = np.full((self.h, self.w, 3), 0)
                    rgb = np.where(mask, white, black)
                    image = QImage(rgb.astype('uint8').flatten(), self.w, self.h,  3 * self.w, QImage.Format_RGB888)
                else:
                    image = np.where(mask, self.img, white)
                    alpha = ~np.all(image == 255, axis=2) * 255
                    rgba = np.dstack((image, alpha))
                    image = QImage(rgba.astype('uint8').flatten(), self.w, self.h, 4 * self.w, QImage.Format_RGBA8888)
            image.save(file_name)
            self.statusMsg.showMessage('Saved {} image to {}'.format(obj, file_name))

    def init_img(self, file_name):
        img = plt.imread(file_name)
        self._set_img(img)
        self.orig_img = np.copy(self.img)

    def _reset_gui(self):
        self.debug_cache = {}
        self.contour_mode = ContourMode.null
        self.toolBox.setCurrentIndex(0)
        self.workBox.setCurrentIndex(1)  # contour mode
        self.viewer.reset()

    def _set_img(self, img):
        self.set_image(img)
        self._reset_gui()
        self.viewer.draw_image(self.img)
        self.cost_matrix = self.compute_cost()
        self.debug_cache['Pixel Node'] = self.compute_pixel_node()
        self.debug_cache['Cost Graph'] = self.compute_cost_graph()

    def toggle_snapping(self):
        self.use_snapping = not self.use_snapping
        self.statusMsg.showMessage('Seed snapping {}activated'.format('' if self.use_snapping else 'de'))

    def blurring(self, value):
        img = cv2.GaussianBlur(self.orig_img, (5, 5), value, value) if value > 0 else self.orig_img
        self.statusMsg.showMessage('Set Gaussian blurring to sigma {}'.format(value) if value > 0 else 'No blurring')
        self._set_img(img)

    def change_cost_function(self, func_name):
        self.dist_mode = DistanceMode.lecture if func_name == DistanceMode.lecture.name else DistanceMode.paper
        self.statusMsg.showMessage('Set {} distance function'.format(func_name))
        self._set_img(self.img)

    def progress_status(self, n):
        self.progress.setValue(int(n * 100))

    def save_path_tree(self, graph):
        self.debug_cache['Path Tree'] = graph
        self.change_debug_graph(graph)
        self.statusMsg.showMessage('Path tree completed')
        self.contour_mode = ContourMode.following

    def start_path_tree(self):
        self.progress.setValue(0)
        worker = Worker(self.compute_path_tree)
        worker.signals.result.connect(self.save_path_tree)
        worker.signals.progress.connect(self.progress_status)
        self.threadPool.start(worker)

    def change_graph_mode(self):
        if self.toolBox.currentIndex() == 0:
            self.debugBox.setCurrentIndex(0)
        else:
            self.workBox.setCurrentIndex(0)

    def change_work_graph(self, graph_name):
        self.viewer.contour_visible = True if graph_name == 'Contour' else False
        self.viewer.draw_image(self.img)

    def change_debug_graph(self, result_graph=None):
        if result_graph is not None:
            self.viewer.contour_visible = False
            graph_name = self.debugBox.currentText()
            result_graph = self.debug_cache.get(graph_name)
            if graph_name == 'Path Tree' and result_graph is None:
                if self.seed is None:
                    self.statusMsg.showMessage('Please set first seed')
                    import urllib.request
                    messageBox = QMessageBox(QMessageBox.Warning, 'Please set first seed', '')
                    url = 'http://thumbsnap.com/s/SJ97O8Gd.jpg'
                    data = urllib.request.urlopen(url).read()
                    pixmap = QPixmap()
                    pixmap.loadFromData(data)
                    messageBox.setIconPixmap(pixmap)
                    messageBox.exec()
                else:
                    self.start_path_tree()
        if result_graph is not None:
            self.viewer.draw_image(result_graph)

    def keyPressEvent(self, QKeyEvent):
        if self.viewer.contour_visible:
            if (QKeyEvent.key() == Qt.Key_Control) and (self.contour_mode == ContourMode.null):
                if not QKeyEvent.isAutoRepeat():
                    self.contour_mode = ContourMode.first
            if (self.contour_mode == ContourMode.first) or (self.contour_mode == ContourMode.following):
                self.pressed_keys.append(QKeyEvent.key())
                if (Qt.Key_Enter in self.pressed_keys) or (Qt.Key_Return in self.pressed_keys):
                    if Qt.Key_Control in self.pressed_keys:
                        self.save_contour()
                        self.viewer.draw_contour(self.contours[-1])
                        self.statusMsg.showMessage('Finished current contour as closed')
                        self.contour_mode = ContourMode.null
                    else:
                        self.viewer.delete_live_wire()
                        self.statusMsg.showMessage('Live wire on hold, click left button to continue')
                        self.contour_mode = ContourMode.hold
                if QKeyEvent.key() == Qt.Key_Backspace:
                    if len(self.viewer.seeds) > 1:
                        self.viewer.delete_seed()
                        self.pop_path()
                        self.viewer.draw_prev_wire(list(chain(*self.paths)))
                    else:
                        self.statusMsg.showMessage('No more following seed')

    def keyReleaseEvent(self, QKeyEvent):
        if self.viewer.contour_visible and (self.contour_mode == ContourMode.first):
            self.contour_mode = ContourMode.null
        self.pressed_keys = []

    def viewerMouseMove(self, x, y):
        if self.viewer.contour_visible:
            if self.contour_mode == ContourMode.following:
                path = self.get_path(x, y)
                self.viewer.draw_live_wire(path)
            elif (self.contour_mode == ContourMode.null) or (self.contour_mode == ContourMode.hold):
                self.viewer.color_change(x, y)

    def viewerMousePress(self, x, y):
        if self.viewer.contour_visible:

            if self.contour_mode == ContourMode.first and self.seed is None:
                x, y = self.set_seed(x, y)
                self.viewer.draw_seed(x, y, self.paths)
                self.statusMsg.showMessage('Set first seed at {}, {}'.format(x, y))
                self.contour_mode = ContourMode.following

            elif self.contour_mode == ContourMode.following:
                x, y = self.commit_path(x, y)
                self.viewer.draw_seed(x, y, self.paths)
                self.statusMsg.showMessage('Set seed at {}, {}'.format(x, y))

            elif self.contour_mode == ContourMode.hold:
                x, y = self.commit_path(x, y)
                self.viewer.draw_seed(x, y, self.paths)
                self.statusMsg.showMessage('Set seed at {}, {}'.format(x, y))
                self.contour_mode = ContourMode.following


def start():
    app = QApplication(['Intelligent Scissor'])
    scissor = IntelligentScissor()
    sys.exit(app.exec())


if __name__ == '__main__':
    start()