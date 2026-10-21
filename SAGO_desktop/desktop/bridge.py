"""Serialized direct engine calls. No network server, browser, or web view."""
import threading
import traceback
from PySide6.QtCore import QObject, QThread, Signal, Slot


class EngineWorker(QObject):
    progress = Signal(int, str, object)
    done = Signal(object)
    failed = Signal(str)
    stopped = Signal()

    def __init__(self, factory):
        super().__init__()
        self.factory = factory
        self.engine = None

    @Slot(str, object, object)
    def execute(self, action, data, cancel):
        try:
            if self.engine is None:
                self.engine = self.factory()
            result = self.engine.execute(action, data,
                lambda n, message, views=None: self.progress.emit(n, message, views), cancel)
            self.done.emit(result)
        except Exception as error:
            if not isinstance(error, ValueError):traceback.print_exc()
            self.failed.emit(str(error))

    @Slot()
    def shutdown(self):
        if self.engine is not None:
            self.engine = None
        self.stopped.emit()


class EngineBridge(QObject):
    request = Signal(str, object, object)
    shutdownRequested = Signal()
    progress = Signal(int, str, object)
    completed = Signal(str, object)
    error = Signal(str, str)
    idle = Signal()

    def __init__(self, factory, parent=None):
        super().__init__(parent)
        self.thread = QThread(self)
        self.worker = EngineWorker(factory)
        self.worker.moveToThread(self.thread)
        self.request.connect(self.worker.execute)
        self.shutdownRequested.connect(self.worker.shutdown)
        self.worker.progress.connect(self.progress)
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.stopped.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.busy = False
        self.action = ''
        self.cancel_event = threading.Event()
        self.thread.start()

    def submit(self, action, data):
        if self.busy:
            return False
        self.busy = True
        self.action = action
        self.cancel_event = threading.Event()
        self.request.emit(action, data, self.cancel_event)
        return True

    def cancel(self):
        self.cancel_event.set()

    @Slot(object)
    def _done(self, result):
        action = self.action
        self.busy = False
        self.completed.emit(action, result)
        self.idle.emit()

    @Slot(str)
    def _failed(self, message):
        action = self.action
        self.busy = False
        self.error.emit(action, message)
        self.idle.emit()

    def close(self):
        if self.busy:
            self.cancel()
            return False
        self.shutdownRequested.emit()
        return True
