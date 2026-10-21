"""Native QWidget viewport and pixel-accurate interaction/geometry overlays."""
import base64
import math
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QEvent
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget


def decode_image(data):
    image = QImage()
    if data:
        image.loadFromData(base64.b64decode(data.split(',', 1)[-1]))
    return image


def project_segment(a, b, matrix, width, height):
    def clip(p):
        return [sum(v*matrix[i][j] for i, v in enumerate([*p, 1])) for j in range(4)]
    u, v = clip(a), clip(b)
    if not all(math.isfinite(n) for n in u+v):
        return None
    for plane in (lambda p:p[3]-1e-6, lambda p:p[2], lambda p:p[0]+p[3],
                  lambda p:p[3]-p[0], lambda p:p[1]+p[3], lambda p:p[3]-p[1]):
        x, y = plane(u), plane(v)
        if x < 0 and y < 0:
            return None
        if x < 0 or y < 0:
            t = x/(x-y)
            cut = [a+t*(b-a) for a, b in zip(u, v)]
            if x < 0: u = cut
            else: v = cut
    return [QPointF((p[0]/p[3]+1)*width/2-.5, (p[1]/p[3]+1)*height/2-.5) for p in (u,v)]


class Viewport(QWidget):
    gesture = Signal(object, object, int, bool)
    zoom = Signal(float)
    resized = Signal()
    dropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.image = QImage()
        self.base = QImage()
        self.foreground, self.background, self.box = [], [], []
        self.ground = None
        self.visualization = {}
        self.show_bbox = self.show_route = False
        self.mode = 'camera'
        self.tool = 'point'
        self.start = self.end = None
        self.button = 0
        self.blocked = False
        self.hint = 'Open a Gaussian PLY to begin'

    def set_image(self, data):
        self.base = decode_image(data)
        self.image = self.base
        self.update()

    def image_rect(self):
        if self.image.isNull(): return QRectF(self.rect())
        scale = min(self.width()/self.image.width(), self.height()/self.image.height())
        w, h = self.image.width()*scale, self.image.height()*scale
        return QRectF((self.width()-w)/2, (self.height()-h)/2, w, h)

    def pixel(self, pos, clamp=False):
        if self.image.isNull(): return None
        rect = self.image_rect()
        if not clamp and not rect.contains(pos): return None
        return [max(0., min(self.image.width()-1., (pos.x()-rect.x())/rect.width()*self.image.width())),
                max(0., min(self.image.height()-1., (pos.y()-rect.y())/rect.height()*self.image.height()))]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#090b10'))
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        if self.image.isNull():
            painter.setPen(QColor('#9a9eac'))
            painter.drawText(self.rect(), Qt.AlignCenter, self.hint)
            return
        rect = self.image_rect()
        painter.drawImage(rect, self.image)
        painter.save()
        painter.translate(rect.topLeft())
        painter.scale(rect.width()/self.image.width(), rect.height()/self.image.height())
        painter.setClipRect(QRectF(0, 0, self.image.width(), self.image.height()))
        painter.setPen(QPen(QColor('#ff4058'), 2))
        self.draw_geometry(painter)
        if self.ground:
            pixels = self.ground.get('ground_pixels', [])
            painter.setPen(QPen(QColor('#ffc86b'), 2))
            if len(pixels)>1: painter.drawPolyline(QPolygonF([QPointF(*p) for p in pixels]))
            if len(pixels)==3: painter.drawLine(QPointF(*pixels[2]), QPointF(*pixels[0]))
            for i, p in enumerate(pixels):
                painter.setBrush(QColor('#ffc86b')); painter.drawEllipse(QPointF(*p), 5, 5)
                painter.drawText(QPointF(p[0]+8, p[1]-8), str(i+1))
            if self.ground.get('ground_arrow'):
                a,b = self.ground['ground_arrow']; painter.setPen(QPen(QColor('#ff4058'),3))
                painter.drawLine(QPointF(*a),QPointF(*b)); painter.drawText(QPointF(*b),'↑')
        for points, color in ((self.foreground,'#68f4b3'),(self.background,'#ff7777')):
            painter.setPen(QPen(QColor('#090b10'),1)); painter.setBrush(QColor(color))
            for p in points: painter.drawEllipse(QPointF(*p),5,5)
        box = [self.start,self.end] if self.start and self.end and (self.mode=='correction' or self.mode=='prompt' and self.tool=='box') else self.box
        if len(box)==2:
            painter.setBrush(Qt.NoBrush);painter.setPen(QPen(QColor('#ffc857'),2,Qt.DashLine))
            painter.drawRect(QRectF(QPointF(*box[0]),QPointF(*box[1])).normalized())
        painter.restore()

    def draw_geometry(self, painter):
        data=self.visualization
        if not data.get('projection'):return
        def line(a,b,color,label=None):
            points=project_segment(a,b,data['projection'],self.image.width(),self.image.height())
            if points:
                painter.setPen(QPen(QColor(color),2));painter.drawLine(*points)
                if label:painter.drawText(points[1]+QPointF(6,-6),label)
        def frustum(f,color,label=None):
            if not f:return
            tx,ty=f['tan_half_fov'];d=f['depth'];pos=f['position']
            corners=[[pos[i]+d*(f['forward'][i]+x*tx*f['right'][i]+y*ty*f['down'][i]) for i in range(3)] for x,y in ((-1,-1),(1,-1),(1,1),(-1,1))]
            for i in range(4):
                line(pos,corners[i],color);line(corners[i],corners[(i+1)%4],color)
            if label:line(pos,pos,color,label)
        if self.show_bbox and data.get('bbox'):
            c=data['bbox']['corners']
            for i in range(8):
                for j in range(3):
                    if not i & (1<<j):line(c[i],c[i|(1<<j)],'#ff4058')
        route=data.get('route')
        if self.show_route and route:
            frustum(route.get('origin_frustum'),'#f2e6ef','s0')
            for side,color in ((0,'#ffc86b'),(1,'#68d8ff')):
                prev=route['origin'];step=0
                for v in route['views']:
                    if v['drone']!=side:continue
                    step+=1;line(prev,v['position'],color,f'D{side} · {step}')
                    frustum(v.get('frustum'),color);prev=v['position']

    def mousePressEvent(self, event):
        if self.blocked:return
        p=self.pixel(event.position())
        if p is None:return
        self.setFocus();self.start=self.end=p
        self.button=2 if event.button()==Qt.RightButton else 0
        self.shift=bool(event.modifiers() & Qt.ShiftModifier)
        self.update()

    def mouseMoveEvent(self, event):
        if self.start is not None:self.end=self.pixel(event.position(),True);self.update()

    def mouseReleaseEvent(self, event):
        if self.start is None:return
        start=self.start;end=self.pixel(event.position(),True)
        self.start=self.end=None;self.update()
        if end is not None:self.gesture.emit(start,end,self.button,self.shift)

    def wheelEvent(self, event):
        if not self.blocked and self.mode=='camera':self.zoom.emit(event.angleDelta().y())
        event.accept()

    def event(self, event):
        result=super().event(event)
        if event.type()==QEvent.DevicePixelRatioChange:self.resized.emit()
        return result

    def resizeEvent(self, event):
        super().resizeEvent(event);self.resized.emit()

    def dragEnterEvent(self,event):
        urls=event.mimeData().urls()
        if len(urls)==1 and urls[0].isLocalFile() and urls[0].toLocalFile().lower().endswith('.ply'):event.acceptProposedAction()

    def dropEvent(self,event):
        urls=event.mimeData().urls()
        if len(urls)==1:self.dropped.emit(urls[0].toLocalFile());event.acceptProposedAction()
