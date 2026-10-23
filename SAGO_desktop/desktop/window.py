import math
import shutil
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PySide6.QtCore import Qt, QTimer, QSignalBlocker, QSize, QUrl, QEvent
from PySide6.QtGui import QAction, QDesktopServices, QIcon, QKeySequence, QMovie, QPixmap, QColor
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QSplitter, QScrollArea, QPushButton, QLabel, QLineEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QToolButton, QProgressBar, QFileDialog,
    QMessageBox, QInputDialog, QDialog, QDialogButtonBox, QSizePolicy, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QTabWidget, QColorDialog)

from desktop.bridge import EngineBridge
from desktop.canvas import Viewport, decode_image
from desktop.config import ROOT, DATA, DEFAULT_CHECKPOINT, read_config, write_config
from desktop.i18n import Translator


class MainWindow(QMainWindow):
    def __init__(self, factory, output, initial_path=None):
        super().__init__()
        self.settings=read_config();self.tr=Translator(self.settings.get('language','en'))
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.state={};self.capture=None;self.ground=None;self.mode='all'
        self.fg=[];self.bg=[];self.box=[];self.masks=[];self.level=0
        self.loaded_model=None;self.cancel_round_pending=False
        self.foreground_ids=set();self.known_objects=set();self.revision=0
        self.correction=False;self.correction_id=0;self.correction_mode='ellipse'
        self._rendered_aspect=None;self._rendered_size=None;self._objects_signature=None
        self.gif_items=[];self.drone_views=[];self.callback=None;self.close_pending=False
        self._translations=[];self._building=True
        self.bridge=EngineBridge(factory,self)
        self.bridge.completed.connect(self.completed);self.bridge.error.connect(self.failed)
        self.bridge.progress.connect(self.progress);self.bridge.thread.finished.connect(self.close)
        self.preview_timer=QTimer(self);self.preview_timer.setSingleShot(True);self.preview_timer.timeout.connect(self.preview)
        self.resize_timer=QTimer(self);self.resize_timer.setSingleShot(True);self.resize_timer.timeout.connect(self.resize_frame)
        self.quality_timer=QTimer(self);self.quality_timer.setSingleShot(True);self.quality_timer.timeout.connect(self.refine_frame)
        self.wheel_timer=QTimer(self);self.wheel_timer.setSingleShot(True);self.wheel_timer.timeout.connect(self.apply_zoom)
        self.navigation_keys=set();self.pending_pan=[0.,0.];self.navigation_last=time.monotonic()
        self.navigation_timer=QTimer(self);self.navigation_timer.timeout.connect(self.navigate);self.navigation_timer.start(40)
        self.elapsed_timer=QTimer(self);self.elapsed_timer.timeout.connect(self.elapsed);self.started=0
        self.setWindowTitle('SAGO Desktop');self.resize(1620,980)
        self.setWindowIcon(QIcon(str(ROOT/'packaging/sago.svg')))
        self.build_ui();self._building=False;self.update_controls();self.retranslate()
        self.setAcceptDrops(True)
        if initial_path:QTimer.singleShot(0,lambda:self.load_scene(initial_path))

    def text(self,key):return self.tr.text(key)
    def translated(self,widget,key,method='setText'):
        self._translations.append((widget,key,method));getattr(widget,method)(self.text(key));return widget
    def label(self,key):return self.translated(QLabel(),key)
    def button(self,key,fn,checkable=False):
        b=self.translated(QPushButton(),key);b.setCheckable(checkable)
        b.clicked.connect(lambda checked=False:fn());return b
    def check(self,key,fn=None,checked=False):
        b=self.translated(QCheckBox(),key);b.setChecked(checked)
        if fn:b.toggled.connect(lambda _:fn())
        return b
    def combo(self,items):
        c=QComboBox()
        for label,value in items:c.addItem(self.text(label),value)
        c.setProperty('i18n_items',items);return c
    def row(self,*widgets):
        w=QWidget();l=QHBoxLayout(w);l.setContentsMargins(0,0,0,0)
        for widget in widgets:l.addWidget(widget)
        return w
    def section(self,layout,key):
        outer=QWidget();v=QVBoxLayout(outer);v.setContentsMargins(2,3,2,3)
        head=self.translated(QToolButton(),key);head.setCheckable(True);head.setChecked(True)
        head.setToolButtonStyle(Qt.ToolButtonTextBesideIcon);head.setArrowType(Qt.DownArrow)
        head.setSizePolicy(head.sizePolicy().horizontalPolicy(),head.sizePolicy().verticalPolicy())
        body=QWidget();content=QVBoxLayout(body);content.setContentsMargins(5,5,5,8)
        v.addWidget(head);v.addWidget(body);layout.addWidget(outer)
        def toggle(checked):body.setVisible(checked);head.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        head.toggled.connect(toggle)
        return content
    def spin(self,value,low,high,decimals=0):
        w=QDoubleSpinBox() if decimals else QSpinBox()
        w.setRange(low,high);w.setValue(value)
        if decimals:w.setDecimals(decimals)
        return w
    def field(self,layout,key,widget):layout.addWidget(self.label(key));layout.addWidget(widget)
    def scroll_panel(self):
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setMinimumWidth(260)
        content=QWidget();layout=QVBoxLayout(content);layout.setContentsMargins(8,8,8,8)
        scroll.setWidget(content);return scroll,layout

    def build_ui(self):
        toolbar=self.addToolBar('SAGO');toolbar.setMovable(False)
        heading=QLabel('◈ SAGO DESKTOP');heading.setObjectName('heading');toolbar.addWidget(heading)
        toolbar.addSeparator()
        toolbar.addWidget(self.button('Open PLY…',self.open_file))
        toolbar.addWidget(self.button('Open scene folder…',self.open_folder))
        toolbar.addWidget(self.button('Preferences…',self.preferences))
        self.language=QPushButton();self.language.clicked.connect(self.toggle_language);toolbar.addWidget(self.language)
        toolbar.addWidget(self.button('Help',self.help))
        self.splitter=QSplitter();self.setCentralWidget(self.splitter)
        self.left,left=self.scroll_panel();self.splitter.addWidget(self.left)
        scene=self.section(left,'Scene')
        self.path=QLineEdit();self.path.setPlaceholderText('Gaussian PLY / scene directory')
        self.path.returnPressed.connect(lambda:self.load_scene(self.path.text()))
        scene.addWidget(self.path);scene.addWidget(self.button('Open PLY…',lambda:self.load_scene(self.path.text()) if self.path.text() else self.open_file()))
        self.recent=QComboBox();self.refresh_recent();self.recent.activated.connect(lambda i:self.load_scene(self.recent.itemData(i)) if self.recent.itemData(i) else None)
        self.field(scene,'Recent scenes',self.recent);scene.addWidget(self.button('Clear recent',self.clear_recent));self.scene_info=QLabel('—');self.scene_info.setWordWrap(True);scene.addWidget(self.scene_info)
        camera=self.section(left,'Camera');self.preset=QComboBox();self.preset.activated.connect(self.camera_preset)
        self.field(camera,'Original camera',self.preset)
        self.fov=self.spin(50,5,150);self.size=self.combo([('Auto',None)]+[(str(n),n) for n in (512,768,1024,1536,1920,2560)]);self.size.activated.connect(self.refresh)
        camera.addWidget(self.row(self.label('FOV'),self.fov,self.label('Render size'),self.size))
        self.background_rgb=[QDoubleSpinBox() for _ in range(3)]
        for channel, control in zip(('R','G','B'),self.background_rgb):
            control.setRange(0,1);control.setDecimals(4);control.setSingleStep(.01);control.setValue(0);control.setPrefix(channel+' ')
            control.editingFinished.connect(self.refresh)
        self.background_picker=self.button('Choose color…',self.choose_background)
        self.background_flip=self.button('Black / White',self.flip_background)
        self.field(camera,'Background color R / G / B (0–1)',self.row(*self.background_rgb))
        camera.addWidget(self.row(self.background_picker,self.background_flip))
        self.up=[self.spin(n,-1e8,1e8,5) for n in (0,-1,0)]
        self.target=[self.spin(0,-1e8,1e8,5) for _ in range(3)]
        self.field(camera,'Up X / Y / Z',self.row(*self.up));self.field(camera,'Target X / Y / Z',self.row(*self.target))
        camera.addWidget(self.button('Update camera',self.update_camera))
        ground=self.section(left,'Ground calibration')
        ground.addWidget(self.row(self.button('Three points',lambda:self.ground_action('start')),self.button('Flip up',lambda:self.ground_action('flip'))))
        ground.addWidget(self.row(self.button('Undo point',lambda:self.ground_action('undo')),self.button('Reset points',lambda:self.ground_action('reset'))))
        ground.addWidget(self.row(self.button('Apply ground',lambda:self.ground_action('apply')),self.button('Cancel ground',lambda:self.ground_action('cancel'))))
        self.ground_info=QLabel();self.ground_info.setWordWrap(True);ground.addWidget(self.ground_info)
        objects=self.section(left,'Objects');self.object_name=QLineEdit()
        self.field(objects,'Name',self.object_name)
        objects.addWidget(self.button('Save selection',lambda:self.object_action('save',name=self.object_name.text())))
        self.objects=QTableWidget(0,3);self.objects.setSelectionBehavior(QAbstractItemView.SelectRows);self.objects.setSelectionMode(QAbstractItemView.SingleSelection)
        self.objects.setEditTriggers(QAbstractItemView.NoEditTriggers);self.objects.setMinimumHeight(170)
        self.objects.horizontalHeader().setSectionResizeMode(1,QHeaderView.Stretch)
        self.objects.verticalHeader().hide();self.objects.setColumnWidth(0,44);self.objects.setColumnWidth(2,60);self.objects.cellDoubleClicked.connect(lambda row,col:self.rename_object())
        objects.addWidget(self.objects)
        objects.addWidget(self.row(self.button('Select',lambda:self.selected_object_action('select')),self.button('Hide / Show',lambda:self.selected_object_action('hide'))))
        self.delete_object=self.button('Delete',lambda:self.selected_object_action('delete'));self.translated(self.delete_object,'Delete object and return its Gaussians to background (ID 0)','setToolTip')
        objects.addWidget(self.row(self.button('Rename',self.rename_object),self.delete_object))
        self.merge_a=QComboBox();self.merge_b=QComboBox();objects.addWidget(self.row(self.merge_a,self.merge_b))
        objects.addWidget(self.button('Merge IDs',self.merge_ids));objects.addWidget(self.button('Renumber IDs',lambda:self.object_action('renumber')))
        objects.addWidget(self.button('Save IDs',lambda:self.object_action('save_ids')))
        self.ids_info=QLabel();self.ids_info.setWordWrap(True);objects.addWidget(self.ids_info);left.addStretch()
        center=QWidget();cl=QVBoxLayout(center);cl.setContentsMargins(0,0,0,0);self.splitter.addWidget(center)
        self.display=self.combo([('All','all'),('Current foreground','foreground'),('Current background','background')]);self.display.activated.connect(self.change_display)
        self.render_mode=self.combo([(n,n.lower()) for n in ('RGB','Splats','Points','Depth')]);self.render_mode.activated.connect(self.change_render)
        self.correct_button=self.button('Correction',self.start_correction)
        self.correct_confirm=self.button('Confirm correction',self.confirm_correction)
        self.correct_discard=self.button('Discard selection',self.discard_correction)
        cl.addWidget(self.row(self.display,self.render_mode,self.correct_button,self.correct_confirm,self.correct_discard))
        self.canvas=Viewport();cl.addWidget(self.canvas,1)
        self.canvas.pan.connect(self.queue_pan);self.canvas.gesture.connect(self.gesture);self.canvas.zoom.connect(self.zoom)
        self.canvas.resized.connect(lambda:self.resize_timer.start(300));self.canvas.dropped.connect(self.load_scene)
        self.hint=QLabel();cl.addWidget(self.hint)
        self.undo=self.button('Undo',lambda:self.object_action('undo'));self.redo=self.button('Redo',lambda:self.object_action('redo'))
        cl.addWidget(self.row(self.undo,self.redo,self.button('Clear selection',lambda:self.object_action('clear'))))
        self.busy_label=QLabel();self.progress_bar=QProgressBar();self.progress_bar.setRange(0,100)
        self.cancel_button=self.button('Cancel',self.bridge.cancel)
        # Status visibility must never resize the viewport and trigger another render.
        self.busy_label.setFixedHeight(24)
        self.busy_label.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
        self.job_status=self.row(self.progress_bar,self.cancel_button)
        self.job_status.setFixedHeight(34)
        for control in (self.progress_bar,self.cancel_button):
            policy=control.sizePolicy();policy.setRetainSizeWhenHidden(True);control.setSizePolicy(policy)
        cl.addWidget(self.busy_label);cl.addWidget(self.job_status)
        self.right,right=self.scroll_panel();self.splitter.addWidget(self.right)
        prompts=self.section(right,'Segmentation prompts')
        self.field(prompts,'Model',QLabel('SAM2'))
        self.prompt_check=self.check('Enter segmentation prompts (M)',self.toggle_prompts);prompts.addWidget(self.prompt_check)
        self.point_tool=self.button('Point',lambda:self.set_tool('point'),True);self.point_tool.setChecked(True)
        self.box_tool=self.button('Box',lambda:self.set_tool('box'),True)
        self.point_tool.setMaximumWidth(55);self.box_tool.setMaximumWidth(55)
        prompts.addWidget(self.row(self.point_tool,self.box_tool,self.button('Clear prompts',self.clear_prompts)))
        self.candidates=QComboBox();self.candidates.activated.connect(self.choose_candidate);prompts.addWidget(self.candidates)
        self.preview_info=QLabel();self.preview_info.setWordWrap(True);prompts.addWidget(self.preview_info)
        self.padding=self.spin(2,0,50);self.field(prompts,'Padding (px)',self.padding)
        segment=self.section(right,'3D segmentation')
        self.operation=self.combo([('Replace','set'),('Add','add'),('Subtract','subtract'),('Intersect','intersect')]);self.field(segment,'Selection operation',self.operation)
        self.adversarial_background=self.check('Adversarial background',checked=True)
        self.translated(self.adversarial_background,'Choose black or white from the unpadded mask boundary; disable to use the manual background color.','setToolTip')
        segment.addWidget(self.adversarial_background)
        self.segment_button=self.button('3D segmentation',self.segment);segment.addWidget(self.segment_button)
        self.complete_cube=self.button('Cube completion',lambda:self.complete_foreground('cube'))
        self.complete_cylinder=self.button('Cylinder completion',lambda:self.complete_foreground('cylinder'))
        segment.addWidget(self.row(self.complete_cube,self.complete_cylinder))
        self.add_id=self.button('Add ID',self.add_object);segment.addWidget(self.add_id)
        visual=self.section(right,'3D visualization')
        self.bbox=self.check('3D bbox',self.redraw_overlays);self.route=self.check('V-drone route and frustums',self.redraw_overlays)
        visual.addWidget(self.bbox);visual.addWidget(self.route)
        self.show_background=self.check('Show background (ID 0)',self.refresh,True)
        self.show_foreground=self.check('Show object foreground',self.refresh,True)
        self.color_ids=self.check('Color foreground by ID',self.refresh)
        for c in (self.show_background,self.show_foreground,self.color_ids):visual.addWidget(c)
        visual.addWidget(self.row(self.button('Select all',lambda:self.select_display_ids(True)),self.button('Select none',lambda:self.select_display_ids(False))))
        self.id_list=QWidget();self.id_layout=QVBoxLayout(self.id_list);self.id_layout.setContentsMargins(0,0,0,0)
        id_scroll=QScrollArea();id_scroll.setWidgetResizable(True);id_scroll.setWidget(self.id_list);id_scroll.setMinimumHeight(120);id_scroll.setMaximumHeight(230);visual.addWidget(id_scroll)
        self.gallery_button=self.button('V-drone views',self.show_gallery);visual.addWidget(self.gallery_button)
        timing=self.section(right,'Timing');self.timing=QLabel('—');self.timing.setWordWrap(True);timing.addWidget(self.timing)
        rag=self.section(right,'GaussianRAG')
        api=self.section(rag,'API settings')
        self.rag_base=QLineEdit();self.rag_base.setPlaceholderText('https://provider.example/v1');self.field(api,'Base URL',self.rag_base)
        self.rag_model=QLineEdit();self.field(api,'API model',self.rag_model)
        self.rag_key=QLineEdit();self.rag_key.setEchoMode(QLineEdit.Password);self.field(api,'API key (this session only)',self.rag_key)
        note=self.label('Requires Chat Completions and tool calls. Leave blank to use backend environment variables. Keys are not saved.');note.setWordWrap(True);api.addWidget(note)
        self.rag_test=self.button('Test API connection',self.test_rag);api.addWidget(self.rag_test)
        self.rag_describe=self.check('Generate object descriptions with a vision API');rag.addWidget(self.rag_describe)
        note=self.label('Enabling this sends four-view images to the configured API; questions send object descriptions and spatial facts.');note.setWordWrap(True);rag.addWidget(note)
        self.rag_build=self.button('Build / update spatial knowledge',self.build_rag);rag.addWidget(self.rag_build)
        note=self.label('Uses all object IDs and also saves IDs. Distances are in scene units, not necessarily metres.');note.setWordWrap(True);rag.addWidget(note)
        self.rag_question=QLineEdit();self.rag_question.setMaxLength(4000);self.field(rag,'Spatial question',self.rag_question)
        self.rag_question.returnPressed.connect(self.ask_rag);self.rag_question.textChanged.connect(lambda _:self.update_controls())
        self.rag_ask=self.button('Ask about the scene',self.ask_rag);rag.addWidget(self.rag_ask)
        self.rag_answer=QLabel();self.rag_answer.setWordWrap(True);self.rag_answer.setTextFormat(Qt.PlainText);self.rag_answer.setTextInteractionFlags(Qt.TextSelectableByMouse);rag.addWidget(self.rag_answer)
        self.rag_signature=None;self.rag_report=None
        semantic=self.section(right,'Semantic knowledge')
        self.semantic_build=self.button('Build / update semantic knowledge',self.build_semantic)
        semantic.addWidget(self.semantic_build)
        note=self.label('All object IDs · Four views · GIF elevation and auto-fit distance · Also saves IDs');note.setWordWrap(True);semantic.addWidget(note)
        self.semantic_query=QLineEdit();semantic.addWidget(self.semantic_query)
        self.semantic_query.returnPressed.connect(self.search_semantic)
        self.semantic_query.textChanged.connect(lambda _:self.update_controls())
        self.semantic_query.setMaxLength(512)
        self.semantic_search=self.button('Find object IDs',self.search_semantic);semantic.addWidget(self.semantic_search)
        self.semantic_clear=self.button('Clear search highlight',self.clear_search_highlight);semantic.addWidget(self.semantic_clear)
        semantic.addWidget(self.label('Describe an object (English recommended)'))
        self.semantic_results=QLabel();self.semantic_results.setWordWrap(True);self.semantic_results.setTextFormat(Qt.PlainText);self.semantic_results.setTextInteractionFlags(Qt.TextSelectableByMouse);semantic.addWidget(self.semantic_results)
        self.semantic_signature=None;self.semantic_report=None
        exports=self.section(right,'Export')
        for key,kind in (('Export object PLY','ply'),('Export 3D mask','mask'),('Export image','image'),('Export 2D mask','mask2d')):
            exports.addWidget(self.button(key,lambda k=kind:self.export(k)))
        self.mesh_export=self.button('Export object meshes',self.export_mesh)
        exports.addWidget(self.mesh_export)
        self.gif_all=self.button('Export all object GIFs',lambda:self.export_gif('all'))
        self.gif_current=self.button('Export current GIF',lambda:self.export_gif('current'))
        exports.addWidget(self.gif_all);exports.addWidget(self.gif_current)
        exports.addWidget(self.button('View GIFs',self.show_gifs));exports.addWidget(self.button('Open output folder',lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output)))))
        gif=self.section(exports,'GIF settings')
        self.gif_size=self.combo([(str(n),n) for n in (256,512,768)]);self.gif_size.setCurrentIndex(1)
        self.gif_frames=self.combo([(str(n),n) for n in (36,72)]);self.gif_frames.setCurrentIndex(1)
        self.gif_elevation=self.spin(60,-89,89,1)
        self.field(gif,'Resolution',self.gif_size);self.field(gif,'Frames per orbit',self.gif_frames);self.field(gif,'Orbit elevation (degrees)',self.gif_elevation)
        gif.addWidget(self.label('One orbit · 20 FPS · White background'));right.addStretch()
        for panel in (self.left,self.right):
            panel.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            for widget in panel.findChildren(QWidget):
                if isinstance(widget,(QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox,QPushButton)):
                    widget.setMinimumWidth(0)
                    widget.setSizePolicy(QSizePolicy.Ignored,widget.sizePolicy().verticalPolicy())
        self.splitter.setSizes([300,1000,300]);self.splitter.setStretchFactor(1,1)
        self.status=self.statusBar();self.stats=QLabel();self.status.addPermanentWidget(self.stats)
        for key,seq,fn in (('Exit application (Esc)','Esc',self.close),('Open PLY…','Ctrl+O',self.open_file),('Undo','Ctrl+Z',lambda:self.object_action('undo')),('Redo','Ctrl+Shift+Z',lambda:self.object_action('redo'))):
            action=QAction(self);action.setShortcut(QKeySequence(seq));action.triggered.connect(fn);self.addAction(action)
        self.model_dialog=None

    def retranslate(self):
        from shiboken6 import isValid
        self._translations=[item for item in self._translations if isValid(item[0])]
        for widget,key,method in self._translations:getattr(widget,method)(self.text(key))
        for combo in self.findChildren(QComboBox):
            items=combo.property('i18n_items')
            if items:
                for i,(key,value) in enumerate(items):combo.setItemText(i,self.text(key))
        from PySide6.QtCore import QTranslator, QLibraryInfo
        app=QApplication.instance()
        if getattr(self,'qt_translator',None):app.removeTranslator(self.qt_translator)
        self.qt_translator=QTranslator(self)
        if self.tr.language=='zh' and self.qt_translator.load('qtbase_zh_CN',QLibraryInfo.path(QLibraryInfo.TranslationsPath)):
            app.installTranslator(self.qt_translator)
        self.show_semantic_report();self.show_rag_report()
        self.language.setText('中文' if self.tr.language=='en' else 'English')
        self.objects.setHorizontalHeaderLabels(['ID',self.text('Name'),self.text('Hidden')])
        self.canvas.hint=self.text('Open a Gaussian PLY to begin');self.canvas.update();self.update_controls()

    def toggle_language(self):
        self.tr.language='zh' if self.tr.language=='en' else 'en'
        self.settings['language']=self.tr.language;write_config(self.settings);self.retranslate()

    def message(self,text,error=False):
        self.status.showMessage(self.tr.backend_text(text));self.status.setStyleSheet('color:#ff7186' if error else '')

    def call(self,action,data=None,callback=None):
        if self.bridge.busy:return False
        data=dict(data or {})
        interactive=data.pop('_interactive',False)
        if action in ('load','view','frame','capture','release','cancel_round') or action=='ground' and data.get('operation')=='start':
            data['aspect']=max(.25,min(4,self.canvas.width()/max(1,self.canvas.height())))
            data['render_size']=min(768,self.target_render_size()) if interactive else self.target_render_size()
            if interactive:self.quality_timer.start(450)
            else:self.quality_timer.stop()
        if action in ('frame','view','export','release','cancel_round'):
            data.update(render_mode=self.render_mode.currentData(),id_colors=self.color_ids.isChecked(),
                show_background=self.show_background.isChecked(),show_foreground=self.show_foreground.isChecked(),foreground_ids=sorted(self.foreground_ids))
        data['background_color']=[control.value() for control in self.background_rgb]
        self.callback=callback;self.current_data=data;self.started=time.monotonic();self.elapsed_timer.start(100)
        self.progress_bar.setValue(0);self.busy_label.setText(self.text('Working…'))
        self.bridge.submit(action,data);self.update_controls();return True

    def completed(self,action,result):
        if action in ('objects','ground'):self.clear_semantic_report();self.rag_report=None;self.rag_answer.clear()
        callback=self.callback;self.callback=None;self.elapsed_timer.stop()
        self.busy_label.setText(f'{time.monotonic()-self.started:.2f} s');self.progress_bar.setValue(100)
        if self.cancel_round_pending:
            if self.model_dialog:self.model_dialog.close();self.model_dialog=None
            self.finish_cancel_round();return
        if result.get('image') and 'aspect' in self.current_data:
            self._rendered_aspect=self.current_data['aspect']
            self._rendered_size=self.current_data.get('render_size')
        if 'count' in result:self.present(result)
        if result.get('timing'):self.show_timing(result['timing'])
        self.update_controls()
        if callback:
            try:callback(result)
            except Exception as error:
                import traceback
                traceback.print_exc();self.message(str(error),True)
        if getattr(self,'release_pending',False):
            self.release_pending=False;self.call('release',{'mode':self.mode});return
        if self.close_pending:
            QTimer.singleShot(0,self.close);return
        if action=='preview' and self.capture and self.preview_revision!=self.revision:self.preview_timer.start(0)

    def failed(self,action,message):
        self.callback=None;self.elapsed_timer.stop()
        if self.cancel_round_pending:
            if self.model_dialog:self.model_dialog.close();self.model_dialog=None
            self.finish_cancel_round();return
        self.message(message,True)
        if action=='load_model' and self.model_dialog:
            self.model_dialog.download_label.setText(self.tr.backend_text(message))
            self.model_dialog.download_bar.hide();self.model_dialog.download_size.hide()
            self.model_dialog.retry_button.show()
            self.model_dialog.cancel_button.setText(self.text('Close'))
        if action=='apply':self.set_capture(None)
        self.update_controls()
        if getattr(self,'release_pending',False):
            self.release_pending=False;self.call('release',{'mode':self.mode});return
        if self.close_pending:QTimer.singleShot(0,self.close)
        elif action=='preview' and self.capture and self.preview_revision!=self.revision:self.preview_timer.start(0)

    def elapsed(self):
        if self.bridge.busy:self.busy_label.setText(f'{self.progress_message if hasattr(self,"progress_message") else self.text("Working…")} · {time.monotonic()-self.started:.1f} s')

    def progress(self,value,message,views):
        if self.cancel_round_pending:return
        maximum=6 if self.bridge.action=='apply' else 100
        self.progress_bar.setValue(round(value/maximum*100));self.progress_message=self.text(self.tr.backend_text(message))
        if isinstance(views,dict) and views.get('kind') in ('model_download','model_loading'):
            if self.model_dialog:
                dialog=self.model_dialog
                dialog.download_label.setText(self.text(views['stage']))
                if views['kind']=='model_loading':
                    dialog.download_bar.setRange(0,0);dialog.download_size.clear()
                    self.elapsed();return
                dialog.download_bar.setRange(0,100);dialog.download_bar.setValue(value)
                dialog.download_size.setText(f"{views['downloaded']/1024**2:.1f} / {views['total']/1024**2:.1f} MiB")
        elif views:self.drone_views=views
        self.elapsed()

    def present(self,result):
        rag_signature=repr((result.get('knowledge_path'),result.get('objects'),result.get('camera',{}).get('up'),result.get('ids_dirty')))
        if rag_signature!=self.rag_signature:
            self.rag_signature=rag_signature;self.rag_report=None;self.rag_answer.clear()
        signature=repr((result.get('knowledge_path'),result.get('objects'),result.get('camera',{}).get('up'),result.get('ids_dirty')))
        if signature!=self.semantic_signature:
            self.semantic_signature=signature;self.clear_semantic_report()
        self.state={**self.state,**result}
        if result.get('image'):self.canvas.set_image(result['image'])
        if 'visualization' in result:self.canvas.visualization=result['visualization']
        v=result.get('camera',{})
        if v:
            self.fov.setValue(v['fov'])
            for controls,key in ((self.up,'up'),(self.target,'target')):
                for control,value in zip(controls,v[key]):control.setValue(value)
        if 'background_color' in result:
            for control,value in zip(self.background_rgb,result['background_color']):
                control.setValue(value)
        if result.get('render_mode'):
            self.render_mode.setCurrentIndex(self.render_mode.findData(result['render_mode']))
        self.scene_info.setText(f"{result.get('scene','')}\n{result.get('count',0):,} Gaussians")
        self.stats.setText(f"{result.get('selected',0):,} / {result.get('count',0):,}")
        self.ids_info.setText(('● ' if result.get('ids_dirty') else '')+str(result.get('knowledge_path') or ''))
        display=result.get('display',{})
        if display.get('foreground_ids') is not None:
            self.foreground_ids=set(display['foreground_ids'])
            self.known_objects={o['id'] for o in result.get('objects',[])}
        self.refresh_objects();self.update_controls();self.canvas.update()

    def flip_background(self):
        if self.bridge.busy or self.capture or self.ground or self.correction:return
        color=1. if sum(c.value()*w for c,w in zip(self.background_rgb,(.2126,.7152,.0722)))<.5 else 0.
        for control in self.background_rgb:control.setValue(color)
        self.refresh()

    def navigation_allowed(self):
        active=QApplication.focusWidget()
        return bool(self.state.get('count') and self.isActiveWindow() and not QApplication.activeModalWidget()
            and not (QApplication.keyboardModifiers() & (Qt.ControlModifier|Qt.AltModifier|Qt.MetaModifier))
            and not (self.capture or self.ground or self.correction or self.close_pending)
            and not isinstance(active,(QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox)))

    def queue_pan(self,delta):
        if not self.navigation_allowed():return
        self.pending_pan=[a+b for a,b in zip(self.pending_pan,delta)]

    def navigate(self):
        now=time.monotonic()
        if not self.navigation_allowed() or self.bridge.busy and self.bridge.action!='view':
            self.navigation_keys.clear();self.pending_pan=[0.,0.];self.navigation_last=now;return
        if self.bridge.busy:return
        data=None
        if any(self.pending_pan):
            pan=[max(-1.,min(1.,x)) for x in self.pending_pan]
            self.pending_pan=[a-b for a,b in zip(self.pending_pan,pan)]
            data={'pan':pan}
        elif self.navigation_keys:
            x=int(Qt.Key_D in self.navigation_keys)-int(Qt.Key_A in self.navigation_keys)
            y=int(Qt.Key_W in self.navigation_keys)-int(Qt.Key_S in self.navigation_keys)
            length=math.hypot(x,y)
            if length:
                step=.6*min(.1,now-self.navigation_last)
                data={'move':[x/length*step,y/length*step]}
        self.navigation_last=now
        if data:self.call('view',dict(data,mode=self.mode,_interactive=True))

    def keyReleaseEvent(self,event):
        if event.key() in (Qt.Key_W,Qt.Key_A,Qt.Key_S,Qt.Key_D):
            if not event.isAutoRepeat():self.navigation_keys.discard(event.key())
            event.accept();return
        super().keyReleaseEvent(event)

    def changeEvent(self,event):
        if event.type()==QEvent.ActivationChange and not self.isActiveWindow() and hasattr(self,'navigation_keys'):
            self.navigation_keys.clear();self.pending_pan=[0.,0.]
        super().changeEvent(event)

    def choose_background(self):
        color=QColorDialog.getColor(QColor.fromRgbF(*[c.value() for c in self.background_rgb]),self,self.text('Background color'))
        if color.isValid():
            for control,value in zip(self.background_rgb,(color.redF(),color.greenF(),color.blueF())):
                control.setValue(value)
            self.refresh()

    def update_controls(self):
        if self._building:return
        busy=self.bridge.busy;scene=bool(self.state.get('count'))
        editing=bool(self.capture or self.ground or self.correction)
        blocking=busy and self.bridge.action not in ('frame','view','release')
        self.left.setEnabled(not blocking);self.right.setEnabled(not blocking or self.bridge.action=='preview')
        self.canvas.blocked=busy and not (self.capture and self.bridge.action=='preview')
        self.canvas.mode='correction' if self.correction else 'ground' if self.ground else 'prompt' if self.capture else 'camera'
        self.canvas.foreground=self.fg;self.canvas.background=self.bg;self.canvas.box=self.box;self.canvas.ground=self.ground
        self.adversarial_background.setEnabled(scene and not busy and not self.ground and not self.correction)
        for widget in (*self.background_rgb,self.background_picker,self.background_flip):widget.setEnabled(scene and not busy and not editing)
        for widget in (self.display,self.render_mode):widget.setEnabled(scene and not busy and not editing)
        self.correct_button.setEnabled(scene and self.mode=='all' and not self.capture and not self.ground and not busy)
        self.correct_confirm.setVisible(self.correction);self.correct_discard.setVisible(self.correction)
        self.correct_confirm.setEnabled(not busy and bool(self.state.get('preview_count')))
        self.correct_discard.setEnabled(not busy)
        self.prompt_check.setEnabled(scene and not self.ground and not self.correction and (not busy or self.bridge.action=='preview'))
        self.point_tool.setEnabled(bool(self.capture));self.box_tool.setEnabled(bool(self.capture))
        self.segment_button.setEnabled(bool(self.capture and self.masks) and not busy and not self.preview_timer.isActive())
        self.add_id.setEnabled(bool(self.state.get('selected')) and not busy and not editing)
        for button in (self.complete_cube,self.complete_cylinder):button.setEnabled(bool(self.state.get('selected')) and not busy and not editing)
        self.gif_current.setEnabled(bool(self.state.get('selected')) and not busy)
        self.mesh_export.setEnabled(bool(self.state.get('objects')) and not busy)
        self.gif_all.setEnabled(bool(self.state.get('objects')) and not busy)
        semantic_ready=scene and bool(self.state.get('objects')) and not busy and not editing
        self.rag_build.setEnabled(semantic_ready);self.rag_question.setEnabled(semantic_ready);self.rag_describe.setEnabled(semantic_ready)
        self.rag_ask.setEnabled(semantic_ready and bool(self.rag_question.text().strip()))
        self.semantic_build.setEnabled(semantic_ready);self.semantic_query.setEnabled(semantic_ready)
        self.semantic_search.setEnabled(semantic_ready and bool(self.semantic_query.text().strip()))
        self.semantic_clear.setEnabled(bool(self.state.get('semantic_match')) and not busy and not editing)
        self.undo.setEnabled(bool(self.state.get('canUndo')) and not busy and not editing)
        self.redo.setEnabled(bool(self.state.get('canRedo')) and not busy and not editing)
        self.cancel_button.setVisible(busy and self.bridge.action in ('apply','gif','mesh','semantic','rag'))
        self.progress_bar.setVisible(busy)
        self.gallery_button.setEnabled(bool(self.drone_views) and not busy)
        self.hint.setText(self.text('Left: foreground · Right: background · Drag: box')+' · '+self.text('Cancel current round (C)') if self.capture else self.text('Drag to rotate · Right/Shift drag to pan · WASD to move · Scroll to zoom'))

    def open_file(self):
        if self.bridge.busy:return
        path,_=QFileDialog.getOpenFileName(self,self.text('Open PLY…'),'','Gaussian PLY (*.ply)')
        if path:self.load_scene(path)
    def open_folder(self):
        if self.bridge.busy:return
        path=QFileDialog.getExistingDirectory(self,self.text('Open scene folder…'))
        if path:self.load_scene(path)
    def refresh_recent(self):
        self.recent.clear();self.recent.addItem('—',None)
        for path in self.settings.get('recent',[]):self.recent.addItem(path,path)
    def clear_recent(self):
        self.settings['recent']=[];write_config(self.settings);self.refresh_recent()

    def load_scene(self,path):
        if self.bridge.busy or not path:return
        if self.state.get('ids_dirty'):
            answer=QMessageBox.question(self,self.text('Unsaved segmentation'),self.text('Save segmentation before closing?'),QMessageBox.Save|QMessageBox.Discard|QMessageBox.Cancel)
            if answer==QMessageBox.Cancel:return
            if answer==QMessageBox.Save:
                self.object_action('save_ids',after=lambda:self.load_scene(path));return
        self.release();self.ground=None;self.correction=False;self.path.setText(str(path));self.foreground_ids.clear();self.known_objects.clear();self.drone_views=[]
        def loaded(result):
            self.mode='all';self.display.setCurrentIndex(0)
            self.preset.clear();self.preset.addItem('—',None)
            for i,name in enumerate(result.get('cameras',[])):self.preset.addItem(name,i)
            recent=[str(Path(path).expanduser().resolve())]+[p for p in self.settings.get('recent',[]) if p!=str(Path(path).expanduser().resolve())]
            self.settings['recent']=recent[:12];write_config(self.settings);self.refresh_recent()
            # The incoming frame owns the new scene's display options.
            self.message(self.text('Ready'))
        self.call('load',{'path':str(path)},loaded)

    def refresh_objects(self):
        objects=self.state.get('objects',[])
        existing={o['id'] for o in objects}
        self.foreground_ids &= existing
        self.foreground_ids |= existing-self.known_objects
        signature=(tuple((o['id'],o['number'],o['name'],o['hidden'],o.get('count')) for o in objects),frozenset(self.foreground_ids))
        if signature==self._objects_signature:return
        self._objects_signature=signature
        selected=self.selected_object_id()
        merge_a,merge_b=self.merge_a.currentData(),self.merge_b.currentData()
        self.objects.setRowCount(len(objects))
        self.merge_a.clear();self.merge_b.clear()
        while self.id_layout.count():
            item=self.id_layout.takeAt(0)
            if item.widget():item.widget().deleteLater()
        existing={o['id'] for o in objects};self.foreground_ids &= existing
        for row,obj in enumerate(objects):
            if obj['id'] not in self.known_objects:self.foreground_ids.add(obj['id'])
            for col,text in enumerate((str(obj['number']),obj['name'],'●' if obj['hidden'] else '○')):
                item=QTableWidgetItem(text);item.setData(Qt.UserRole,obj['id']);self.objects.setItem(row,col,item)
            if obj['id']==selected:self.objects.selectRow(row)
            for combo in (self.merge_a,self.merge_b):combo.addItem(f"ID {obj['number']} · {obj['name']}",obj['id'])
            check=QCheckBox(f"ID {obj['number']} · {obj['name']}");check.setChecked(obj['id'] in self.foreground_ids)
            check.setMinimumHeight(23);check.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Fixed)
            check.toggled.connect(lambda enabled,key=obj['id']:self.toggle_id(key,enabled));self.id_layout.addWidget(check)
        if len(objects)>1:self.merge_b.setCurrentIndex(1)
        for combo,key in ((self.merge_a,merge_a),(self.merge_b,merge_b)):
            index=combo.findData(key)
            if index>=0:combo.setCurrentIndex(index)
        self.id_list.setMinimumHeight(len(objects)*29)
        self.id_list.updateGeometry()
        self.known_objects=existing
    def selected_object_id(self):
        row=self.objects.currentRow()
        return self.objects.item(row,0).data(Qt.UserRole) if row>=0 and self.objects.item(row,0) else None
    def object_action(self,action,after=None,**data):
        if self.ground:
            self.message(self.text('Exit prompts or calibration first.'),True);return
        if self.bridge.busy or not self.state.get('count'):return
        self.release();self.correction=False
        if action=='delete':
            self.mode='all';self.display.setCurrentIndex(self.display.findData('all'))
        def done(result):
            if after:after()
        self.call('objects',dict(action=action,**data),done)
    def selected_object_action(self,action):
        key=self.selected_object_id()
        if key:self.object_action(action,id=key)
    def rename_object(self):
        key=self.selected_object_id();obj=next((o for o in self.state.get('objects',[]) if o['id']==key),None)
        if not obj:return
        name,ok=QInputDialog.getText(self,self.text('Rename'),self.text('Name'),text=obj['name'])
        if ok and name.strip():self.object_action('rename',id=key,name=name.strip())
    def merge_ids(self):
        a,b=self.merge_a.currentData(),self.merge_b.currentData()
        if not a or not b or a==b:self.message(self.text('Choose two different object IDs.'),True);return
        self.object_action('merge',ids=[a,b])
    def add_object(self):
        def added(result):
            self.object_name.clear();self.operation.setCurrentIndex(0)
            self.mode='all';self.display.setCurrentIndex(0);self.set_prompt_checked(True);self.toggle_prompts()
        self.release();self.call('objects',{'action':'add_id','name':self.object_name.text()},added)

    def toggle_id(self,key,enabled):
        if enabled:self.foreground_ids.add(key)
        else:self.foreground_ids.discard(key)
        self.refresh()
    def select_display_ids(self,enabled):
        self.foreground_ids={o['id'] for o in self.state.get('objects',[])} if enabled else set()
        self.refresh()
    def refresh(self):
        if self._building or not self.state.get('count') or self.bridge.busy:return
        if self.capture or self.ground or self.correction:self.message(self.text('Exit prompts or calibration first.'),True);return
        self.call('frame',{'mode':self.mode})
    def redraw_overlays(self):
        self.canvas.show_bbox=self.bbox.isChecked();self.canvas.show_route=self.route.isChecked();self.canvas.update()
    def change_display(self):self.mode=self.display.currentData();self.refresh()
    def change_render(self):self.refresh()
    def resize_frame(self):
        if not self.state or self.capture or self.ground or self.correction:return
        if self.bridge.busy:self.resize_timer.start(150);return
        aspect=max(.25,min(4,self.canvas.width()/max(1,self.canvas.height())))
        if self._rendered_aspect is None or abs(aspect-self._rendered_aspect)>.002 or self.target_render_size()!=self._rendered_size:self.refresh()
    def target_render_size(self):
        from desktop.render_quality import target_size
        return target_size(self.canvas.width(),self.canvas.height(),self.canvas.devicePixelRatioF(),self.size.currentData())
    def refine_frame(self):
        if self.capture or self.ground or self.correction or not self.state.get('count'):return
        if self.bridge.busy or self.wheel_timer.isActive() or self.canvas.start is not None:
            self.quality_timer.start(150);return
        if self._rendered_size!=self.target_render_size():self.refresh()
    def camera_preset(self,index):
        preset=self.preset.itemData(index)
        if preset is not None and not self.bridge.busy and not (self.capture or self.ground or self.correction):self.call('view',{'preset':preset,'mode':self.mode})
    def update_camera(self):
        if self.capture or self.ground or self.correction:self.message(self.text('Exit prompts or calibration first.'),True);return
        self.call('view',{'fov':self.fov.value(),'size':self.target_render_size(),'up':[w.value() for w in self.up],
            'target':[w.value() for w in self.target],'mode':self.mode})

    def set_prompt_checked(self,value):
        blocker=QSignalBlocker(self.prompt_check);self.prompt_check.setChecked(value)
    def set_capture(self,capture):
        self.capture=capture;self.set_prompt_checked(bool(capture));self.update_controls()
    def release(self):
        self.quality_timer.stop();self.preview_timer.stop();self.revision+=1;self.capture=None;self.fg=[];self.bg=[];self.box=[];self.masks=[]
        self.level=0;self.candidates.clear();self.set_prompt_checked(False)
        self.canvas.image=self.canvas.base;self.update_controls();self.canvas.update()
    def cancel_round(self):
        if not self.state.get('count') or self.ground or self.correction:return
        if self.bridge.busy and self.bridge.action not in ('capture','preview','apply','load_model','frame','view','release'):return
        self.cancel_round_pending=True
        self.release();self.preview_info.clear();self.timing.clear();self.drone_views=[]
        if self.bridge.busy:
            self.bridge.cancel()
            self.message(self.text('Cancelling round after the current GPU step…'))
        else:self.finish_cancel_round()
    def finish_cancel_round(self):
        self.cancel_round_pending=False;self.release_pending=False;self.drone_views=[]
        self.mode='all';self.display.setCurrentIndex(0)
        def done(result):
            self.message(self.text('Round cancelled. Prompts cleared; saved IDs kept.'))
            if self.close_pending:QTimer.singleShot(0,self.close)
        self.call('cancel_round',{'mode':'all'},done)
    def toggle_prompts(self):
        if self.bridge.busy:
            if self.bridge.action=='preview' and not self.prompt_check.isChecked():
                self.release();self.release_pending=True
            return
        if not self.prompt_check.isChecked():self.release();self.call('release',{'mode':self.mode});return
        def captured(result):
            self.set_capture(result['capture']);self.mode=result.get('capture_mode','all')
            self.display.setCurrentIndex(self.display.findData(self.mode));self.render_mode.setCurrentIndex(0)
            self.clear_prompts()
            if self.loaded_model!='sam2':self.load_model()
        self.release();self.call('capture',{},captured)
    def set_tool(self,tool):
        self.canvas.tool=tool;self.point_tool.setChecked(tool=='point');self.box_tool.setChecked(tool=='box')
    def clear_prompts(self):
        self.fg=[];self.bg=[];self.box=[];self.schedule_preview(0)
    def schedule_preview(self,delay=160):
        if self._building or not self.capture:return
        self.revision+=1;self.masks=[];self.candidates.clear();self.canvas.image=self.canvas.base
        self.preview_timer.start(delay);self.update_controls();self.canvas.update()
    def preview(self):
        if not self.capture:return
        if self.bridge.busy:self.preview_timer.start(120);return
        if not (self.fg or self.box):return
        if self.loaded_model!='sam2':
            self.load_model(after=lambda:self.preview_timer.start(0));return
        self.preview_revision=self.revision
        data={'capture':self.capture,'model':'sam2',
            'foreground':[list(p) for p in self.fg],'background':[list(p) for p in self.bg],'box':[list(p) for p in self.box]}
        def candidates(result):
            if self.preview_revision!=self.revision or not self.capture:return
            self.masks=result.get('masks',[]);self.level=0;self.candidates.clear()
            for i,src in enumerate(self.masks):
                labels=result.get('candidate_labels',[]);scores=result.get('scores',[])
                title=self.tr.backend_text(labels[i]) if i<len(labels) else f'{i+1}'
                if i<len(scores) and scores[i] is not None:title+=f' · {scores[i]*100:.0f}%'
                self.candidates.addItem(title,i)
            if self.masks:self.choose_candidate(0)
            else:self.message(self.text('No matching mask. Adjust the prompts.'),True)
            self.preview_info.setText(f'{len(self.masks)} '+('候选' if self.tr.language=='zh' else 'candidates'));self.update_controls()
        self.call('preview',data,candidates)
    def choose_candidate(self,index):
        if 0<=index<len(self.masks):self.level=index;self.canvas.image=decode_image(self.masks[index]);self.canvas.update()
    def load_model(self,after=None):
        if self.bridge.busy or self.model_dialog:return
        version='sam2'
        dialog=QDialog(self);dialog.setWindowTitle(self.text('Loading model…'));layout=QVBoxLayout(dialog)
        label=QLabel(version.upper());label.setWordWrap(True);layout.addWidget(label)
        dialog.download_label=label
        dialog.download_bar=QProgressBar();dialog.download_bar.setRange(0,0);layout.addWidget(dialog.download_bar)
        dialog.download_size=QLabel();layout.addWidget(dialog.download_size)
        def retry():
            dialog.close();self.load_model(after)
        dialog.retry_button=self.button('Retry download',retry);dialog.retry_button.hide()
        dialog.cancel_button=self.button('Cancel',dialog.close)
        layout.addWidget(self.row(dialog.retry_button,dialog.cancel_button))
        dialog.load_succeeded=False
        def closed():
            if dialog.load_succeeded:return
            self.preview_timer.stop()
            if self.model_dialog is dialog:self.model_dialog=None
            if self.bridge.busy and self.bridge.action=='load_model':self.bridge.cancel()
        dialog.finished.connect(closed)
        dialog.setModal(False);dialog.show();self.model_dialog=dialog
        def done(result):
            dialog.load_succeeded=True
            self.loaded_model=version
            label.setText(f"{version.upper()} · {result['model_load_seconds']:.2f} s")
            dialog.download_bar.setRange(0,100);dialog.download_bar.setValue(100)
            dialog.retry_button.hide();dialog.cancel_button.setText(self.text('Close'))
            timer=QTimer(dialog);timer.setSingleShot(True);timer.timeout.connect(dialog.close);timer.start(1200)
            self.model_dialog=None
            if after:after()
        self.call('load_model',{'model':version},done)
    def complete_foreground(self,shape):
        if self.bridge.busy or self.capture or self.ground or self.correction or not self.state.get('selected'):return
        def done(result):
            self.release();self.mode='foreground';self.display.setCurrentIndex(self.display.findData('foreground'))
            self.message(self.text('Completion added {0} Gaussians').format(result['completion']['added']))
        self.call('objects',{'action':'complete','shape':shape},done)

    def segment(self):
        if not self.capture or not self.masks or self.bridge.busy:return
        data={'capture':self.capture,'model':'sam2','level':self.level,'padding':self.padding.value(),'operation':self.operation.currentData(),'adversarial_background':self.adversarial_background.isChecked(),'drone':True}
        self.drone_views=[]
        def done(result):
            self.release();self.mode='foreground';self.display.setCurrentIndex(1)
            if result.get('route'):self.drone_views=result['route']
            self.present(result)
        self.call('apply',data,done)
    def show_timing(self,timing):
        self.timing.setText('\n'.join(f'{key}: {value:.3f} s' for key,value in timing.items() if isinstance(value,(float,int))))

    def ground_action(self,operation,**extra):
        if self.bridge.busy or not self.state.get('count'):return
        if operation=='start':self.release();self.correction=False;self.mode='all';self.display.setCurrentIndex(0)
        def done(result):
            self.ground=result if result.get('ground_capture') else None
            if self.ground:self.canvas.ground=self.ground
            plane=result.get('ground_draft') or result.get('ground')
            self.ground_info.setText(str(plane) if plane else '')
            self.update_controls();self.canvas.update()
        self.call('ground',{'operation':operation,'capture':self.ground.get('ground_capture') if self.ground else None,**extra},done)
    def start_correction(self):
        if self.correction:self.correction=False;self.refresh();return
        dialog=QDialog(self);dialog.setWindowTitle(self.text('Correction'));layout=QVBoxLayout(dialog)
        mode=self.combo([('Ellipse correction','ellipse'),('Point correction','point')]);self.field(layout,'Correction',mode)
        target=self.spin(0,0,2147483647);self.field(layout,'Target ID (0 = background)',target)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel);layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject)
        if dialog.exec()==QDialog.Accepted:
            self.correction=True;self.correction_id=target.value();self.correction_mode=mode.currentData();self.update_controls()
    def confirm_correction(self):
        self.call('objects',{'action':'confirm_correction','preview_id':self.state.get('preview_id')},lambda _:self.finish_correction())
    def finish_correction(self):self.correction=False;self.update_controls()
    def discard_correction(self):
        self.correction=False;self.refresh()
    def gesture(self,start,end,button,shift):
        dx,dy=end[0]-start[0],end[1]-start[1]
        if self.ground:
            if button==0 and abs(dx)+abs(dy)<6:self.ground_action('pick',pixel=end)
            return
        if self.correction:
            if button!=0:return
            data={'action':'preview_correction','target_id':self.correction_id,'correction_mode':self.correction_mode,'frame_id':self.state.get('frame_id')}
            data.update({'pixel':end} if abs(dx)<3 and abs(dy)<3 else {'rectangle':[start,end]})
            self.call('objects',data);return
        if self.capture:
            if self.canvas.tool=='point':(self.bg if button==2 else self.fg).append(end)
            else:self.box=[[min(start[0],end[0]),min(start[1],end[1])],[max(start[0],end[0]),max(start[1],end[1])]] if abs(dx)>2 and abs(dy)>2 else []
            self.schedule_preview();return
        if self.bridge.busy or not self.state.get('camera') or abs(dx)+abs(dy)<2:return
        v=self.state['camera'];data={'mode':self.mode,'_interactive':True}
        if shift or button==2:
            data['pan']=[dx/self.canvas.image.width(),dy/self.canvas.image.height()]
        else:data.update(yaw=v['yaw']+dx/self.canvas.image.width()*180,pitch=max(-85,min(85,v['pitch']+dy/self.canvas.image.height()*120)))
        self.call('view',data)
    def zoom(self,delta):
        if self.state.get('camera') and not self.bridge.busy:
            radius=self.state['camera']['radius']*math.exp(-delta*.001)
            self.zoom_radius=max(.001,min(1e7,radius));self.state['camera']['radius']=self.zoom_radius;self.wheel_timer.start(100)
    def apply_zoom(self):
        if self.bridge.busy:self.wheel_timer.start(100);return
        self.call('view',{'radius':self.zoom_radius,'mode':self.mode,'_interactive':True})

    def local_asset(self,url):
        path=unquote(urlsplit(url).path)
        if path.startswith('/downloads/'):return self.output/Path(path).name
        if path.startswith('/previews/'):
            return self.output/'drone-previews'/Path(path).relative_to('/previews')
        return Path(path)
    def save_asset(self,path,name=None):
        path=Path(path)
        destination,_=QFileDialog.getSaveFileName(self,self.text('Save as…'),str(Path.home()/(name or path.name)))
        if destination and Path(destination).resolve()!=path.resolve():shutil.copy2(path,destination)
    def export(self,kind):
        data={'kind':kind,'mode':self.mode}
        if kind=='mask2d':data.update(capture=self.capture,level=self.level)
        def done(result):
            path=self.local_asset(result['download']);self.message(f"{self.text('Files saved to:')} {path}");self.save_asset(path)
        self.call('export',data,done)
    def rag_api(self):
        return dict(base_url=self.rag_base.text().strip(),model=self.rag_model.text().strip(),key=self.rag_key.text())

    def show_rag_report(self):
        if not self.rag_report:return
        operation,result=self.rag_report
        if operation=='test':text=self.text('API connected: {0}').format(result['model'])
        elif operation=='build':text=self.text('Spatial knowledge saved for {0} objects: {1}').format(result['indexed'],result['saved_path'])
        else:
            note=self.text('Model: {0} · Cited IDs: {1} · {2} s').format(result['model'],', '.join(map(str,result['referenced_ids'])) or '—',f"{result['seconds']:.2f}")
            text=result['answer']+'\n\n'+note
        self.rag_answer.setText(text)

    def rag_done(self,operation,result):
        self.rag_report=(operation,result);self.show_rag_report()

    def test_rag(self):
        self.call('rag',dict(operation='test',api=self.rag_api()),lambda r:self.rag_done('test',r))

    def build_rag(self):
        def done(result):
            self.state['ids_dirty']=False;self.ids_info.setText(result['knowledge_path'])
            self.rag_signature=repr((self.state.get('knowledge_path'),self.state.get('objects'),self.state.get('camera',{}).get('up'),False))
            self.rag_done('build',result)
        self.call('rag',dict(operation='build',describe=self.rag_describe.isChecked(),api=self.rag_api(),elevation_degrees=self.gif_elevation.value(),size=self.gif_size.currentData()),done)

    def ask_rag(self):
        if not self.rag_ask.isEnabled():return
        self.rag_report=None;self.rag_answer.clear()
        self.call('rag',dict(operation='ask',question=self.rag_question.text(),api=self.rag_api()),lambda r:self.rag_done('ask',r))

    def clear_semantic_report(self):
        self.semantic_report=None;self.semantic_results.clear()

    def show_semantic_report(self):
        if self.semantic_report is None:return
        kind,result=self.semantic_report
        if kind=='build':
            text=self.text('Indexed {0} objects; updated {1}. Saved to {2}').format(result['indexed'],result['updated'],result['saved_path'])
        else:
            rows=[f"ID {item['number']} · {item['name']} · {item['score']:.3f}" for item in result['results']]
            note=self.text('Rebuild the index for IDs: {0}').format(', '.join(map(str,result['stale_ids']))) if result['stale_ids'] else self.text('Ranked by maximum four-view cosine similarity; scores are not probabilities')
            text='\n'.join(rows+['',note])
        self.semantic_results.setText(text)

    def build_semantic(self):
        def done(result):
            self.state['ids_dirty']=False;self.ids_info.setText(result['saved_path'])
            self.semantic_signature=repr((self.state.get('knowledge_path'),self.state.get('objects'),self.state.get('camera',{}).get('up'),False))
            self.semantic_report=('build',result);self.show_semantic_report()
        self.call('semantic',dict(operation='build',elevation_degrees=self.gif_elevation.value(),size=self.gif_size.currentData()),done)

    def search_semantic(self):
        if not self.semantic_search.isEnabled():return
        self.clear_semantic_report()
        def done(result):
            self.mode='all';self.display.setCurrentIndex(self.display.findData('all'))
            self.bbox.setChecked(True);self.redraw_overlays()
            self.semantic_report=('search',result);self.show_semantic_report()
        self.call('semantic',dict(operation='search',text=self.semantic_query.text()),done)

    def clear_search_highlight(self):
        self.call('semantic',dict(operation='clear'))

    def export_gif(self,source):
        def done(result):
            self.gif_items=result['items'];self.show_gifs()
        self.call('gif',{'source':source,'size':self.gif_size.currentData(),'frames_per_orbit':self.gif_frames.currentData(),
            'elevation_degrees':self.gif_elevation.value(),'fps':20},done)
    def show_gifs(self):
        if not self.gif_items:return
        dialog=QDialog(self);dialog.setWindowTitle('GIF');dialog.resize(620,700);layout=QVBoxLayout(dialog)
        combo=QComboBox();layout.addWidget(combo);image=QLabel();image.setAlignment(Qt.AlignCenter);layout.addWidget(image,1)
        info=QLabel();info.setWordWrap(True);layout.addWidget(info)
        movies=[]
        for item in self.gif_items:combo.addItem(item.get('filename','Current foreground'))
        def select(index):
            if movies:movies.pop().stop()
            item=self.gif_items[index];path=self.local_asset(item['download']);movie=QMovie(str(path));movies.append(movie)
            movie.setScaledSize(QSize(512,512));image.setMovie(movie);movie.start()
            info.setText(f"{item['frames']} frames · {item['fps']} FPS · {item['duration_seconds']:.1f} s\n{path}")
        combo.currentIndexChanged.connect(select)
        layout.addWidget(self.button('Save as…',lambda:self.save_asset(self.local_asset(self.gif_items[combo.currentIndex()]['download']),self.gif_items[combo.currentIndex()].get('filename'))))
        layout.addWidget(self.button('Close',dialog.close));select(0);dialog.exec()
        for movie in movies:movie.stop()
    def show_gallery(self):
        if not self.drone_views:return
        dialog=QDialog(self);dialog.setWindowTitle(self.text('V-drone views'));dialog.resize(1000,700)
        layout=QVBoxLayout(dialog);tabs=QTabWidget();layout.addWidget(tabs)
        for i,view in enumerate(self.drone_views):
            page=QWidget();v=QVBoxLayout(page);assets=view
            for key,title in (('render','Render'),('overlay','Overlay'),('mask','Binary mask')):
                url=assets.get(key)
                if not url:continue
                path=self.local_asset(url);label=QLabel();label.setAlignment(Qt.AlignCenter)
                label.setPixmap(QPixmap(str(path)).scaled(700,260,Qt.KeepAspectRatio,Qt.SmoothTransformation));v.addWidget(label)
                v.addWidget(self.button(title+' · Save as…',lambda p=path:self.save_asset(p)))
            info=QLabel(str({k:value for k,value in view.items() if k not in ('render','overlay','mask')}));info.setWordWrap(True);v.addWidget(info)
            scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(page);tabs.addTab(scroll,f's{i+1}')
        dialog.exec()

    def preferences(self):
        if self.bridge.busy:return
        dialog=QDialog(self);dialog.setWindowTitle(self.text('Preferences…'));layout=QVBoxLayout(dialog)
        fields={}
        for key,title,default in (('sam2_checkpoint','SAM2 checkpoint',DEFAULT_CHECKPOINT),('output','Output folder',self.output)):
            edit=QLineEdit(str(self.settings.get(key) or default));fields[key]=edit
            def browse(e=edit,directory=key=='output'):
                path=QFileDialog.getExistingDirectory(dialog,title,e.text()) if directory else QFileDialog.getOpenFileName(dialog,title,e.text())[0]
                if path:e.setText(path)
            self.field(layout,title,self.row(edit,self.button('…',browse)))
        layout.addWidget(self.label('Settings take effect after restarting the application.'))
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel);layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject)
        if dialog.exec()==QDialog.Accepted:
            for key,edit in fields.items():self.settings[key]=edit.text().strip()
            write_config(self.settings)
    def help(self):
        QMessageBox.information(self,'SAGO Desktop',self.text('Native Linux Gaussian segmentation')+'\n\n'+
            self.text('Drag to rotate · Right/Shift drag to pan · WASD to move · Scroll to zoom')+'\n'+
            self.text('Enter segmentation prompts (M)')+'\n'+self.text('Left: foreground · Right: background · Drag: box')+'\n'+
            self.text('3D segmentation')+' (P)\n'+self.text('Cancel current round (C)')+'\n'+self.text('Exit application (Esc)')+'\nCtrl+O · Ctrl+Z · Ctrl+Shift+Z\n\n'+
            self.text('Files saved to:')+' '+str(self.output))
    def keyPressEvent(self,event):
        active=QApplication.focusWidget()
        if isinstance(active,(QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox)):return super().keyPressEvent(event)
        if event.modifiers() & (Qt.ControlModifier|Qt.AltModifier|Qt.MetaModifier):return super().keyPressEvent(event)
        if event.isAutoRepeat():return
        if event.key() in (Qt.Key_W,Qt.Key_A,Qt.Key_S,Qt.Key_D):
            if self.navigation_allowed():
                if not self.navigation_keys:self.navigation_last=time.monotonic()
                self.navigation_keys.add(event.key())
            event.accept()
        elif event.key()==Qt.Key_M and self.prompt_check.isEnabled():self.prompt_check.click();event.accept()
        elif event.key()==Qt.Key_P and self.segment_button.isEnabled():self.segment();event.accept()
        elif event.key()==Qt.Key_C:self.cancel_round();event.accept()
        elif event.key()==Qt.Key_Escape:
            self.close();event.accept()
        else:super().keyPressEvent(event)
    def closeEvent(self,event):
        if not self.bridge.thread.isRunning():event.accept();return
        event.ignore()
        if self.bridge.busy:
            if not self.close_pending and QMessageBox.question(self,self.text('Wait for current task'),self.text('Cancel task and close after it finishes?'))==QMessageBox.Yes:
                self.close_pending=True;self.bridge.cancel()
            return
        self.close_pending=False
        if self.state.get('ids_dirty'):
            answer=QMessageBox.question(self,self.text('Unsaved segmentation'),self.text('Save segmentation before closing?'),QMessageBox.Save|QMessageBox.Discard|QMessageBox.Cancel)
            if answer==QMessageBox.Cancel:return
            if answer==QMessageBox.Save:
                self.object_action('save_ids',after=lambda:QTimer.singleShot(0,self.close));return
        self.navigation_timer.stop();self.navigation_keys.clear();self.quality_timer.stop();self.preview_timer.stop();self.resize_timer.stop();self.wheel_timer.stop();self.bridge.close()

    def export_mesh(self):
        if self.bridge.busy or not self.state.get('objects'):return
        dialog=QDialog(self);dialog.setWindowTitle(self.text('Export object meshes'));dialog.resize(540,760)
        layout=QVBoxLayout(dialog)
        note=self.label('Choose objects; export one OBJ per ID. Selected objects are reconstructed in parallel; this takes longer than segmentation.');note.setWordWrap(True);layout.addWidget(note)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setMinimumHeight(120);scroll.setMaximumHeight(220)
        content=QWidget();choices=QVBoxLayout(content);checks=[]
        for obj in self.state['objects']:
            check=QCheckBox(f"ID {obj['number']} · {obj['name']}");check.setChecked(obj['id'] in self.foreground_ids)
            choices.addWidget(check);checks.append((obj['id'],check))
        scroll.setWidget(content);layout.addWidget(scroll)
        row=QHBoxLayout();row.addWidget(self.button('Select all',lambda:[c.setChecked(True) for _,c in checks]));row.addWidget(self.button('Select none',lambda:[c.setChecked(False) for _,c in checks]));layout.addLayout(row)
        saved=self.settings.get('mesh_export',{})
        root=QLineEdit(saved.get('neus2_root',''));root.setPlaceholderText('Auto')
        python=QLineEdit(saved.get('neus2_python',''));python.setPlaceholderText('Auto')
        network=QLineEdit(saved.get('network','workspace_mask002.json'))
        steps=self.spin(saved.get('steps',15000),100,100000);steps.setSingleStep(100)
        resolution=self.combo([(str(n),n) for n in (128,256,512,1024)]);resolution.setCurrentIndex(max(0,resolution.findData(saved.get('resolution',512))))
        size=self.combo([(str(n),n) for n in (256,512,768)]);size.setCurrentIndex(max(0,size.findData(saved.get('size',512))))
        views=self.spin(saved.get('views_per_orbit',72),8,144)
        parallel=self.spin(saved.get('parallel_jobs',2),1,4)
        for key,widget in [('Mesh exporting directory',root),('Mesh exporting Python executable',python),('Mesh exporting network configuration',network),('Training steps',steps),('Mesh resolution',resolution),('View resolution',size),('Views per orbit',views),('Parallel exports',parallel)]:self.field(layout,key,widget)
        note=self.label('Four orbits at 0°, 20°, 30°, 60°. Leave paths blank to use environment variables or automatic discovery.');note.setWordWrap(True);layout.addWidget(note)
        layout.addWidget(self.label('Each parallel job uses its own GPU memory'))
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel);layout.addWidget(buttons)
        buttons.rejected.connect(dialog.reject)
        def accept():
            if not any(c.isChecked() for _,c in checks):
                QMessageBox.information(dialog,self.text('Export object meshes'),self.text('Select at least one foreground object'));return
            dialog.accept()
        buttons.accepted.connect(accept)
        if dialog.exec()!=QDialog.Accepted:return
        data=dict(neus2_root=root.text().strip(),neus2_python=python.text().strip(),network=network.text().strip(),steps=steps.value(),resolution=resolution.currentData(),size=size.currentData(),views_per_orbit=views.value(),parallel_jobs=parallel.value())
        self.settings['mesh_export']=data.copy();write_config(self.settings)
        data['ids']=[key for key,c in checks if c.isChecked()]
        def done(result):
            report=QDialog(self);report.setWindowTitle(self.text('Mesh export complete'));report.resize(600,440);body=QVBoxLayout(report)
            body.addWidget(QLabel(self.text('Exported {0} meshes · {1} s').format(len(result['items']),f"{result['generation_seconds']:.2f}")))
            scroller=QScrollArea();scroller.setWidgetResizable(True);content=QWidget();rows=QVBoxLayout(content)
            for item in result['items']:
                path=Path(item['saved_path']);label=QLabel(f"{item['filename']}\n"+self.text('{0} vertices · {1} faces').format(item['vertices'],item['faces'])+f"\n{path}")
                label.setWordWrap(True);label.setTextFormat(Qt.PlainText);label.setTextInteractionFlags(Qt.TextSelectableByMouse);rows.addWidget(label)
                rows.addWidget(self.button('Save as…',lambda checked=False,p=path,n=item['filename']:self.save_asset(p,n)))
            scroller.setWidget(content);body.addWidget(scroller)
            body.addWidget(self.button('Open output folder',lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(result['output_directory']))))
            close=QDialogButtonBox(QDialogButtonBox.Close);close.rejected.connect(report.reject);body.addWidget(close);report.exec()
        self.call('mesh',data,done)
