"""Native Qt workflows using deterministic direct-engine calls, no browser/server."""
import base64
import copy
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtCore import Qt, QPointF, QBuffer, QIODevice
from PySide6.QtGui import QImage, QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from desktop.canvas import Viewport, project_segment
from desktop.window import MainWindow
from desktop.bridge import EngineBridge

APP=QApplication.instance() or QApplication([])

def spin_until(predicate,timeout=5000):
    end=time.monotonic()+timeout/1000
    while time.monotonic()<end:
        APP.processEvents();QTest.qWait(10)
        if predicate():return
    raise AssertionError('Qt event loop timed out')

def image_data():
    image=QImage(300,200,QImage.Format_RGB32);image.fill(QColor('#702030'))
    buf=QBuffer();buf.open(QIODevice.WriteOnly);image.save(buf,'PNG')
    return 'data:image/png;base64,'+base64.b64encode(bytes(buf.data())).decode()


class FakeEngine:
    def __init__(self):
        self.calls=[];self.preview_delay=.01;self.apply_delay=0;self.previous_selection=None;self.ignore_cancel=False;self.load_delay=0
        self.state=dict(scene='test.ply',count=100,selected=0,objects=[],camera=dict(yaw=0,pitch=10,radius=2,
            target=[0,0,0],up=[0,0,1],size=768,fov=50,aspect=1.5),cameras=['cam0'],knowledge_path='/tmp/scene/SAGO_knowledge.json',
            ids_dirty=False,canUndo=False,canRedo=False,next_object_id=1,render_mode='rgb',visualization={})
    def frame(self):return dict(copy.deepcopy(self.state),image=image_data(),frame_id='frame')
    def execute(self,action,data,progress,cancel):
        self.calls.append((action,copy.deepcopy(data)))
        if action=='load_model':
            time.sleep(self.load_delay);return dict(model=data['model'],model_load_seconds=.05)
        if action=='capture':return dict(self.frame(),capture='cap',capture_mode='all')
        if action=='preview':
            time.sleep(self.preview_delay)
            return dict(masks=[image_data(),image_data()],scores=[None,.8],model=data['model'],timing={'sam_seconds':.01})
        if action=='cancel_round':
            if self.previous_selection is not None:
                self.state['selected']=self.previous_selection;self.previous_selection=None
            return self.frame()
        if action=='apply':
            self.previous_selection=self.state['selected']
            deadline=time.monotonic()+self.apply_delay
            while time.monotonic()<deadline:
                if cancel.is_set() and not self.ignore_cancel:raise ValueError('cancelled')
                time.sleep(.005)
            self.state['selected']=20;return dict(self.frame(),timing={'refine_seconds':.1})
        if action=='objects':
            op=data['action']
            if op=='complete':
                self.state['selected']=25
                return dict(self.frame(),completion=dict(added=5,selected=25))
            if op in ('save','add_id'):
                self.state['objects'].append(dict(id='obj',number=1,name=data.get('name') or 'Object',hidden=op=='add_id',count=20))
                self.state['selected']=0 if op=='add_id' else 20;self.state['ids_dirty']=True
            if op=='preview_correction':return dict(self.frame(),preview_id='preview',preview_count=3,target_id=data['target_id'])
            if op=='confirm_correction':
                assert data['preview_id']=='preview';self.state['ids_dirty']=True
            if op=='delete':
                self.state['objects']=[o for o in self.state['objects'] if o['id']!=data['id']]
                self.state['selected']=0;self.state['ids_dirty']=True
            if op=='save_ids':self.state['ids_dirty']=False
            if op=='rename':self.state['objects'][0]['name']=data['name']
            return self.frame()
        if action=='semantic':
            if data['operation']=='build':return dict(indexed=1,updated=1,saved_path='/tmp/SAGO_knowledge.json')
            if data['operation']=='clear':
                self.state['semantic_match']=None;return self.frame()
            self.state['semantic_match']='obj'
            return dict(self.frame(),results=[dict(id='obj',number=1,name='Chair',score=.345)],stale_ids=[])
        if action=='gif':return {'items':[]}
        if action=='ground':
            if data['operation'] in ('apply','cancel'):return self.frame()
            return dict(self.frame(),ground_capture='ground',ground_points=[],ground_pixels=[],ground_draft=None)
        if action=='view':
            for key in self.state['camera']:
                if key in data:self.state['camera'][key]=data[key]
        if action=='frame':self.state['render_mode']=data.get('render_mode','rgb')
        return self.frame()


class NativeTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.engine=FakeEngine()
        self.settings_patch=patch('desktop.window.read_config',return_value={});self.settings_patch.start()
        self.write_patch=patch('desktop.window.write_config');self.write_patch.start()
        self.window=MainWindow(lambda:self.engine,self.temp.name);self.window.show()
        self.window.load_scene('/tmp/example.ply');spin_until(lambda:bool(self.window.state))
    def tearDown(self):
        self.window.state['ids_dirty']=False
        if self.window.bridge.busy:spin_until(lambda:not self.window.bridge.busy)
        self.window.close();spin_until(lambda:not self.window.bridge.thread.isRunning())
        self.window.deleteLater();APP.processEvents()
        self.settings_patch.stop();self.write_patch.stop();self.temp.cleanup()
    def test_semantic_build_search_and_edit_locks(self):
        w=self.window;spin_until(lambda:not w.bridge.busy)
        self.assertFalse(w.semantic_build.isEnabled())
        self.engine.state['objects']=[dict(id='obj',number=1,name='Chair',hidden=True,count=20)]
        w.present(self.engine.frame());w.update_controls()
        self.assertTrue(w.semantic_build.isEnabled());self.assertFalse(w.semantic_search.isEnabled())
        w.gif_elevation.setValue(35);w.semantic_build.click();spin_until(lambda:not w.bridge.busy)
        action,data=self.engine.calls[-1]
        self.assertEqual(action,'semantic');self.assertEqual(data['operation'],'build');self.assertEqual(data['elevation_degrees'],35)
        self.assertIn('Indexed 1 objects',w.semantic_results.text())
        w.semantic_query.setText('a chair');self.assertTrue(w.semantic_search.isEnabled())
        QTest.keyClick(w.semantic_query,Qt.Key_Return);spin_until(lambda:not w.bridge.busy)
        self.assertEqual(self.engine.calls[-1][1]['text'],'a chair')
        self.assertIn('ID 1 · Chair · 0.345',w.semantic_results.text())
        self.assertTrue(w.bbox.isChecked());self.assertTrue(w.canvas.show_bbox)
        self.assertEqual(w.mode,'all');self.assertTrue(w.semantic_clear.isEnabled())
        self.assertIn('not probabilities',w.semantic_results.text())
        w.toggle_language();self.assertIn('分数不是概率',w.semantic_results.text());w.toggle_language()
        w.semantic_clear.click();spin_until(lambda:not w.bridge.busy)
        self.assertEqual(self.engine.calls[-1][1]['operation'],'clear');self.assertFalse(w.semantic_clear.isEnabled())
        self.prompt_mode();self.assertFalse(w.semantic_build.isEnabled());self.assertFalse(w.semantic_search.isEnabled())
        w.tr.language='zh'
        self.assertEqual(w.tr.backend_text('Indexed objects: 1/2'),'已索引物体：1/2')

    def prompt_mode(self):
        self.window.prompt_check.click()
        spin_until(lambda:self.window.capture and self.window.loaded_model=='sam2' and not self.window.bridge.busy)
    def test_background_controls_default_black_send_rgb_and_lock_during_prompts(self):
        w=self.window
        spin_until(lambda:not w.bridge.busy)
        self.assertEqual([c.value() for c in w.background_rgb],[0,0,0])
        for control,value in zip(w.background_rgb,[0.125,0.25,0.5]):control.setValue(value)
        w.background_rgb[-1].editingFinished.emit()
        spin_until(lambda:not w.bridge.busy)
        self.assertEqual(self.engine.calls[-1][1]['background_color'],[0.125,0.25,0.5])
        self.prompt_mode()
        self.assertTrue(all(not c.isEnabled() for c in w.background_rgb))
        self.assertFalse(w.background_picker.isEnabled())

    def test_idle_and_render_status_do_not_trigger_resize_loop(self):
        w=self.window
        QTest.qWait(500);spin_until(lambda:not w.bridge.busy)
        before=sum(a=='frame' for a,d in self.engine.calls)
        size=(w.canvas.width(),w.canvas.height())
        QTest.qWait(1100)
        self.assertEqual(sum(a=='frame' for a,d in self.engine.calls),before)
        w.refresh();spin_until(lambda:not w.bridge.busy)
        self.assertEqual((w.canvas.width(),w.canvas.height()),size)
        QTest.qWait(700)
        self.assertEqual(sum(a=='frame' for a,d in self.engine.calls),before+1)
        w.resize(w.width()+150,w.height())
        spin_until(lambda:sum(a=='frame' for a,d in self.engine.calls)>before+1)
        spin_until(lambda:not w.bridge.busy)
        QTest.qWait(700)
        self.assertEqual(sum(a=='frame' for a,d in self.engine.calls),before+2)

    def test_render_auto_dpi_and_manual_choice_survive_frames(self):
        w=self.window
        self.assertIsNone(w.size.currentData())
        with patch.object(w.canvas,'devicePixelRatioF',return_value=2):
            self.assertGreaterEqual(w.target_render_size(),2*max(w.canvas.width(),w.canvas.height()))
        w.size.setCurrentIndex(w.size.findData(1920));w.refresh();spin_until(lambda:not w.bridge.busy)
        self.assertEqual(self.engine.calls[-1][1]['render_size'],1920)
        self.assertEqual(w.size.currentData(),1920)
    def test_navigation_renders_preview_then_one_sharp_frame(self):
        w=self.window
        QTest.qWait(400);spin_until(lambda:not w.bridge.busy)
        w.size.setCurrentIndex(w.size.findData(1920))
        before=len(self.engine.calls)
        w.gesture([10,10],[50,20],0,False)
        spin_until(lambda:not w.bridge.busy)
        self.assertEqual(self.engine.calls[before][1]['render_size'],768)
        spin_until(lambda:any(a=='frame' and d.get('render_size')==1920 for a,d in self.engine.calls[before:]))
        spin_until(lambda:not w.bridge.busy)
        count=len(self.engine.calls);QTest.qWait(700)
        self.assertEqual(len(self.engine.calls),count)
        self.assertEqual(w.size.currentData(),1920)
    def test_prompt_capture_uses_full_quality_and_stays_frozen(self):
        w=self.window;w.size.setCurrentIndex(w.size.findData(1920))
        self.prompt_mode()
        data=next(d for a,d in reversed(self.engine.calls) if a=='capture')
        self.assertEqual(data['render_size'],1920)
        before=len(self.engine.calls)
        w.quality_timer.start(0);QTest.qWait(500)
        self.assertEqual(len(self.engine.calls),before)
        self.assertTrue(w.capture)
    def test_proportional_resize_updates_auto_pixel_count(self):
        w=self.window;QTest.qWait(400);spin_until(lambda:not w.bridge.busy)
        previous=w._rendered_size
        w.resize(w.width()+250,w.height()+180)
        spin_until(lambda:w._rendered_size!=previous)
        spin_until(lambda:not w.bridge.busy)
        self.assertEqual(w._rendered_size,w.target_render_size())

    def test_camera_frames_preserve_object_widgets_and_merge_choices(self):
        w=self.window
        self.engine.state['objects']=[dict(id='a',number=1,name='A',hidden=False,count=10),
            dict(id='b',number=2,name='B',hidden=False,count=20)]
        w.present(self.engine.frame())
        checkbox=w.id_layout.itemAt(0).widget();table_item=w.objects.item(0,1)
        w.merge_a.setCurrentIndex(1);w.merge_b.setCurrentIndex(0)
        w.refresh();spin_until(lambda:not w.bridge.busy)
        self.assertIs(w.id_layout.itemAt(0).widget(),checkbox)
        self.assertIs(w.objects.item(0,1),table_item)
        self.assertEqual(w.merge_a.currentData(),'b');self.assertEqual(w.merge_b.currentData(),'a')
        self.engine.state['objects'][0]['name']='Updated'
        w.present(self.engine.frame())
        self.assertEqual(w.objects.item(0,1).text(),'Updated')

    def test_delete_button_uses_delete_action_and_returns_to_all(self):
        w=self.window
        spin_until(lambda:not w.bridge.busy)
        self.engine.state['objects']=[dict(id='obj',number=1,name='Object',hidden=True,count=20)]
        self.engine.state['selected']=20
        w.present(self.engine.frame());w.objects.selectRow(0)
        w.mode='foreground';w.display.setCurrentIndex(w.display.findData('foreground'))
        w.delete_object.click()
        spin_until(lambda:not w.bridge.busy)
        self.assertTrue(any(a=='objects' and d.get('action')=='delete' and d.get('id')=='obj' for a,d in self.engine.calls))
        self.assertFalse(any(d.get('action')=='remove' for a,d in self.engine.calls))
        self.assertEqual(w.objects.rowCount(),0)
        self.assertEqual(w.state['selected'],0)
        self.assertEqual(w.mode,'all')
        self.assertTrue(w.state['ids_dirty'])

    def test_adversarial_background_defaults_on_and_can_be_disabled_before_refinement(self):
        w=self.window
        self.assertTrue(w.adversarial_background.isChecked())
        self.prompt_mode()
        w.gesture([10,10],[10,10],0,False)
        spin_until(lambda:bool(w.masks) and not w.bridge.busy)
        self.assertTrue(w.adversarial_background.isEnabled())
        w.adversarial_background.setChecked(False)
        w.segment();spin_until(lambda:w.state.get('selected')==20 and not w.capture)
        data=[data for action,data in self.engine.calls if action=='apply'][-1]
        self.assertIs(data['adversarial_background'],False)

    def test_default_language_model_lazy_load_and_auto_prompt(self):
        w=self.window
        self.assertEqual(w.prompt_check.text(),'Enter segmentation prompts (M)')
        self.assertFalse(any(a=='load_model' for a,d in self.engine.calls))
        self.assertEqual(w.padding.value(),2)
        self.assertFalse(hasattr(w,'text_prompt'));self.assertFalse(hasattr(w,'model'))
        self.prompt_mode()
        self.assertTrue(any(a=='load_model' and d['model']=='sam2' for a,d in self.engine.calls))
        w.gesture([10,10],[10,10],0,False)
        spin_until(lambda:bool(w.masks));self.assertEqual(w.candidates.count(),2)
        self.assertEqual(self.engine.calls[-1][1]['foreground'],[[10,10]])
        w.choose_candidate(1);self.assertEqual(w.level,1)
        w.gesture([20,20],[20,20],0,False)
        spin_until(lambda:any(a=='preview' and len(d.get('foreground',[]))==2 for a,d in self.engine.calls))
        spin_until(lambda:not w.bridge.busy)
        w.segment();spin_until(lambda:w.state.get('selected')==20 and not w.capture)
        self.assertEqual(w.mode,'foreground')
        self.assertEqual(next(d for a,d in reversed(self.engine.calls) if a=='apply')['level'],0)
    def test_model_download_progress_does_not_replace_drone_views(self):
        w=self.window;self.engine.load_delay=.2
        w.load_model();dialog=w.model_dialog
        w.drone_views=['existing']
        w.progress(50,'Downloading SAM2 checkpoint…',dict(kind='model_download',stage='Downloading SAM2 checkpoint…',downloaded=1024**2,total=2*1024**2))
        self.assertEqual(dialog.download_bar.value(),50)
        self.assertEqual(dialog.download_size.text(),'1.0 / 2.0 MiB')
        self.assertEqual(w.drone_views,['existing'])
        w.progress(0,'Loading SAM2…',dict(kind='model_loading',stage='Loading SAM2…'))
        self.assertEqual(dialog.download_bar.maximum(),0)
        spin_until(lambda:not w.bridge.busy)
        w.preview_timer.start(5000)
        spin_until(lambda:not dialog.isVisible(),2500)
        self.assertTrue(w.preview_timer.isActive())
        w.preview_timer.stop()

    def test_failed_model_download_retry_and_auto_close(self):
        w=self.window;execute=self.engine.execute
        def failing(action,data,progress,cancel):
            if action=='load_model':raise ValueError('Download failed')
            return execute(action,data,progress,cancel)
        self.engine.execute=failing
        w.load_model();dialog=w.model_dialog
        spin_until(lambda:not w.bridge.busy)
        self.assertTrue(dialog.retry_button.isVisible())
        self.assertIn('Download failed',dialog.download_label.text())
        self.engine.execute=execute;dialog.retry_button.click()
        spin_until(lambda:w.loaded_model=='sam2' and not w.bridge.busy)
        self.assertFalse(dialog.isVisible());self.assertIsNone(w.model_dialog)

    def test_black_white_background_toggle_and_prompt_lock(self):
        w=self.window;spin_until(lambda:not w.bridge.busy)
        w.background_flip.click();spin_until(lambda:not w.bridge.busy)
        self.assertEqual([c.value() for c in w.background_rgb],[1,1,1])
        self.assertEqual(self.engine.calls[-1][1]['background_color'],[1,1,1])
        w.background_flip.click();spin_until(lambda:not w.bridge.busy)
        self.assertEqual([c.value() for c in w.background_rgb],[0,0,0])
        self.prompt_mode();self.assertFalse(w.background_flip.isEnabled())

    def test_held_wasd_stops_on_release_and_text_input_does_not_navigate(self):
        w=self.window;spin_until(lambda:not w.bridge.busy);w.canvas.setFocus()
        QTest.keyPress(w.canvas,Qt.Key_W)
        spin_until(lambda:any(a=='view' and 'move' in d for a,d in self.engine.calls))
        QTest.keyRelease(w.canvas,Qt.Key_W);spin_until(lambda:not w.bridge.busy)
        count=sum(a=='view' and 'move' in d for a,d in self.engine.calls)
        QTest.qWait(150)
        self.assertEqual(sum(a=='view' and 'move' in d for a,d in self.engine.calls),count)
        data=next(d for a,d in self.engine.calls if a=='view' and 'move' in d)
        self.assertGreater(data['move'][1],0);self.assertEqual(data['move'][0],0)
        w.object_name.setFocus();QTest.keyClicks(w.object_name,'wasd');QTest.qWait(100)
        self.assertEqual(w.object_name.text(),'wasd')
        self.assertEqual(sum(a=='view' and 'move' in d for a,d in self.engine.calls),count)
        w.quality_timer.stop();spin_until(lambda:not w.bridge.busy)
        self.prompt_mode();w.canvas.setFocus();QTest.keyClick(w.canvas,Qt.Key_D);QTest.qWait(100)
        self.assertEqual(sum(a=='view' and 'move' in d for a,d in self.engine.calls),count)

    def test_right_drag_pans_before_release_and_prompt_right_click_stays_background(self):
        from PySide6.QtCore import QPoint
        w=self.window;spin_until(lambda:not w.bridge.busy)
        center=w.canvas.rect().center()
        QTest.mousePress(w.canvas,Qt.RightButton,pos=center)
        QTest.mouseMove(w.canvas,center+QPoint(50,25))
        spin_until(lambda:any(a=='view' and 'pan' in d for a,d in self.engine.calls))
        self.assertIsNotNone(w.canvas.start)
        QTest.mouseRelease(w.canvas,Qt.RightButton,pos=center+QPoint(50,25))
        spin_until(lambda:not w.bridge.busy)
        data=next(d for a,d in self.engine.calls if a=='view' and 'pan' in d)
        self.assertGreater(data['pan'][0],0);self.assertGreater(data['pan'][1],0)
        self.prompt_mode()
        QTest.mouseClick(w.canvas,Qt.RightButton,pos=center)
        self.assertEqual(len(w.bg),1);self.assertFalse(w.fg)

    def test_escape_exits_when_text_field_has_focus(self):
        w=self.window;w.quality_timer.stop();spin_until(lambda:not w.bridge.busy)
        w.object_name.setFocus();QTest.keyClick(w.object_name,Qt.Key_Escape)
        spin_until(lambda:not w.bridge.thread.isRunning() and not w.isVisible())
        self.assertFalse(w.isVisible())

    def test_escape_preserves_unsaved_result_confirmation(self):
        from PySide6.QtWidgets import QMessageBox
        w=self.window;w.quality_timer.stop();spin_until(lambda:not w.bridge.busy)
        w.state['ids_dirty']=True;w.canvas.setFocus()
        with patch('desktop.window.QMessageBox.question',return_value=QMessageBox.Cancel) as question:
            QTest.keyClick(w.canvas,Qt.Key_Escape)
            question.assert_called_once()
        self.assertTrue(w.isVisible());self.assertTrue(w.bridge.thread.isRunning())

    def test_escape_cancels_job_and_waits_before_exiting(self):
        from PySide6.QtWidgets import QMessageBox
        w=self.window;w.quality_timer.stop();spin_until(lambda:not w.bridge.busy)
        self.engine.preview_delay=.2;w.call('preview',{})
        w.canvas.setFocus()
        with patch('desktop.window.QMessageBox.question',return_value=QMessageBox.Yes):
            QTest.keyClick(w.canvas,Qt.Key_Escape)
        self.assertTrue(w.bridge.cancel_event.is_set())
        spin_until(lambda:not w.bridge.thread.isRunning() and not w.isVisible())
        self.assertFalse(w.isVisible())

    def test_volume_completion_buttons_follow_foreground_and_frozen_state(self):
        w=self.window;spin_until(lambda:not w.bridge.busy)
        self.assertFalse(w.complete_cube.isEnabled());self.assertFalse(w.complete_cylinder.isEnabled())
        self.engine.state['selected']=w.state['selected']=20;w.update_controls()
        for button,shape in ((w.complete_cube,'cube'),(w.complete_cylinder,'cylinder')):
            button.click();spin_until(lambda:not w.bridge.busy)
            self.assertEqual(self.engine.calls[-1][1]['action'],'complete')
            self.assertEqual(self.engine.calls[-1][1]['shape'],shape)
            self.assertEqual(w.mode,'foreground');self.assertEqual(w.state['selected'],25)
        w.toggle_language();self.assertEqual(w.complete_cylinder.text(),'圆柱体补全')
        self.prompt_mode();self.assertFalse(w.complete_cube.isEnabled());self.assertFalse(w.complete_cylinder.isEnabled())

    def test_stale_preview_is_discarded(self):
        self.prompt_mode();w=self.window;self.engine.preview_delay=.15
        w.gesture([12,12],[12,12],0,False)
        spin_until(lambda:w.bridge.busy and w.bridge.action=='preview')
        w.gesture([20,20],[20,20],0,False)
        spin_until(lambda:bool(w.masks) and not w.bridge.busy)
        previews=[d for a,d in self.engine.calls if a=='preview']
        self.assertEqual(previews[-1]['foreground'],[[12,12],[20,20]])
    def test_exit_during_auto_preview_releases_backend(self):
        self.prompt_mode();w=self.window;self.engine.preview_delay=.15
        w.gesture([12,12],[12,12],0,False)
        spin_until(lambda:w.bridge.busy and w.bridge.action=='preview')
        w.prompt_check.click()
        spin_until(lambda:not w.bridge.busy and any(a=='release' for a,d in self.engine.calls))
        self.assertIsNone(w.capture);self.assertFalse(w.prompt_check.isChecked());self.assertFalse(w.masks)

    def press_cancel(self):
        self.window.canvas.setFocus();QTest.keyClick(self.window.canvas,Qt.Key_C)
        spin_until(lambda:not self.window.bridge.busy and not self.window.cancel_round_pending)
    def test_c_clears_prompts_and_returns_to_navigation(self):
        self.prompt_mode();w=self.window
        w.gesture([10,10],[10,10],0,False);spin_until(lambda:bool(w.masks) and not w.bridge.busy)
        self.press_cancel()
        self.assertIsNone(w.capture);self.assertFalse(w.fg);self.assertFalse(w.masks)
        self.assertFalse(w.prompt_check.isChecked());self.assertEqual(w.canvas.mode,'camera')
        self.assertEqual(w.mode,'all');self.assertFalse(w.preview_timer.isActive())
    def test_c_during_model_loading_closes_dialog_and_exits_prompts(self):
        w=self.window;self.engine.load_delay=.15
        w.prompt_check.click();spin_until(lambda:w.bridge.busy and w.bridge.action=='load_model')
        dialog=w.model_dialog
        self.press_cancel()
        self.assertIsNone(w.model_dialog);self.assertFalse(dialog.isVisible())
        self.assertIsNone(w.capture);self.assertFalse(w.masks)
    def test_c_during_preview_discards_stale_candidates(self):
        self.prompt_mode();w=self.window;self.engine.preview_delay=.15
        w.gesture([10,10],[10,10],0,False)
        spin_until(lambda:w.bridge.busy and w.bridge.action=='preview')
        self.press_cancel();QTest.qWait(250)
        self.assertIsNone(w.capture);self.assertFalse(w.masks);self.assertFalse(w.preview_timer.isActive())
    def test_c_stops_running_three_dimensional_refinement(self):
        self.prompt_mode();w=self.window;self.engine.apply_delay=.3
        w.gesture([10,10],[10,10],0,False);spin_until(lambda:bool(w.masks) and not w.bridge.busy)
        w.segment();spin_until(lambda:any(a=='apply' for a,d in self.engine.calls))
        self.press_cancel()
        self.assertEqual(w.state['selected'],0);self.assertFalse(w.capture);self.assertEqual(w.mode,'all')
        self.assertEqual(self.engine.calls[-1][0],'cancel_round')
    def test_c_rolls_back_a_result_finishing_during_cancellation(self):
        self.prompt_mode();w=self.window
        self.engine.apply_delay=.1;self.engine.ignore_cancel=True
        w.gesture([10,10],[10,10],0,False);spin_until(lambda:bool(w.masks) and not w.bridge.busy)
        w.segment();spin_until(lambda:any(a=='apply' for a,d in self.engine.calls))
        self.press_cancel();self.assertEqual(w.state['selected'],0)
        self.assertFalse(w.capture);self.assertEqual(w.mode,'all')
    def test_c_reverts_completed_uncommitted_result(self):
        self.engine.state['selected']=5
        self.prompt_mode();w=self.window
        w.gesture([10,10],[10,10],0,False);spin_until(lambda:bool(w.masks) and not w.bridge.busy)
        w.segment();spin_until(lambda:not w.bridge.busy and not w.capture)
        self.assertEqual(w.state['selected'],20)
        self.press_cancel();self.assertEqual(w.state['selected'],5)
        self.press_cancel();self.assertEqual(w.state['selected'],5)
    def test_c_does_not_cancel_while_typing(self):
        self.prompt_mode();w=self.window
        w.object_name.setFocus();QTest.keyClicks(w.object_name,'c')
        self.assertEqual(w.object_name.text(),'c');self.assertTrue(w.capture)
        self.assertFalse(any(a=='cancel_round' for a,d in self.engine.calls))

    def test_add_id_hides_and_enters_next_round(self):
        w=self.window;w.state['selected']=self.engine.state['selected']=20
        w.add_object();spin_until(lambda:w.capture and not w.bridge.busy)
        self.assertTrue(w.state['objects'][0]['hidden']);self.assertEqual(w.state['selected'],0)
        self.assertEqual(w.mode,'all')
    def test_correction_click_box_and_preview_commit(self):
        w=self.window;w.correction=True;w.correction_id=7;w.correction_mode='ellipse';w.update_controls()
        w.gesture([50,40],[50,40],0,False);spin_until(lambda:w.state.get('preview_id')=='preview')
        self.assertTrue(w.correct_confirm.isEnabled())
        w.gesture([40,30],[70,80],0,False);spin_until(lambda:not w.bridge.busy)
        self.assertEqual(self.engine.calls[-1][1]['rectangle'],[[40,30],[70,80]])
        w.confirm_correction();spin_until(lambda:not w.correction)
        self.assertEqual(self.engine.calls[-1][1]['preview_id'],'preview')
    def test_gif_single_orbit_parameters_for_both_sources(self):
        w=self.window;self.assertEqual(w.gif_elevation.value(),60)
        with patch.object(w,'show_gifs'):
            w.export_gif('all');spin_until(lambda:not w.bridge.busy)
            self.assertEqual(self.engine.calls[-1][1]['elevation_degrees'],60)
            self.assertEqual(self.engine.calls[-1][1]['source'],'all')
            self.assertNotIn('ids',self.engine.calls[-1][1])
            w.gif_elevation.setValue(25);w.export_gif('current');spin_until(lambda:not w.bridge.busy)
            self.assertEqual(self.engine.calls[-1][1]['elevation_degrees'],25)
    def test_language_and_four_native_render_modes(self):
        w=self.window;w.toggle_language();self.assertIn('进入分割提示',w.prompt_check.text())
        w.toggle_language();self.assertEqual(w.render_mode.count(),4)
        for i,name in enumerate(('rgb','splats','points','depth')):
            w.render_mode.setCurrentIndex(i);w.change_render();spin_until(lambda:not w.bridge.busy)
            self.assertEqual(self.engine.calls[-1][1]['render_mode'],name)
    def test_ground_mode_and_native_file_drop_signal(self):
        w=self.window;w.ground_action('start');spin_until(lambda:bool(w.ground))
        w.gesture([10,10],[10,10],0,False);spin_until(lambda:not w.bridge.busy)
        self.assertEqual(self.engine.calls[-1][1]['operation'],'pick')
        w.ground_action('cancel');spin_until(lambda:not w.ground)
    def test_shortcuts_avoid_text_fields(self):
        w=self.window;w.canvas.setFocus();QTest.keyClick(w.canvas,Qt.Key_M)
        spin_until(lambda:bool(w.capture) and not w.bridge.busy)
        w.object_name.setFocus();QTest.keyClicks(w.object_name,'m')
        self.assertTrue(w.capture);self.assertEqual(w.object_name.text(),'m')


class GeometryTest(unittest.TestCase):
    def test_letterbox_pixel_mapping_matches_painting(self):
        canvas=Viewport();canvas.resize(800,800);canvas.set_image(image_data())
        rect=canvas.image_rect();self.assertAlmostEqual(rect.height(),800*2/3)
        self.assertIsNone(canvas.pixel(QPointF(10,10)))
        center=canvas.pixel(QPointF(400,400));self.assertAlmostEqual(center[0],150);self.assertAlmostEqual(center[1],100)
        p=QPointF(rect.x()+80,rect.y()+80)
        self.assertEqual(canvas.pixel(p),canvas.pixel(p,True))
    def test_projection_clips_behind_and_keeps_visible(self):
        matrix=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        self.assertIsNone(project_segment([0,0,-1],[0,0,-.1],matrix,300,200))
        points=project_segment([-2,0,.5],[0,0,.5],matrix,300,200)
        self.assertAlmostEqual(points[0].x(),-.5);self.assertAlmostEqual(points[1].x(),149.5)


class WorkerTest(unittest.TestCase):
    def test_serialization_progress_cancellation_and_shutdown(self):
        class Engine:
            def execute(self,action,data,progress,cancel):
                for i in range(50):
                    if cancel.is_set():raise ValueError('cancelled')
                    progress(i,'working');time.sleep(.005)
                return {'ok':True}
        bridge=EngineBridge(Engine);errors=[];progress=[]
        bridge.error.connect(lambda a,m:errors.append(m));bridge.progress.connect(lambda n,m,v:progress.append(n))
        self.assertTrue(bridge.submit('gif',{}));self.assertFalse(bridge.submit('frame',{}))
        spin_until(lambda:bool(progress));bridge.cancel();spin_until(lambda:bool(errors))
        self.assertEqual(errors,['cancelled']);self.assertFalse(bridge.busy)
        bridge.close();spin_until(lambda:not bridge.thread.isRunning())

if __name__=='__main__':unittest.main()
