"""Round cancellation restores only transient selection and history."""
import tempfile
import threading
import unittest
from unittest.mock import patch
from desktop.engine import DesktopEngine
from backend.engine import Engine


class RoundCancellationTest(unittest.TestCase):
    def test_cancel_restores_selection_route_history_and_preserves_objects(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=DesktopEngine('unused.pt',folder)
            engine.selection=object();before=engine.selection
            engine.drone_path={'old':True};old_route=engine.drone_path
            engine.undo_stack=['before'];engine.redo_stack=['redo']
            engine.objects=[{'id':'saved'}];engine.ids_dirty=True
            def execute(instance,action,data,progress,cancel):
                if action=='apply':
                    instance.selection=object();instance.drone_path={'new':True}
                    instance.undo_stack.append('this round');instance.redo_stack.clear()
                return {'selected':instance.selection}
            with patch.object(Engine,'execute',execute):
                engine.execute('apply',{},lambda *a:None,threading.Event())
                engine.execute('cancel_round',{},lambda *a:None,threading.Event())
                self.assertIs(engine.selection,before);self.assertIs(engine.drone_path,old_route)
                self.assertEqual(engine.undo_stack,['before']);self.assertEqual(engine.redo_stack,['redo'])
                self.assertEqual(engine.objects,[{'id':'saved'}]);self.assertTrue(engine.ids_dirty)
                engine.execute('cancel_round',{},lambda *a:None,threading.Event())
                self.assertIs(engine.selection,before)
    def test_saved_object_or_new_capture_ends_previous_round_rollback(self):
        for commit in ('objects','capture','load'):
            with self.subTest(commit=commit),tempfile.TemporaryDirectory() as folder:
                engine=DesktopEngine('unused.pt',folder)
                def execute(instance,action,data,progress,cancel):
                    if action=='apply':instance.selection=object()
                    return {}
                with patch.object(Engine,'execute',execute):
                    engine.execute('apply',{},lambda *a:None,threading.Event())
                    result=engine.selection
                    engine.execute(commit,{},lambda *a:None,threading.Event())
                    engine.execute('cancel_round',{},lambda *a:None,threading.Event())
                    self.assertIs(engine.selection,result)
