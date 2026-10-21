"""Desktop session lifecycle and cancellation around the vendored engine."""
from backend.engine import Engine


class DesktopEngine(Engine):
    def load_model(self, data):
        self.validate_segmenter(data)
        if self.tracker is None:
            from desktop.checkpoints import resolve_checkpoint
            self.checkpoint = resolve_checkpoint(self.checkpoint,
                getattr(self, '_checkpoint_progress', None), getattr(self, '_checkpoint_cancel', None))
            progress = getattr(self, '_checkpoint_progress', None)
            if progress:
                progress(0, 'Loading SAM2…', dict(kind='model_loading', stage='Loading SAM2…'))
        return super().load_model(data)

    def execute(self, action, data, progress, cancel):
        self._checkpoint_progress, self._checkpoint_cancel = progress, cancel
        if action == 'cancel_round':
            previous = getattr(self, '_round_rollback', None)
            if previous is not None:
                self.selection, self.drone_path, self.undo_stack, self.redo_stack = previous
                self._round_rollback = None
            self.invalidate()
            return super().execute('frame', data, progress, cancel)
        # A new capture starts another round. Object edits commit ownership of
        # the previous result and must never be undone by round cancellation.
        if action in ('load', 'capture', 'objects') or action == 'ground' and data.get('operation') == 'apply':
            self._round_rollback = None
        if action == 'apply':
            # apply replaces selection tensors rather than modifying them in place.
            # Preserve history too: cancellation must not add a bogus Undo entry.
            self._round_rollback = (self.selection, self.drone_path,
                                    list(self.undo_stack), list(self.redo_stack))
        if action == 'release':
            self.invalidate()
            return super().execute('frame', data, progress, cancel)
        return super().execute(action, data, progress, cancel)
