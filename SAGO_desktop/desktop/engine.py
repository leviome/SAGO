"""Desktop session lifecycle and cancellation around the vendored engine."""
from backend.engine import Engine


class DesktopEngine(Engine):
    def execute(self, action, data, progress, cancel):
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
