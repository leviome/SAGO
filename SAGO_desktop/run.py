#!/usr/bin/env python3
"""Native Linux entry point. Accepts `sago path/to/scene.ply`."""
import argparse
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
# The deb bundles Qt bindings; development can use its isolated .venv instead.
for folder in (ROOT/'.runtime/qt',ROOT/'.runtime/extensions'):
    if folder.is_dir():sys.path.insert(0,str(folder))


def main():
    parser=argparse.ArgumentParser(description='SAGO Desktop — native Linux Gaussian segmentation')
    parser.add_argument('scene',nargs='?',help='Gaussian PLY or training output directory')
    parser.add_argument('--checkpoint',help='SAM2 checkpoint override')
    parser.add_argument('--output',help='Export directory override')
    parser.add_argument('--version',action='version',version='SAGO Desktop 1.5.1')
    args=parser.parse_args()
    # Resolve relative CLI paths before switching to writable application data.
    scene=str(Path(args.scene).expanduser().resolve()) if args.scene else None
    native_cursor=ROOT/'.runtime/lib/libxcb-cursor.so.0'
    if native_cursor.is_file():
        import ctypes
        ctypes.CDLL(str(native_cursor),mode=ctypes.RTLD_GLOBAL)
    from PySide6.QtWidgets import QApplication
    from desktop.config import read_config,configure
    settings=read_config()
    if args.checkpoint:settings['sam2_checkpoint']=str(Path(args.checkpoint).expanduser().resolve())
    if args.output:settings['output']=str(Path(args.output).expanduser().resolve())
    checkpoint,output=configure(settings)
    app=QApplication([sys.argv[0]])
    app.setApplicationName('SAGO Desktop');app.setOrganizationName('SAGO')
    app.setDesktopFileName('sago-desktop')
    from desktop.theme import STYLE
    from desktop.window import MainWindow
    from desktop.engine import DesktopEngine
    app.setStyleSheet(STYLE)
    window=MainWindow(lambda:DesktopEngine(checkpoint,output),output,scene)
    window.show()
    return app.exec()


if __name__=='__main__':raise SystemExit(main())
