#!/usr/bin/env python3
"""Build a relocatable Debian application; CUDA runtime remains configurable."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parent.parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--without-models',action='store_true',help='Compatibility option: all packages download checkpoints on first use')
    parser.add_argument('--qt-site',type=Path,help='Site-packages containing PySide6/shiboken6, bundled for offline Qt startup')
    parser.add_argument('--output',type=Path,default=ROOT/'dist')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='sago-deb-') as folder:
        package=Path(folder);app=package/'opt/sago-desktop';app.mkdir(parents=True)
        for name in ('desktop','backend','gs','sago_core','packaging','scripts'):
            shutil.copytree(ROOT/name,app/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','build','.git','*.egg-info'))
        for name in ('simple-knn','rasterizer-depth','rasterizer-mask','rasterizer-topk'):
            shutil.copytree(ROOT/'vendor'/name,app/'vendor'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','build','.git','*.egg-info'))
        for name in ('run.py','launch.py','requirements.txt','README.md','LICENSE','THIRD_PARTY_NOTICES.md'):
            shutil.copy2(ROOT/name,app/name)
        # Checkpoints are always external, including the default build.
        (app/'models').mkdir()
        shutil.copy2(ROOT/'models/README.md',app/'models/README.md')
        for resource in ('lib','licenses'):
            source=ROOT/'.runtime'/resource
            if source.is_dir():shutil.copytree(source,app/'.runtime'/resource)
        if args.qt_site:
            for name in ('PySide6','shiboken6'):
                shutil.copytree(args.qt_site/name,app/'.runtime/qt'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            for path in args.qt_site.glob('*dist-info'):
                if path.name.lower().startswith(('pyside6','shiboken6')):shutil.copytree(path,app/'.runtime/qt'/path.name)
        binary=package/'usr/bin/sago';binary.parent.mkdir(parents=True)
        binary.write_text('#!/bin/sh\nexec /usr/bin/python3 /opt/sago-desktop/launch.py "$@"\n');binary.chmod(0o755)
        for src,destination in (('sago-desktop.desktop','usr/share/applications/sago-desktop.desktop'),
             ('sago.svg','usr/share/icons/hicolor/scalable/apps/sago-desktop.svg'),
             ('sago-mime.xml','usr/share/mime/packages/sago-desktop.xml')):
            target=package/destination;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'packaging'/src,target)
        debian=package/'DEBIAN';debian.mkdir()
        size=sum(p.stat().st_size for p in package.rglob('*') if p.is_file())//1024
        (debian/'control').write_text(f'''Package: sago-desktop
Version: 1.5.0
Section: graphics
Priority: optional
Architecture: amd64
Maintainer: SAGO Desktop maintainers
Depends: python3 (>= 3.9), libgl1, libegl1, libxkbcommon0, libxkbcommon-x11-0, libxcb-icccm4, libxcb-keysyms1, libxcb-shape0, libxcb-xinerama0, libxcb-render-util0
Recommends: shared-mime-info, desktop-file-utils
Installed-Size: {size}
Description: Native Qt Gaussian segmentation desktop application
 SAM2 point and box prompts, Virtual Drone 3D segmentation,
 multi-object IDs, corrections, ground calibration, GIF and OBJ mesh export.
 Requires an NVIDIA driver and a configured CUDA Python runtime.
''')
        for name in ('postinst','postrm'):
            script=debian/name
            script.write_text('#!/bin/sh\nset -e\nif command -v update-desktop-database >/dev/null 2>&1; then update-desktop-database -q /usr/share/applications || true; fi\nif command -v update-mime-database >/dev/null 2>&1; then update-mime-database /usr/share/mime || true; fi\nexit 0\n');script.chmod(0o755)
        output=args.output/'sago-desktop_1.5.0_amd64.deb'
        subprocess.run(['dpkg-deb','--root-owner-group','-Zzstd','-z1','--build',str(package),str(output)],check=True)
        print(output)

if __name__=='__main__':main()
