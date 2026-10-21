"""Verified, resumable first-use SAM2 downloads into the user's cache."""
from dataclasses import dataclass
import fcntl
import hashlib
import os
from pathlib import Path
import re
import threading
import time
from urllib.error import URLError
from http.client import HTTPException
from urllib.request import Request, urlopen

from desktop.config import DEFAULT_CHECKPOINT, ROOT


@dataclass(frozen=True)
class CheckpointSpec:
    url: str
    size: int
    sha256: str


SAM2 = CheckpointSpec(
    'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt',
    897952466, '7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b')


class DownloadCancelled(ValueError):
    pass


def download_checkpoint(target, progress=None, cancel=None, spec=SAM2, opener=urlopen):
    """Never expose a partial or unverified file under the final model name."""
    target = Path(target).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.part')
    cancel = cancel if cancel is not None else threading.Event()
    def check_cancel():
        if cancel.is_set():raise DownloadCancelled('Model download cancelled; partial data is kept for retry.')
    def notify(stage, size=0):
        if progress:
            progress(min(100, round(size / spec.size * 100)), stage,
                     dict(kind='model_download', stage=stage, downloaded=size, total=spec.size, path=str(target)))
    def verified(path):
        notify('Verifying SAM2 checkpoint…')
        if not path.is_file() or path.stat().st_size != spec.size:return False
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while True:
                check_cancel();chunk=stream.read(4 * 1024 * 1024)
                if not chunk:break
                digest.update(chunk)
        return digest.hexdigest() == spec.sha256
    with target.with_suffix(target.suffix + '.lock').open('a+b') as lock:
        while True:
            check_cancel()
            try:fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB);break
            except BlockingIOError:
                notify('Waiting for another SAM2 download…');cancel.wait(.2)
        if target.exists():
            if verified(target):notify('SAM2 checkpoint ready', spec.size);return target
            target.unlink()  # Only the managed cache is repaired; custom files are never overwritten.
        if partial.exists() and partial.stat().st_size > spec.size:partial.unlink()
        for attempt in range(3):
            check_cancel()
            offset=partial.stat().st_size if partial.exists() else 0
            if offset == spec.size:break
            headers={'Accept-Encoding':'identity','User-Agent':'SAGO-Desktop'}
            if offset:headers['Range']=f'bytes={offset}-'
            notify('Downloading SAM2 checkpoint…', offset)
            try:
                with opener(Request(spec.url, headers=headers), timeout=20) as response:
                    status=response.status
                    if status == 206:
                        match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                        if not match or int(match[1]) != offset or int(match[2]) != spec.size-1 or int(match[3]) != spec.size:
                            raise ValueError('Invalid checkpoint download range; retry or choose a local checkpoint.')
                    elif status == 200:offset=0
                    else:raise URLError(f'Unexpected HTTP status {status}')
                    length=response.headers.get('Content-Length')
                    if length is not None and int(length) != spec.size-offset:
                        raise ValueError('Unexpected SAM2 checkpoint size; retry or choose a local checkpoint.')
                    count=offset;last=0.
                    with partial.open('ab' if offset else 'wb') as output:
                        while True:
                            check_cancel();chunk=response.read(1024 * 1024)
                            if not chunk:break
                            count+=len(chunk)
                            if count > spec.size:
                                raise ValueError('SAM2 checkpoint exceeds expected size.')
                            output.write(chunk)
                            if time.monotonic()-last >= .1:
                                notify('Downloading SAM2 checkpoint…',count);last=time.monotonic()
                    if count != spec.size:raise URLError('Download ended before the checkpoint was complete')
                break
            except (URLError, OSError, HTTPException) as error:
                check_cancel()
                if attempt == 2:
                    raise ValueError(f'SAM2 download failed: {error}. Retry, or choose an existing checkpoint in Preferences.') from error
                notify('Download interrupted; retrying…', partial.stat().st_size if partial.exists() else 0)
                cancel.wait(.5)
        check_cancel()
        if not verified(partial):
            partial.unlink(missing_ok=True)
            raise ValueError('SAM2 checksum verification failed. Retry the download or choose an existing checkpoint.')
        check_cancel()
        os.replace(partial,target)
        notify('SAM2 checkpoint ready',spec.size)
        return target


def resolve_checkpoint(requested, progress=None, cancel=None):
    requested=Path(requested).expanduser()
    legacy=ROOT/'models/sam2_hiera_large.pt'
    if requested == DEFAULT_CHECKPOINT or (requested == legacy and not requested.is_file()):
        return download_checkpoint(DEFAULT_CHECKPOINT,progress,cancel)
    if not requested.is_file():
        raise ValueError(f'SAM2 checkpoint not found: {requested}. Choose an existing checkpoint, or clear the checkpoint setting for automatic download.')
    return requested
