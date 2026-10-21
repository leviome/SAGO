"""Per-user writable paths; installed application resources are read-only."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = Path(os.environ.get('XDG_CONFIG_HOME',Path.home()/'.config'))/'sago-desktop'
DATA = Path(os.environ.get('XDG_DATA_HOME',Path.home()/'.local/share'))/'sago-desktop'
CACHE = Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'sago-desktop'


def read_config():
    try:return json.loads((CONFIG/'settings.json').read_text())
    except (OSError,ValueError):return {}


def write_config(data):
    CONFIG.mkdir(parents=True,exist_ok=True)
    pending=CONFIG/'settings.tmp'
    pending.write_text(json.dumps(data,indent=2));pending.replace(CONFIG/'settings.json')


def configure(settings):
    CACHE.mkdir(parents=True,exist_ok=True)
    DATA.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(CACHE/'matplotlib'))
    # Old research utility code writes relative diagnostics. Never write into /opt.
    os.chdir(DATA)
    return Path(settings.get('sam2_checkpoint') or ROOT/'models/sam2_hiera_large.pt'), Path(settings.get('output') or DATA/'outputs')
