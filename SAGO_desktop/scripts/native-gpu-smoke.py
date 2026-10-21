"""Exercise real native Qt + CUDA + SAM2 without modifying scene knowledge."""
import os
from pathlib import Path
import sys
import time
import json
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ['XDG_CONFIG_HOME']=str(ROOT/'.test-config')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.test-cache'))
os.chdir(ROOT)
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from desktop.engine import DesktopEngine
from desktop.window import MainWindow
from desktop.theme import STYLE
app=QApplication([]);app.setStyleSheet(STYLE)
out=ROOT/'validation/native-gpu';out.mkdir(parents=True,exist_ok=True)
w=MainWindow(lambda:DesktopEngine(ROOT/'models/sam2_hiera_large.pt',out),out,sys.argv[1]);w.show()
phase=0;started=time.monotonic();errors=[];report={}
w.bridge.error.connect(lambda a,m:errors.append((a,m)))
sys.excepthook=lambda typ,error,tb:errors.append(('Qt callback',repr(error)))

def tick():
 global phase
 try:
  if errors:raise RuntimeError(errors)
  if time.monotonic()-started>240:raise TimeoutError(phase)
  if w.bridge.busy:return
  if phase==0 and w.state.get('count'):
   w.resize_timer.stop();report['count']=w.state['count'];report['objects']=len(w.state.get('objects',[]))
   w.grab().save(str(out/'01-scene.png'))
   if w.state.get('objects'):
    w.object_action('select',id=w.state['objects'][0]['id']);phase=1
   else:phase=1
  elif phase==1:
   w.prompt_check.click();phase=2
  elif phase==2 and w.capture and w.loaded_model=='sam2':
   image=w.canvas.base
   points=[]
   for y in range(5,image.height()-5,5):
    for x in range(5,image.width()-5,5):
     color=image.pixelColor(x,y)
     if max(color.red(),color.green(),color.blue())>45:points.append((x,y))
   assert points,'Empty captured image'
   point=min(points,key=lambda p:(p[0]-image.width()/2)**2+(p[1]-image.height()/2)**2)
   w.gesture(list(point),list(point),0,False);phase=3
  elif phase==3 and w.masks:
   assert w.candidates.count()==len(w.masks), 'Native candidates were not populated'
   report['candidates']=len(w.masks);w.grab().save(str(out/'02-prompts.png'))
   w.segment();phase=4
  elif phase==4 and not w.capture:
   report['selected_after_3d']=w.state['selected'];report['drone_views']=len(w.drone_views)
   assert w.state['selected']>0
   w.bbox.setChecked(True);w.route.setChecked(True)
   w.grab().save(str(out/'03-segmentation.png'))
   report['elapsed_seconds']=time.monotonic()-started
   (out/'report.json').write_text(json.dumps(report,indent=2))
   print('PASS NATIVE GPU',report,flush=True)
   w.state['ids_dirty']=False;timer.stop();w.close()
 except Exception as error:
  print('FAIL NATIVE GPU',repr(error),flush=True)
  report['error']=repr(error);(out/'report.json').write_text(json.dumps(report,indent=2))
  timer.stop();w.state['ids_dirty']=False;w.close();QTimer.singleShot(2000,app.quit)

timer=QTimer();timer.timeout.connect(tick);timer.start(100)
app.exec()
if 'error' in report:raise SystemExit(1)
