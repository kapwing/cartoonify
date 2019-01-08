from __future__ import division
from app.workflow import Workflow
from app.drawing_dataset import DrawingDataset
from app.image_processor import ImageProcessor, tensorflow_model_name, model_path
from pathlib import Path
import sys

root = Path(__file__).parent

print('\r\r\n\n')
print('\rInitializing machine learning models ...\r')
print('\rThis might take a couple of minutes, please be patient!\r')

# init objects
dataset = DrawingDataset(str(root / 'downloads/drawing_dataset'), str(root / 'app/label_mapping.jsonl'))
imageprocessor = ImageProcessor(str(model_path),
                                str(root / 'app' / 'object_detection' / 'data' / 'mscoco_label_map.pbtxt'),
                                tensorflow_model_name)

def run(path):
  app = Workflow(dataset, imageprocessor, None)
  print('\r\rAccessing database for sample imagery ...\r\r')
  app.setup(setup_gpio=False)
  print('\r\rProcessing your upload ... this may take a couple minutes ...\r\r')
  app.process(str(path), top_x=3)
  print('\r\rProcessing almost done! Please be patient ...\r\r')
  app.save_results()

path = sys.argv[1]
run(path)
sys.exit()

