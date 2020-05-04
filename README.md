# Real-time-Vehicle-and-Pedestrian-Counting

<a name="0Zy34"></a>
# Introduction
This project focuses " counting and statistics of moving targets we care about ", drive by Centernet which was Implemented in Pytorch."<br />It needs to be stated that the Centernet detector of this project is forked from the nice implementation of [YunYang1994](https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3), and CenterNet + DeepSORT tracking implementation: [centerNet-deep-sort](https://github.com/kimyoon-young/centerNet-deep-sort) from [kimyoon-young](https://github.com/xingyizhou/CenterNet).
<a name="HbIRq"></a>
# Project Demo

- The demo is available on Youtube and Bilibili
- on my laptop gtx1060 FPS reached 9-16 (dla34) , 12-35 (resdcn18)，It must be noted that due to DeepSORT efficiency issues, the bottleneck of running frames mainly comes from the computing power of the CPU, which limits the performance of the GPU 
![pic1](https://github.com/Clemente420/Real-time-Vehicle-and-Pedestrian-Counting-CenterNet/blob/master/doc/Screenshot%20from%202020-05-04%2014-31-50.png)

![pic2](https://github.com/Clemente420/Real-time-Vehicle-and-Pedestrian-Counting-CenterNet/blob/master/doc/Screenshot%20from%202020-05-04%2015-20-23.png)

![pic3](https://github.com/Clemente420/Real-time-Vehicle-and-Pedestrian-Counting-CenterNet/blob/master/doc/Screenshot%20from%202020-05-04%2015-27-07.png)

<a name="fOSw7"></a>
# Installation
Environmental requirements
```
CUDNN_MAJOR 7
CUDA Version 10.0.130
```
Compile deformable convolutional
```
cd /CenterNet/src/lib/models/networks/DCNv2
python setup.py build develop
```
Reproduce the environment
```
 conda env create -f environment.yml
```


download the [model](https://github.com/xingyizhou/CenterNet/blob/master/readme/MODEL_ZOO.md) put in `.``/centerNet-deep-sort/CenterNet/models`

two test videos are prepared [here](https://drive.google.com/drive/folders/16ZYObAm48Y0ImnCjtUIzeasyp2QaPphI?usp=sharing), you should download.<br />

<a name="qyyHA"></a>
# Parameter adjustment
To switch the model, you need to modify demo_centernet_deepsort.py and you should write the right ARCH.
```
MODEL_PATH = './CenterNet/models/ctdet_coco_dla_2x.pth'
ARCH = 'dla_34'

#MODEL_PATH = './CenterNet/models/ctdet_coco_resdcn18.pth'
#ARCH = 'resdcn_18'
```
and theb specifies the input and output path of the video.
```
opt.vid_path = './vehicle.mp4'
self.output = cv2.VideoWriter( "./output.mp4", encode, 24, (self.im_width, self.im_height))
```
If you need to filter some categories for testing
```
specified_class_id_filter = 1 # if you wanna filter certain class
```
Adjust the position and layout of the counting line to modify utils.py
```
line = [(0, 530), (2100, 530)]
```
<a name="3fPJF"></a>
## If you need to use a custom training model，you should:


<a name="a6KGM"></a>
### 1. dataset
centernet/src/lib/datasets/dataset/<br />Copy coco.py and rename it a custom file (I define it as your_train_model.py)
```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

class COCO(data.Dataset):   # Modify to class Your_train_model (data.Dataset): This is the class name defined by yourself
  num_classes = 80  # Here is the number of detection categories, which is the total value of class_name, depending on your custom training
  default_resolution = [512, 512]   # If you feel that your hardware can't keep up, you can change the size appropriately
  mean = np.array([0.40789654, 0.44719302, 0.47026115], #　Mean of the image dataset to be trained
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835], # Standard deviation of the image dataset to be trained
                   dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(COCO, self).__init__()    #　Change COCO to your own class name(Your_train_model)
    self.data_dir = os.path.join(opt.data_dir, 'coco')  # Change to self.data_dir = os.path.join (opt.data_dir, 'Your_train_model')
    self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))  # Change to self.img_dir = os.path.join (self.data_dir, 'images')
    if split == 'test': # Change to if split == 'val':
      self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'image_info_test-dev2017.json').format(split) # Change to 'train.json').format(split)
    else:
      if opt.task == 'exdet':
        self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'instances_extreme_{}2017.json').format(split) # Change to 'train.json').format(split)
      else:
        self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'instances_{}2017.json').format(split) # Change to 'train.json').format(split)
    self.max_objs = 128
    self.class_name = [
      '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
      'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
      'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
      'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
      'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
      'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
      'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
      'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
      'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
      'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
      'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
      'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
      'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    # The above self.class_name is modified to the corresponding labeled class name, where '__background__' is reserved
    self._valid_ids = [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
      24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
      37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
      48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
      58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
      72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
      82, 84, 85, 86, 87, 88, 89, 90]
    # The above self._valid_ids is modified to correspond to class_name, no more, no less
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]

    # ----------------snip-------------------------------
```


<a name="gJVfD"></a>
### 2. dataset_factory
Modify centernet/src/lib/datasets/ dataset_factory.py<br />See the comments for details and modify them according to your own situation.
```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.seaship import Seaship
from .dataset.your_train_model import Your_train_model	# Here you need to update to your customization

dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'seaship': Seaship,
  'your_train_model':Your_train_model	# Here you need to update to your customization
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
```
<a name="yrCOX"></a>
### 3. opt
Modifycenternet/src/lib/opts.py<br />See the comments for details and modify them according to your own situation.
```python
class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    # basic experiment setting
    self.parser.add_argument('task', default='ctdet',
                             help='ctdet | ddd | multi_pose | exdet')
    self.parser.add_argument('--dataset', default='coco',　# Update default = 'xx' to be customized
                             help='coco | kitti | coco_hp | pascal | your_train_model')　# Update and insert '| your_train_model'
```
Pull the code to about 339 lines
```python
  def init(self, args=''):
    default_dataset_info = {
      'ctdet': {'default_resolution': [512, 512], 'num_classes': 14, # Class number, corrected according to the labeling situation of your training data set
                'mean': [0.49401256, 0.51401981, 0.53022468], 'std': [0.14567583, 0.14387697, 0.14387015],	# Fill in the mean and standard deviation calculated from the image data set
                'dataset': 'your_train_model'},	# should be updated here 
```
<a name="bf07g"></a>
### 4. debugger
Modifycenternet/src/lib/utils/debugger.py<br />Pull the code to about 45 lines, follow the previous coco template, and modify it according to your own situation
```python
elif num_classes == 80 or dataset == 'coco':
    self.names = coco_class_name
elif num_classes == 14 or dataset == 'your_train_model':  # Need to modify here
    self.names = your_train_model_class_name  # should be named here
```
Pull the code to about 450 lines and insert the category defined by yourself, excluding"__background__", such

<a name="1ZPcX"></a>
### 5. utill
Modifycenternet/util.py<br />Pull the code to about 31 lines and modify it according to your own situationclass_names, note retention"__background__'<br />

<a name="RDh0r"></a>
### 6. centroid of bbox
In deep_sort.py<br />About 89 lines, the position of the target positioning centroid can be modified or choiced
```python
# Track is the center of the detection box
center=(int((x1+x2)/2-),int((y1+y2)/2-))# Draw a trajectory map to record the center point of each time
# Track is the bottom of the detection box
# center = (int((x1 + x2)/2), int(((y2)))# draw a trajectory graph and record the bottom of each time
```
<a name="NDPCC"></a>
# Run demo:
```
conda activate your_env_name
python demo_centernet_deepsort.py
```


<a name="DlQMB"></a>
# Citation
If you use this code for your publications, please cite it as:
```
@ONLINE{vdtc,
    author = "Clemente420",
    title  = "Real-time-Vehicle-and-Pedestrian-Counting-CenterNet",
    year   = "2020",
    url    = "https://github.com/Clemente420/Real-time-Vehicle-and-Pedestrian-Counting-CenterNet"
}
```
<a name="b8tek"></a>
# Author

- Please contact for dataset or more info: clemente0620@gmail.com
<a name="3Lk78"></a>
# License
This system is available under the CC0 1.0 Universal license. See the LICENSE file for more info.
