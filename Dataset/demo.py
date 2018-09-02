from __future__ import print_function, absolute_import, division

import os
import sys
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import voc as PascalVoc


root = '/home/jwwangchn/data/UAV-BD-Release-V1.0.0'
image_set = ['train', 'val', 'test']
preproc = PascalVoc.Preproc(resize=[424, 424], rotation=True)
# preproc = None
voc = PascalVoc.VOC(root, image_set, preproc=preproc, rotation=True, getitem_mode=0)

# visual annotat ions
for idx in np.arange(5):
    img, target = voc.getitem(index = idx)
    voc.vis_anno(img, target)

# resize image and annotation
voc.proc_img(save_path = '/home/jwwangchn/data/UAV-BD-Release-V1.0.0/ICRA2019/Release_resize/images', relative_root=False)
voc.xml2kitti(save_path = '/home/jwwangchn/data/UAV-BD-Release-V1.0.0/ICRA2019/Release_resize/labels', relative_root=False)