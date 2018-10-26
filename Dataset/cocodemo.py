from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

imgDir = '/home/jwwangchn/data/DOTA_KITTI/train/images/'
annFile='/home/jwwangchn/data/DOTA_KITTI/train/dota_rbbox.json'

# imgDir = '/home/jwwangchn/data/VOCdevkit/UAV-Bottle/UAV-Bottle-V2.0.0/JPEGImages/'
# annFile='/home/jwwangchn/data/VOCdevkit/UAV-Bottle/UAV-Bottle-V2.0.0/uav_bd_rbbox.json'

# imgDir = '/home/jwwangchn/data/COCO/val2017/'
# annFile='/home/jwwangchn/data/COCO/annotations/instances_val2017.json'

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['harbor']);
imgIds = coco.getImgIds(catIds=catIds);
imgIds = coco.getImgIds(imgIds = [20180000001])      # 555705, cat
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

# load and display image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
I = io.imread(imgDir + img['file_name'])


# load and display instance annotations
plt.imshow(I); 
plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
print(anns)
# display mask
# coco.showAnns(anns)
# plt.show()

# display bbox
img = cv2.imread(imgDir + img['file_name'])
bbox = anns[1]['bbox']
print(bbox)
cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (255, 0, 0))
# cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 3)

cv2.imshow('demo', img)
cv2.waitKey(8000)


