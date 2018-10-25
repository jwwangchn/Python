from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='/home/jwwangchn/data/VOCdevkit/UAV-Bottle/UAV-Bottle-V2.0.0'
dataType='JPEGImages'
imgDir = dataDir + '/' + dataType + '/'
annFile='/home/jwwangchn/data/VOCdevkit/instances.json'

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['bottle']);
imgIds = coco.getImgIds(catIds=catIds );
imgIds = coco.getImgIds(imgIds = [20180002250])
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

# display mask
# coco.showAnns(anns)
# plt.show()

# display bbox
img = cv2.imread(imgDir + img['file_name'])
bbox = anns[0]['bbox']
cv2.rectangle(img, (int(bbox[0] - bbox[2]/2.0), int(bbox[1] - bbox[3] / 2.0)), (int(bbox[0] + bbox[2]/2.0), int(bbox[1] + bbox[3] / 2.0)), (255, 0, 0))
cv2.imshow('demo', img)
cv2.waitKey(0)


