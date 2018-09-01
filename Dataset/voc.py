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

VOC_CLASSES = ( '__background__', 'bottle')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

class AnnotationTransform(object):
    def __init__(self, class_to_ind=None, keep_difficult=True, rotation=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult
        self.rotation=rotation

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        if self.rotation:
            res = np.empty((0, 6))
        else:
            res = np.empty((0, 5))
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            if self.rotation:
                bbox = obj.find('robndbox')
                pts = ['cx', 'cy', 'w', 'h', 'angle']
            else:
                bbox = obj.find('bndbox')
                pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = float(bbox.find(pt).text)
                # scale height or width
                #cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

class VOC:
    def __init__(self, root, image_set=['train', 'val', 'test'], rotation=False):
        '''
        Arguments:
            root : dataset's root path
            image_set : [train, test, val] or 'all', default is 'train',
                        'all' -> find all annotation file in annotation path
            rotation : True -> rbbox, False -> bbox, default is False(bbox)
        Returns:
            
        '''
        self.root = root
        self.image_set = image_set
        self.rotation = rotation
        self.target_transform = AnnotationTransform(rotation=rotation)
        self._anno_path = os.path.join(self.root, 'Annotations')
        self._anno_file = os.path.join('%s', 'Annotations', '%s.xml')
        self._img_path = os.path.join(self.root, 'JPEGImages')
        self._img_file = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()

        if self.image_set == 'all':
            for anno_file in os.listdir(self._anno_path):
                    self.ids.append((self.root, anno_file.split('.')[0]))
        else:
            for name in self.image_set:
                for line in open(os.path.join(self.root, 'ImageSets', 'Main', name + '.txt')):
                    self.ids.append((self.root, line.strip()))

    def getitem(self, index,  mode=0):
        """
        mode: 0 -> return image and target, 1 -> return image, target=None, 2 -> reutrn target, image=None
        """
        img_id = self.ids[index]
        if mode == 0 or mode == 2:
            target = ET.parse(self._anno_file % img_id).getroot()
            target = self.target_transform(target)
        else:
            target = None
        if mode == 0 or mode == 1:
            img = cv2.imread(self._img_file % img_id, cv2.IMREAD_COLOR)
        else:
            img = None
        return img, target
    
    def len(self):
        '''
        Arguments:
            No
        Returns:
            The number of image set, when image_set = 'all', return whole dataset's annotation number
        '''
        return len(self.ids)

    def generate_image_set(self, save_path, trainval_percentage=0.8, train_percentage=0.8, relative_root=True):
        '''
        Arguments:
            save_path : save .txt files
            trainval_percentage: default is 0.8
            train_percentage: default is 0.8
        Returns:
            No
        '''
        print("Note: please set image_set = 'all'")
        if relative_root:
            save_path = os.path.join(self.root, save_path.split('/')[-1])
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        annofile_array = np.array(self.ids)[:, 1]
        annofile_num = self.len()
        np.random.shuffle(annofile_array)

        trainval_num = int(annofile_num * trainval_percentage)
        train_num = int(annofile_num * trainval_percentage * train_percentage)
        val_num = trainval_num - train_num
        test_num = annofile_num - trainval_num
        
        trainval_list = annofile_array[0: trainval_num - 1]
        train_list = annofile_array[0: train_num - 1]
        val_list = annofile_array[train_num - 1: trainval_num - 1]
        test_list = annofile_array[trainval_num - 1: annofile_num]

        np.savetxt(os.path.join(save_path, 'trainval.txt'), trainval_list, fmt="%s")
        np.savetxt(os.path.join(save_path, 'train.txt'), train_list, fmt="%s")
        np.savetxt(os.path.join(save_path, 'val.txt'), val_list, fmt="%s")
        np.savetxt(os.path.join(save_path, 'test.txt'), test_list, fmt="%s")
        print("trainval: {}, train: {}, val: {}, test: {}".format(trainval_num, train_num, val_num, test_num))
        print("Finish generating image set!")

    def xml2kitti(self, save_path, relative_root=True):
        '''
        Arguments:
            save_path : Path to save kittl label
        Returns:
            No
        '''
        if relative_root:
            save_path = os.path.join(self.root, save_path.split('/')[-1])
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for index in np.arange(self.len()):
            file_name = self.ids[index][-1]
            print("Converting {}".format(file_name))
            img, targets = self.getitem(index, mode=2)
            labelkitti = open(os.path.join(save_path, file_name+'.txt'), 'w')
            
            for target in targets:
                name = VOC_CLASSES[int(target[-1])]
                if self.rotation:
                    cx, cy, w, h, angle = target[0], target[1], target[2], target[3], target[4]
                    box = [cx, cy, w, h, angle]
                else:
                    xmin, ymin, xmax, ymax = target[0], target[1], target[2], target[3]
                    angle = 0
                    box = [xmin, ymin, xmax, ymax, angle]
                write_str = "{} {:.2f} {:.0f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}{}".format(name, 0, 0, 0, box[0], box[1], box[2], box[3], 0, 0, 0, 0, 0, 0, box[4], '\n')
                labelkitti.write(write_str)

        print("Finish convert XML to KITTI!")

    def extract_test_img(self, save_path, relative_root=True):
        '''
        Arguments:
            save_path : save test images
        Returns:
            No
        '''
        assert len(self.image_set) == 1 and self.image_set[0] == 'test', "image_set must be 'test'"
        if relative_root:
            save_path = os.path.join(self.root, save_path.split('/')[-1])
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for index in np.arange(self.len()):
            file_name = self.ids[index][1]
            print("Copying {}".format(file_name))
            shutil.copy(os.path.join(self._img_path, file_name + '.jpg'), save_path)
        print("Finish copy all test files!")
    
if __name__ == "__main__":
    root = '/home/jwwangchn/data/UAV-BD-Release-V1.0.0/ICRA2019'
    image_set = ['test']
    voc = VOC(root, image_set, rotation=True)
    voc.extract_test_img(save_path = './temp', relative_root=True)