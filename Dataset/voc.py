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
COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255),
          (0, 255, 255), (255, 0, 255), (255, 255, 0))

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

class Preproc(object):

    def __init__(self, resize=None, rotation=False):
        self.resize = resize
        self.rotation = rotation

    def __call__(self, image, targets=None):
        
        image, targets = self._resize(image=image,
                                        targets=targets, 
                                        resize=self.resize)
        return image, targets
    
    def _resize(self, image, targets, resize=None):
        assert (image is not None) or (targets is not None), "Please set mode = 0 in getitem, when use Preproc class"
        H, W, _ = image.shape
        if resize == None:
            x_scale = 1
            y_scale = 1
        else:
            x_scale = resize[0] / W
            y_scale = resize[1] / H
        # 1. resize image
        image = cv2.resize(image, (int(W * x_scale), int(H * y_scale)))

        # 2. resize bbox
        if self.rotation:
            coordinate = []
            for target in targets:
                labels = target[-1]

                box = cv2.boxPoints(((target[0], target[1]), (target[2], target[3]), target[4] * 180 / np.pi))
                box = np.reshape(box, [-1, ])
                box = np.array(box)

                box[::2] = box[::2] * x_scale
                box[1::2] = box[1::2] * y_scale
                
                box = box.reshape([4, 2])
                rect = cv2.minAreaRect(box)
                cx, cy, w, h, angle = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
                angle = angle * np.pi / 180.0
                if angle < 0:
                    angle = angle + np.pi
                coordinate.append([cx, cy, w, h, angle, labels])
            boxes = np.array(coordinate)
        else:
            targets[:, 0] = x_scale * targets[:, 0]
            targets[:, 1] = y_scale * targets[:, 1]
            targets[:, 2] = x_scale * targets[:, 2]
            targets[:, 3] = y_scale * targets[:, 3]

        return image, boxes

class VOC:
    def __init__(self, root, image_set=['train', 'val', 'test'], preproc=None, rotation=False, getitem_mode=0):
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
        self.getitem_mode = getitem_mode
        self.target_transform = AnnotationTransform(rotation=rotation)
        self.preproc = preproc
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
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return img, target
    
    def len(self):
        '''
        Arguments:
            No
        Returns:
            The number of image set, when image_set = 'all', return whole dataset's annotation number
        '''
        return len(self.ids)

    def generate_image_set(self, save_path, trainval_percentage=0.8, train_percentage=0.8, relative_root=True, label_path=None):
        '''
        Arguments:
            save_path : save .txt files
            trainval_percentage: default is 0.8
            train_percentage: default is 0.8
        Returns:
            No
        '''
        assert self.image_set == 'all', "image_set must be 'all'"
        if relative_root:
            save_path = os.path.join(self.root, save_path.split('/')[-1])
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if label_path == None:
            annofile_array = np.array(self.ids)[:, 1]
            annofile_num = self.len()
        else:
            annofile_list = []
            for annofile in os.listdir(label_path):
                filename = annofile.split('.')[0]
                annofile_list.append(filename)
            annofile_array = np.array(annofile_list)
            annofile_num = len(annofile_list)

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
            img, targets = self.getitem(index, mode=self.getitem_mode)
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
    
    def vis_anno(self, img, targets):
        if self.rotation:
            for target in targets:
                rect = ((int(target[0]), int(target[1])), (int(target[2]), int(target[3])), int(target[4] * 180 / np.pi))
                rect = cv2.boxPoints(rect)
                rect = np.int0(rect)
                cv2.drawContours(img, [rect], 0, COLORS[0], 3)
            cv2.imshow("vis", img)
            cv2.waitKey(0)
    
    def proc_img(self, save_path, relative_root=True):
        if relative_root:
            save_path = os.path.join(self.root, save_path.split('/')[-1])
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for index in np.arange(self.len()):
            file_name = self.ids[index][-1]
            print("Proc {}".format(file_name))
            img, targets = self.getitem(index, mode=self.getitem_mode)
            file_name = os.path.join(save_path, file_name + '.jpg')
            cv2.imwrite(file_name, img)
        print("Finish processing and saving images!")

if __name__ == "__main__":
    root = '/home/jwwangchn/data/UAV-BD-Release-V1.0.0'
    image_set = ['train', 'val', 'test']
    preproc = Preproc(resize=[424, 240], rotation=True)
    # preproc = None
    voc = VOC(root, image_set, preproc=preproc, rotation=True, getitem_mode=0)

    # visual annotat ions
    # for idx in np.arange(5):
    #     img, target = voc.getitem(index = idx)
    #     voc.vis_anno(img, target)

    # resize image and annotation
    # voc.proc_img(save_path = '/home/jwwangchn/data/UAV-BD-Release-V1.0.0/ICRA2019/Release_resize/images', relative_root=False)
    voc.xml2kitti(save_path = '/home/jwwangchn/data/UAV-BD-Release-V1.0.0/ICRA2019/Release_resize/labels', relative_root=False)

