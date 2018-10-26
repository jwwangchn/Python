import xml.etree.ElementTree as ET
import os
import json
import numpy as np
import cv2

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_set = dict()
image_set = set()

category_item_id = 0
image_id = 20180000000
annotation_id = 0

def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id

def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id

def addAnnoItem(object_name, image_id, category_id, bbox, rbbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    #bbox[] is x,y,w,h
    #left_top
    seg.append(rbbox[0])
    seg.append(rbbox[1])
    #left_bottom
    seg.append(rbbox[2])
    seg.append(rbbox[3])
    #right_bottom
    seg.append(rbbox[4])
    seg.append(rbbox[5])
    #right_top
    seg.append(rbbox[6])
    seg.append(rbbox[7])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)


def forward_convert(coordinate):
    """
    :param coordinate: format [x_c, y_c, w, h, theta]
    :return: format [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    boxes = []
    rect = coordinate
    box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
    box = np.reshape(box, [-1, ])
    boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]])

    return np.array(box, dtype=np.float32)


def parseXmlFiles(xml_path): 
    for f in os.listdir(xml_path):
        if not f.endswith('.xml'):
            continue
        # image_name = f.split('.')[0]+'.jpg'
        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None

        xml_file = os.path.join(xml_path, f)
        # print(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        #elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None
            
            if elem.tag == 'folder':
                continue
            
            if elem.tag == 'filename':
                file_name = elem.text
                # file_name = image_name
                if file_name in category_set:
                    raise Exception('file_name duplicated')
                
            #add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size)
                    # print('add image with {} and {}'.format(file_name, size))
                else:
                    raise Exception('duplicated image: {}'.format(file_name)) 
            #subelem is <width>, <height>, <depth>, <name>, <bndbox>
            for subelem in elem:
                bndbox['cx'] = None
                bndbox['cy'] = None
                bndbox['w'] = None
                bndbox['h'] = None
                bndbox['angle'] = None
                
                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    if object_name not in category_set:
                        current_category_id = addCatItem(object_name)
                    else:
                        current_category_id = category_set[object_name]

                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)

                #option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == 'robndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = float(option.text)

                #only after parse the <object> tag
                if bndbox['cx'] is not None:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    cx, cy, w, h, angle = bndbox['cx'], bndbox['cy'], bndbox['w'], bndbox['h'], bndbox['angle']*180.0/np.pi
                    minAreaBox = forward_convert([cx, cy, w, h, angle])
                    
                    rbbox = []
                    rbbox.append(round(minAreaBox[0], 2))
                    rbbox.append(round(minAreaBox[1], 2))
                    rbbox.append(round(minAreaBox[2], 2))
                    rbbox.append(round(minAreaBox[3], 2))
                    rbbox.append(round(minAreaBox[4], 2))
                    rbbox.append(round(minAreaBox[5], 2))
                    rbbox.append(round(minAreaBox[6], 2))
                    rbbox.append(round(minAreaBox[7], 2))


                    xmin = round(min(minAreaBox[::2]), 2)
                    ymin = round(min(minAreaBox[1::2]), 2)
                    xmax = round(max(minAreaBox[::2]), 2)
                    ymax = round(max(minAreaBox[1::2]), 2)
                    
                    #x
                    bbox_x = round(xmin, 2)
                    bbox.append(round(xmin, 2))
                    #y
                    bbox_y = round(ymin, 2)
                    bbox.append(round(ymin, 2))
                    #w
                    bbox_w = xmax - xmin
                    bbox.append(xmax - xmin)
                    #h
                    bbox_h = ymax - ymin
                    bbox.append(ymax - ymin)
                    if bbox_w == 0 or bbox_h == 0:
                        # print(cx, cy, w, h, angle)
                        # print(minAreaBox)
                        print(bbox_x, bbox_y, bbox_w, bbox_h)
                        print('error!!!!!!!!!!!!!!!!!!!!')
                    
                    # print('add annotation with {},{},{},{}, {}'.format(object_name, current_image_id, current_category_id, bbox, rbbox))
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox, rbbox)

if __name__ == '__main__':
    xml_path = '/home/jwwangchn/data/DOTA_KITTI/train/labeltxt'
    json_file = '/home/jwwangchn/data/DOTA_KITTI/train/dota_rbbox.json'

    # xml_path = '/home/jwwangchn/data/VOCdevkit/UAV-Bottle/UAV-Bottle-V2.0.0/Annotations_rbbox'
    # json_file = '/home/jwwangchn/data/VOCdevkit/UAV-Bottle/UAV-Bottle-V2.0.0/uav_bd_rbbox.json'

    parseXmlFiles(xml_path)
    json.dump(coco, open(json_file, 'w'))
