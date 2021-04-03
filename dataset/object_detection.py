from fakg.utils.files import (only_xml, only_txt, only_imgs,
                              make_complete_paths, get_im_wh)
import xml.etree.ElementTree as ET
import pandas as pd
import os


'''
Here wee explain the dict structure for the one example of data

The next dictionary are maded to have all information necesary for traingin
models in object detection, the structure are maded, witd in mind,
autosustemtable information.

{
 'information': {
                 filename   : str
                 frame_path : str
                 type       : str
                 im_height  : int
                 im_width   : int
                }

 'anotations' : {

                 1: {
                      'bbox'     : {
                                     'xmin': int
                                     'ymin': int
                                     'xmax': int
                                     'ymax': int
                                    }
                      'category' : str
                     }
                 .
                 .
                 .
                 n: {...}
                }

}


a detalied explaination of every camp are bellow

- anotations
    - i  # num of annotation
        - bbox     : bounding box
        - category : obj name

- informations
    - filename    : img name
    - frame_path  : path of file
    - im_height   : image h
    - im_width    : image w
    - type        : task type (clasification, detection)
'''


# VOC data set
def read_voc_anotation(xml_file: str, path2frames: str):
    '''
    read content from XML VOC anotation

    Return : dict
        a dict with all relevant information form this anotation
          keys:
            - annotations
                - i  # num of annotation
                    - bbox     : bounding box
                    - category : obj name

            - information
                - filename    : img name
                - frame_path  : path of file
                - im_height   : image h
                - im_width    : image w
                - type        : task type (clasification, detection)
    '''
    tree = ET.parse(xml_file)
    root = tree.getroot()
    anotation_dict = {}
    anotation_dict['information'] = {}
    anotation_dict['anotations'] = {}

    #
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)
    filename = root.find('filename').text
    frame_path = os.path.join(path2frames, filename)

    anotation_dict['information']['filename'] = filename
    anotation_dict['information']['frame_path'] = frame_path
    anotation_dict['information']['im_width'] = width
    anotation_dict['information']['im_height'] = height
    anotation_dict['information']['type'] = 'object detection'
    for i, boxes in enumerate(root.iter('object')):

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        bbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        object = boxes.find("name").text

        # put information
        anotation_dict['anotations'][i] = {}
        anotation_dict['anotations'][i]['category'] = object.strip()
        anotation_dict['anotations'][i]['bbox'] = bbox
    return anotation_dict


def read_voc_dataset(path2data: str,
                     labels_dir='labels', frames_dir='frames'):
    '''
    build a dictionary that contains all anotation an their data information
    with data in VOC format.

    parameters:
    -----------

    path2data : str
        path where are the data, must a folder with 2 sub-folders, label's and
        frame's forlders

    return : Dataset in fakg format
    '''

    path2labels = os.path.join(path2data, labels_dir)
    anotations = only_xml(path2labels)
    anotations = make_complete_paths(path2labels, anotations)

    path2frames = os.path.join(path2data, frames_dir)

    # build dict
    to_return = []

    for anotation in anotations:
        label_info = read_voc_anotation(anotation, path2frames)
        if len(label_info['anotations'].keys()) < 1:
            # discard data with out labels
            continue
        to_return.append(label_info)

    return to_return


# yolo dataset
def read_yolo_anotation(txt_file: str, img_file: str, labels: str):
    '''
    Return : dict
        a dict with all relevant information form this anotation
          keys:
            - anotations
                - i  # num of annotation
                    - bbox     : bounding box
                    - category : obj name

            - information
                - filename    : img name
                - frame_path  : path of file
                - im_height   : image h
                - im_width    : image w
                - type        : task type (clasification, detection)
    '''
    # read anotation file
    f = open(txt_file)
    anotations = f.readlines()
    f.close()

    # read .labels file
    f = open(labels)
    labels = f.readlines()
    labels = [obj.strip() for obj in labels]
    f.close()

    anotation_dict = {}
    anotation_dict['information'] = {}
    anotation_dict['anotations'] = {}

    # decode file information
    # filename
    filename = img_file.split('/')[-1]

    # img path
    frame_path = img_file

    # img size
    im_width, im_height = get_im_wh(frame_path)

    anotation_dict['information']['filename'] = filename
    anotation_dict['information']['frame_path'] = frame_path
    anotation_dict['information']['im_width'] = im_width
    anotation_dict['information']['im_height'] = im_height
    anotation_dict['information']['type'] = 'object detection'
    for i, anotation in enumerate(anotations):
        # read parameters
        label_idx, x, y, w, h = anotation.split()

        # define parameters.
        # object type
        object = labels[int(label_idx)].strip()

        # define bbox
        x, w, y, h = float(x), float(w), float(y), float(h)
        xmin = int((x - w/2) * im_width)
        ymin = int((y - h/2) * im_height)
        xmax = int((x + w/2) * im_width)
        ymax = int((y + h/2) * im_height)

        # put information
        bbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        anotation_dict['anotations'][i] = {}
        anotation_dict['anotations'][i]['category'] = object
        anotation_dict['anotations'][i]['bbox'] = bbox
    return anotation_dict


def read_yolo_dataset(path2data: str, names_file: str,
                      labels_dir='labels', frames_dir='frames'):
    '''
    build a dictionary that contains all anotation an their data information
    whith data in yolo format.

    parameters:
    -----------

    path2data : str
        path where are the data, must a folder with 2 sub-folders, label's and
        frame's forlders

    labels_file : str
        file with all classes, in the yolo format dataset.
        like : coco.labels

    labels_dir : str
        name of the subdirectory where are all the labels

    frames_dir: str
        name of the subdirectory where are all the frames

    return : Dataset in fakg format
    '''
    path2labels = os.path.join(path2data, labels_dir)
    path2frames = os.path.join(path2data, frames_dir)
    anotations = only_txt(path2labels)
    anotations = make_complete_paths(path2labels, anotations)
    frames = only_imgs(path2frames)
    frames = make_complete_paths(path2frames, frames)

    # check if the amount of images are equal or more the anotations
    if len(anotations) > len(frames):
        raise ValueError("there is more anotation than frames, can't continue")

    # make pairs path for decode later
    pairs = set()
    for anotation in anotations:
        anotated = False
        anotation_name = anotation.split('/')[-1].split('.')[0]
        for frame in frames:
            frame_name = frame.split('/')[-1].split('.')[0]
            if frame_name == anotation_name:
                pairname = anotation + '@#$' + frame
                pairs.add(pairname)
                anotated = True
                break
        if not anotated:
            raise ValueError('the is no frame that match with the given '
                             'label\ngiven label {}'.format(anotation_name))
    to_return = []
    for par in pairs:
        anotation, frame = par.split('@#$')
        label_info = read_yolo_anotation(anotation, frame, names_file)
        if len(label_info['anotations'].keys()) < 1:
            # discard data with out labels
            continue
        to_return.append(label_info)

    return to_return


def read_dataframe_dataset(df: pd.DataFrame):
    '''
    Read a pandas DataFrame in fakg format
    '''

    data = []

    for frame_path in df['frame_path'].unique():
        # parse information
        annotation_dict = {}
        annotation_dict['information'] = {}
        annotation_dict['anotations'] = {}
        int_df = df[df['frame_path'] == frame_path]

        filename = int_df['filename'].unique()[0]
        im_width = int_df['im_width'].unique()[0]
        im_height = int_df['im_height'].unique()[0]

        annotation_dict['information']['filename'] = filename
        annotation_dict['information']['im_width'] = im_height
        annotation_dict['information']['im_height'] = im_width
        annotation_dict['information']['frame_path'] = frame_path

        for i, row in enumerate(int_df.iterrows()):
            _, row = row
            category = row['category'].strip()
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']

            annotation_dict['anotations'][i] = {}
            annotation_dict['anotations'][i]['bbox'] = {}
            annotation_dict['anotations'][i]['bbox']['xmin'] = xmin
            annotation_dict['anotations'][i]['bbox']['xmax'] = xmax
            annotation_dict['anotations'][i]['bbox']['ymin'] = ymin
            annotation_dict['anotations'][i]['bbox']['ymax'] = ymax
            annotation_dict['anotations'][i]['category'] = category
        data.append(annotation_dict)

    return data


def dataset_frame(dataset: list):
    '''
    analize the dataset to get relevant information

    paramters:
    ----------

    dataset: dict
        Dataset in fackg format
    '''

    datainfo = {'filename': [], 'frame_path': [],
                'im_width': [], 'im_height': [], 'category': [],
                'xmin': [], 'ymin': [], 'xmax': [], 'ymax': []}
    dataset = dataset
    for label_dict in dataset:
        # separate all info

        # 1. frame information
        filename = label_dict['information']['filename']
        frame_path = label_dict['information']['frame_path']
        im_width = label_dict['information']['im_width']
        im_height = label_dict['information']['im_height']
        for anotation_key in label_dict['anotations'].keys():

            # 2. label information
            anotation = label_dict['anotations'][anotation_key]
            object = anotation['category']
            bbox = anotation['bbox']
            xmin, ymin = bbox['xmin'], bbox['ymin']
            xmax, ymax = bbox['xmax'], bbox['ymax']

            # asign
            datainfo['filename'].append(filename)
            datainfo['frame_path'].append(frame_path)
            datainfo['im_width'].append(im_width)
            datainfo['im_height'].append(im_height)

            datainfo['category'].append(object)
            datainfo['xmin'].append(xmin)
            datainfo['ymin'].append(ymin)
            datainfo['xmax'].append(xmax)
            datainfo['ymax'].append(ymax)

    # buidl dataframe
    df = pd.DataFrame(datainfo, columns=['filename', 'frame_path',
                                         'im_width', 'im_height', 'category',
                                         'xmin', 'ymin', 'xmax', 'ymax'])
    return df


def decode_and_rescale_bbox(bbox, im_w, im_h, net_input_shape):
    alpha = net_input_shape[0]/im_w
    betha = net_input_shape[1]/im_h
    xmin, ymin = bbox['xmin'], bbox['ymin']
    xmax, ymax = bbox['xmax'], bbox['ymax']
    return int(xmin*alpha), int(ymin*betha), int(xmax*alpha), int(ymax*betha)
