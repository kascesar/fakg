import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from fakg.utils.yolo import YoloV2Anchors, bbox2yolo, center_iou
from fakg.dataset.object_detection import decode_and_rescale_bbox
from fakg.utils.img import load_img


class YoloV2Generator(Sequence):
    '''
    YoloV2 generator

    This generator builds the "X" for the YoloV2 model.

        the shape of the output that this class builds are:
            x: (batch_size, grid_x, grid_y, anchors, 4 + 1 + len of classes)

        explanation:
            the las part of the shape are valid for the most YoloV2
            implementations, but not for all, so, pay atention to this point
            if you want to use this generator

            4 + 1 + len of classes are that follows:

            4: are for the predicted x, y, w, h
            1: for objectness
            len of classes: are one hot encoder classes

        Note :

        labels dir and frames dir may vary, so if you have a nother
        frames/labels dir name specify it in the class parameters

    parameters:
    -----------

    dataset : fakg DataSet
        This must be a "DataSet" for "object detection"

    anchors : ndarray
        ndarray with shape (n_anchors * 2, )
        NOT necesary, provide if you have it, this clas may build the anchors

        NOTE : the anchors are scalet to the input shape for the net jet.
        for defect (416, 416)

    input_shape : tuple
        tuple with (w, h) in shape for the yolo9000 NNet
    '''

    def __init__(self, dataset=None,      # Data
                 n_anchors=5,             # Predefined yolo anchors
                 input_shape=(416, 416),  # Input shape (may vary)
                 output_grid=(13, 13),    # Output shape grid
                 anchors=None,            # If you have anchors
                 data_aug_pipe=None,      # Data augmentation fakg ImagePipe
                 ):
        self.dataset = dataset
        self.n_anchors = n_anchors
        self.input_shape = input_shape
        self.output_grid = output_grid
        self.anchors = anchors
        self.n_classes = self.dataset.n_classes
        self.data_aug_pipe = data_aug_pipe
        self.build_anchors()

    def build_anchors(self):
        '''
        Based on the "Yolo9000:Better, Faster, Stronger" article, this function
        chose the best anchors for the given dataset
        '''
        if not self.anchors:
            str = 'Building anchors, because that you dont give it'
            print(str)
            anchors = YoloV2Anchors(dataset=self.dataset.to_dataframe(),
                                    input_shape=self.input_shape)
            self.anchors = anchors.build_anchors(self.n_anchors)

    def bbox_preproces(self, bbox, im_w, im_h):

        # decode and rescale to input shape  as defect (416, 416)
        bbox = decode_and_rescale_bbox(bbox, im_w, im_h, self.input_shape)
        bbox = bbox2yolo(bbox)
        bbox = np.array(bbox, np.float32)
        # scale to grid cel
        #
        # as is said in the paper, the scale is always 32
        alpha = 32
        bbox = bbox / alpha

        return bbox

    def idx_best_anchor(self, box):
        anchors = np.reshape(self.anchors, (-1, 2))
        iou = np.apply_along_axis(lambda anchor: center_iou(anchor, box),
                                  1, anchors)
        return np.argmax(iou)

    def do_groundtruth_noaug(self, example):
        '''
        Do the groundtruth for train, given a single data from fakg DataSet

        parameters:
        ----------

        example data from DataSet fakg
        '''
        img_path = example['information']['frame_path']
        im_width = example['information']['im_width']
        im_height = example['information']['im_height']

        img = load_img(img_path, self.input_shape)
        img = img/255  # normalize image

        x = np.zeros((1, *self.input_shape, 3), np.float32)
        # x = tf.cast(x, tf.float32)
        x[0, ...] = img

        y = np.zeros((1, *self.output_grid, self.n_anchors,
                      4 + 1 + self.n_classes), np.float32)

        for key in example['anotations'].keys():
            # get anotaton
            anotation = example['anotations'][key]

            # get the relevan data from the anotation
            bbox = self.bbox_preproces(anotation['bbox'], im_width, im_height)

            # get the best IoU index from anchors vs bbox
            best_anchor_idx = self.idx_best_anchor([bbox[2], bbox[3]])

            # get in wich part of the grid are the box
            gx, gy = tf.math.floor([bbox[0], bbox[1]]).numpy()
            gx, gy = int(gx), int(gy)  # need int for indexing

            # object class to one hot encoder
            object = self.dataset.query_category(anotation['category'])

            # write the data
            y[0, gx, gy, best_anchor_idx, :4] = bbox
            y[0, gx, gy, best_anchor_idx, 4] = 1.0
            y[0, gx, gy, best_anchor_idx, 5:] = object
            return x, y

    def do_groundtruth_aug(self, example):
        img, objects, bboxs = example

        # process the image
        im_height, im_width = img.shape[:2]  # shape
        img = img/255  # normalize image
        img = cv2.resize(img, (*self.input_shape))

        x = np.zeros((1, *self.input_shape, 3), np.float32)
        x[0, ...] = img

        # procees the yolo output
        y = np.zeros((1, *self.output_grid, self.n_anchors,
                      4 + 1 + self.n_classes), np.float32)

        for bbox, object in zip(bboxs, objects):
            bbox = self.bbox_preproces(bbox, im_width, im_height)
            best_anchor_idx = self.idx_best_anchor([bbox[2], bbox[3]])
            gx, gy = tf.math.floor([bbox[0], bbox[1]]).numpy()
            gx, gy = int(gx), int(gy)  # need int for indexing

            # object class to one hot encoder
            object = self.dataset.query_category(object)
            y[0, gx, gy, best_anchor_idx, :4] = bbox
            y[0, gx, gy, best_anchor_idx, 4] = 1.0
            y[0, gx, gy, best_anchor_idx, 5:] = object
            return x, y

    def xy_maker(self, batch_example):
        batch_size = len(batch_example)
        x_true = np.zeros((batch_size, *self.input_shape, 3), np.float32)
        y_true = np.zeros((batch_size, *self.output_grid, self.n_anchors,
                           4 + 1 + self.n_classes), np.float32)
        #  #  #  #  #  #  #  #  #  #  #  #
        #  here apply data augmentation  #
        #  #  #  #  #  #  #  #  #  #  #  #
        if self.data_aug_pipe:
            batch_example = self.data_aug_pipe(batch_example)
            for i, example in enumerate(batch_example):
                x, y = self.do_groundtruth_aug(example)
                x_true[i] = x
                y_true[i] = y
        else:
            for i, example in enumerate(batch_example):
                x, y = self.do_groundtruth_noaug(example)
                x_true[i] = x
                y_true[i] = y
        return x_true, y_true

    def on_epoch_end(self):
        self.dataset.on_epoch_end()

    def __len__(self):
        return self.dataset.batch_len()

    def __getitem__(self, index):
        batch_example = self.dataset.get_batch(index)
        return self.xy_maker(batch_example)
