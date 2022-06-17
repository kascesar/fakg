'''
FAKG ([F]ucking [A]wesome [K]eras [G]enerators)

This project aimed to solve the generic generatos for most commun Neural Nets
archiquectures like Yolo.

This package contains 3 importat functionalities:

[1] Dataset problems:

    Often solving problems reading various dataset in diferent format can be
    challenge. To face this, fakg implelemt the "DataSet" class, that read
    varoius kind of dataset in diferents format or estructure

    example:
    --------
    * form fakg import DataSet
    * ds = DataSet()
    * ds.read_yolo(path2data, labels_dir, frames_dir, classes_file)
    * ds.read_voc(path2data, labels_dir, frames_dir)

    In this example, the DataSet class read 2 kind of dataset for the same
    task (object detection)

[2] Data augmentatin:

    Data augmentation are very importante in every NN training pipeline,
    to evade (or try to) overfiting. Fakg has a lot augmenters on it for
    image data.
    * from fakg.augmenters import RandomCrop, RandomRotation, ImagePipe

    ap = ImagePipe()
    ap.add([RandomCrop(), RandomRotation()])

    --------------------------------------------------------------------------
    NOTE:

    Check for all augmenters making "fakg.augmenters?"
    [it will be more augmenters in the near future for other kind of data]
    --------------------------------------------------------------------------


[3] Generators problems:

    The generator are the pice of a training pipeline that take the data and
    build  X, y for compute loss in the neural net training

    example:
    --------
    * from fakg import YoloV2Generator

    * generator = YoloV2Generator(dataset, augmenter_pipe=ap)
    where dataset are fakg DataSet object
    NOTE: the fakg generator is heredated class from
          tensorflow.keras.utils.Sequence, so it has every propieties form
          Sequence object

conclusion:

follow this steps we build a robust datagenerator for object detection to
trains a yolo v2 model

'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from fakg.dataset.dataset import DataSet
from fakg.generators.object_detection import YoloV2Generator
from fakg.generators.videos.clasification.from_img_files import Vidcg as VideoGenerator_FromFrames

__all__ = ['DataSet', 'YoloV2Generator', 'VideoGenerator_FromFrames']
