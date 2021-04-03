import os
import pandas as pd
from fakg.dataset.object_detection import (read_yolo_dataset,
                                           read_voc_dataset,
                                           read_dataframe_dataset,
                                           dataset_frame
                                           )
from fakg.utils.ml import OneHotEncoder
from fakg.utils.img import plot_example


class DataSet:
    def __init__(self, batch_size=3):
        self.type = None
        self.data = None
        self.seen_paths = set()
        # other options
        self.batch_size = batch_size

    def to_dataframe(self):
        if not self.data:
            print("this DataSet is empty, can't do anything.")
        elif self.type == 'object detection':
            return dataset_frame(self.data)
        else:
            print('there is no data to convert')

    def read_voc(self, path2data: str,
                 labels_dir='labels', frames_dir='frames'):

        # chekck if the data are addeed jet
        existence = self.test_existence(path2data=path2data,
                                        labels_dir=labels_dir,
                                        frames_dir=frames_dir)

        # add the data
        data = read_voc_dataset(path2data=path2data,
                                labels_dir=labels_dir,
                                frames_dir=frames_dir)

        self.add_data(data, existence, type='object detection')

    def read_yolo(self, path2data: str, names_file=None,
                  labels_dir='labels', frames_dir='frames'):
        # chekck if the data are addeed jet
        existence = self.test_existence(path2data=path2data,
                                        labels_dir=labels_dir,
                                        frames_dir=frames_dir)
        # add the data
        data = read_yolo_dataset(path2data=path2data,
                                 names_file=names_file,
                                 labels_dir=labels_dir,
                                 frames_dir=frames_dir)

        self.add_data(data, existence, type='object detection')

    def read_dataframe(self, df):
        '''
        Read dataset in DataFrame fakg format
        '''
        data = read_dataframe_dataset(df)
        self.add_data(data, False, type='object detection')

    def test_existence(self, path2data, frames_dir=None, labels_dir=None):
        '''
        check if the path has been seen before, of course, that is to evade
        duplicate datasets
        '''
        p2d = False
        frames = False
        labels = False
        if frames_dir:
            path2frames = os.path.join(path2data, frames_dir)
            if path2frames in self.seen_paths:
                print('This data are added before.')
                frames = True
            else:
                self.seen_paths.add(path2frames)
        if labels_dir:
            path2labels = os.path.join(path2data, labels_dir)
            if path2labels in self.seen_paths:
                print('This data are added before.')
                labels = True
            else:
                self.seen_paths.add(path2frames)

        if (not frames_dir) and (not labels_dir):
            if path2data in self.seen_paths:
                print('This data are added before.')
                p2d = True
            else:
                self.seen_paths.add(path2data)

        if (not frames_dir) and (not labels_dir) and (p2d):
            return True

        elif (frames_dir or labels_dir) and (frames or labels):
            return True
        else:
            return False

    def add_data(self, data, existence, type: str):
        if not self.data and not existence:
            self.data = data
            self.type = type
            self.clasification_task()
        elif self.type == type and not existence:
            self.data.extend(data)
            self.clasification_task()
        elif existence:
            pass
        else:
            raise Exception('You tray to add an "object detection" dataset.'
                            'You can not add this kind of data because\n'
                            'this DataSet object is for {}'.format(self.type))

    def choice_classes(self, classes: list):
        df = self.to_dataframe()

        # check if the class list is well given
        actual_classes = df['category'].unique()
        for cla in classes:
            if cla not in actual_classes:
                raise ValueError('The class "{}" are not in the dataset'
                                 ''.format(cla))

        big_df = None
        for cla in classes:
            df_int = df[df['category'] == cla]
            if big_df is None:
                big_df = df_int
            else:
                big_df = pd.concat([big_df, df_int])
        self.data = read_dataframe_dataset(big_df)

    def plot_example(self):
        plot_example(self.data)

    def batch_len(self):
        from numpy import modf
        dec, integer = modf(len(self.data)/self.batch_size)

        if dec == 0.0:
            return int(integer)
        else:
            return int(integer + 1.0)

    def get_batch(self, i):
        i_0 = i*self.batch_size
        i_1 = self.batch_size*i + self.batch_size
        return(self.data[i_0: i_1])

    def on_epoch_end(self):
        from numpy.random import shuffle
        shuffle(self.data)

    def n_classes(self):
        '''
        get how mutch classes have this DataSet
        '''
        if not self.data:
            print('this DataSet is empty, nothing to do')

        suported_type = ['object detection', 'image clasification',
                         'video clasification']

        if self.type in suported_type:
            self.n_classes = len(self.to_dataframe()['category'].unique())

    def clasification_task(self):
        if not self.data:
            print('this DataSet is empty, nothing to do')

        suported_type = ['object detection', 'image clasification',
                         'video clasification']

        if self.type in suported_type:
            self.n_classes = len(self.to_dataframe()['category'].unique())
            self.onehot = OneHotEncoder(self.to_dataframe())

    def query_category(self, val):
        if not self.data:
            print('this DataSet is empty, nothing to do')
            return
        suported_type = ['object detection', 'image clasification',
                         'video clasification']
        if self.type in suported_type:
            return self.onehot.query(val)
