import os
import numpy as np
import cv2
import pandas as pd


class AnomalyGenerator:
    ''' Anomaly Generator

    This generator is for training models that recibe
    just 1 type of data examples, like a security camera.

    Parameters:
    ===========
    data_path: Path where are the date

    Fakg Anomaly Example:
    {    'type': 'AnomalyExample'
      'example': {'path_to_video': '/root/sub_dir/frame_of_videos',
                   'video_frames': [frame_1, ..., frame_n]
                 }
    }
    '''
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.dirs = os.listdir(data_path)

    def video_paths(self):
        '''
        '''
        paths = []
        for folder in self.dirs:
            path = os.path.join(self.data_path, folder)
            paths.append(path)
        return paths

    def build_all_examples(self):
        '''
        This builds all examples.

        An example for this class, is an a dictionary that have some
        propieties:


        parameters
        ----------

        returns
        -------

        example_i = {'path_to_video': '/root/dir/file.extension',
                     'category': 'categori_name',
                     'video_inds': range(0, 1, 2, 3 ..., n)}


        NOTE: remember that dir is the categori of video.
        '''
        def get_pack_frames(path):
            all_inds_frames = []
            base_window = range(self.window_size)
            cap = FramesConstructor(path)
            video_length_frames = cap.get_frames_count()

            for step in self.strides:
                base_window_augmented = [i + i*(step - 1) for i in base_window]
                for i in range(video_length_frames - self.window_size):
                    frames_inds = [n + i for n in base_window_augmented]
                    if frames_inds[-1] > video_length_frames - 1:
                        break
                    all_inds_frames.append(frames_inds)
            return all_inds_frames

        examples = []
        for video_path in self.videos_all_paths:
            category = video_path.split('/')[-2]
            for inds_frames in get_pack_frames(video_path):
                example = {}
                # asign video path
                example['path_to_video'] = video_path
                # assingn category
                example['category'] = category
                # asing inds
                example['video_inds'] = inds_frames
                examples.append(example)
        examples_random = self.add_random_steeps_examples(
                                                random_steps=self.random_steps)
        if self.steps_augmentation_random:
            return examples + examples_random
        else:
            return examples

    def add_random_steeps_examples(self, random_steps=None):
        '''
        Make more examples with random steps augmentation methot

        Augmented data methot bases on in random walker over a given (or not)

        Parameters
        ----------
        random_steeps : list
            list with random steps for the metoth
            ex: [1, 2, 3, 4]
        '''
        if not self.steps_augmentation_random:
            print('no steps_augmentation_random technique are used')
            return
        if self.random_steps is None:
            print('automatically setting the random steps equal to'
                  '{}'.format(self.strides))
            self.random_steps = self.strides

        def set_random_steps(list_, random_steps):
            '''
            sum number randomly to a list
            '''
            inds = range(len(list_))
            for i in inds:
                old_val = list_[i]
                new_val = old_val + choice(random_steps)
                list_[i] = new_val
                if not i == inds[-1]:
                    list_[i + 1] = new_val
                if not i == inds[0]:
                    if list_[i] <= list_[i - 1]:
                        list_[i] = new_val + 1
            return list_

        def get_pack_frames(path):
            all_inds_frames = []
            tester = set()
            base_window = range(self.window_size)
            cap = FramesConstructor(path)
            video_length_frames = cap.get_frames_count()
            for i in range(video_length_frames - self.window_size):
                frames_inds = [n + i for n in base_window]
                frames_inds = set_random_steps(frames_inds, self.random_steps)
                if frames_inds[-1] > video_length_frames - 1:
                    continue

                # add the inds to tester for check if wee not make a same inds
                # (due for randome nature for this task)
                string_ind = ''
                for j in frames_inds:
                    string_ind += str(j)
                if string_ind not in tester:
                    all_inds_frames.append(frames_inds)
                    tester.add(string_ind)
                    continue
            return all_inds_frames
        examples = []

        for video_path in self.videos_all_paths:
            category = video_path.split('/')[-2]
            for inds_frames in get_pack_frames(video_path):
                example = {}
                # asign video path
                example['path_to_video'] = video_path
                # assingn category
                example['category'] = category
                # asing inds
                example['video_inds'] = inds_frames
                examples.append(example)
        return examples

