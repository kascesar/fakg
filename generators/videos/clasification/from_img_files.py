import cv2
from fakg.fakg_utils.fakg_img import togray, torgb
from fakg.fakg_utils.fakg_files import FramesConstructor
from numpy import array
from numpy.random import choice, shuffle
import os
import tqdm
from tensorflow.keras.utils import Sequence

#  make test for this generator ............................................. X


class Vidcg(Sequence):
    ''' Make a data generator to use with a tf.keras.model

    Make a frame sequence "Data generator" over a collection of images
    classified by categories. The structure must be equal:

       /root.
            /Data.
                 |
                 + - /Class_1
                 |
                 + - /Class_2.
                 |           |-/Vid_1.
                 |           |      |
                 |           |      |-image#1
                 |           |      |-image#2
                 |           |      |-...
                 |           |      |-image#j
                 |           |
                 |           |
                 |           |-/Vid_2.
                 |           |      |
                 |           |      |-image#1
                 |           |      |-image#2
                 |           |      |-...
                 |           |      |-image#n
                 |           .
                 |           .
                 |           .
                 |           |
                 |           |-/Vid_m.
                 +
                 .
                 .
                 .
                 + - /Class_n

    The path names, files names don care, just video folder container must be
    the class names, and images name must bee ordered by some name structure

    Parameters
    ----------
    data_path : str
        Path to "Data".

    sparse_categorical : bool
        True if you wnat a sparse categorical output like
        [N_class_0, ..., N_class_i]. As default generator return a one hot
        encoder output.

        : default False:

    frame_size : tuple
        set size of frames
        (batch, window, w, h, chanels)
                        ^--^
        : default (256, 256) :

    data_aug_pipe : fukg data_aug_pipe object
        Set a data augmentation pipe to make data augmentation based on image
        transformations

        :default None:

    strides : list
        set the strides for example and make augmentation
        if its posible
        : default [1, 2, 3, 4] :

    random_steps : list
        set the random steps taked for make random consecutive video frames
        : default <equal to steps_augmentation>:

    steps_augmentation_random : bool
        True or False, if you wnat a random steps data augmentation metoth
        : default False :

    color_mode : str
        either "rgb" or "gray"
        : default "rgb" :

    batch_size : int
        How many example that have every data batch;
        Data batch :(batch, frames, w, h, chanels)
                     ^---^
                       |
                       |-- batch_size
        : default 1 :

    window_size : int
        How many frame that have every example; (batch, frames, w, h, chanels)
                                                        ^----^
        : default 10 :

    to_train : bool
        set how is the output genrated, if 'True', generator return only X,
        other way, return 'X', 'Y'.
        : default True:
    '''
    def __init__(self,
                 data_path,
                 sparse_categorical=False,
                 frame_size=(256, 256),
                 random_steps=None,
                 data_aug_pipe=None,
                 strides=[1, 2, 3, 4],
                 steps_augmentation_random=False,
                 color_mode='rgb',
                 batch_size=1,
                 window_size=10,
                 to_train=True):
        self.sparse_categorical = sparse_categorical
        self.data_path = data_path
        self.frame_size = frame_size
        self.random_steps = random_steps
        self.strides = strides
        self.steps_augmentation_random = steps_augmentation_random
        self.color_mode = color_mode
        self.window_size = window_size
        self.batch_size = batch_size
        self.dirs = os.listdir(self.data_path)
        self.videos_all_paths = self.path_frames_full_paths()
        self.to_train = to_train
        self.categories_by_inds, self.categories_by_name = self.categories()
        self.examples = self.build_all_examples()
        self.batchs = self.batch_maker()
        self.one_hot_base = [0 for _ in self.categories_by_inds.keys()]
        self.data_aug_pipe = data_aug_pipe
        self.check()  # function that check if parameters are well select
        # Setting chanel option based on color mode
        if self.color_mode == 'rgb':
            self.chanels = 3
        if self.color_mode == 'gray':
            self.chanels = 1

    def check(self):

        '''
        This function check if some parameters are well select for the data set

        Check, if size of frames, batch are correct, and if the data path is
        well select.
        '''
        # - check if the windows size is higer than 1
        if self.window_size <= 1 or (self.window_size is None):
            raise ValueError('window_size must be higer than 1'
                             ' for a video classifier task')

        # - check if the data_path exist
        if not os.path.isdir(self.data_path):
            raise ValueError('the data_path that you provide not exist')

        # - get all video lengths, and check if window_size is small than
        #   minumun of video lengths (Check if the window_size is correct)
        videos_lengths = set()
        for video_absolute_path in self.videos_all_paths:
            cap = FramesConstructor(video_absolute_path)
            video_length_frames = cap.get_frames_count()
            videos_lengths.add(video_length_frames)

        if not (min(videos_lengths) >= self.window_size):
            message = 'The windows_size parameter, is grater than minimun of '\
                      'video lengths, pls, use a small window_size\n'
            message += '############ NOTE: #########################\n'
            message += '# The windows_size you provide are: '\
                       '{}\n'.format(self.window_size)
            message += '# The minimun lengths of video that you have are: '\
                       '{}\n'.format(min(videos_lengths))
            message += '#############################################'
            raise Exception(message)

        # check if strides*window_size <= min(video_lengths)
        bad_strides = []
        good_strides = []
        for step in self.strides:
            if not min(videos_lengths) >= self.window_size * step:
                bad_strides.append(step)
            else:
                good_strides.append(step)
        if len(bad_strides) > 0:
            print('################# fukg warning  #########################')
            print('The stride/s "{}" are/is to high, \nthe generator cant be '
                  'generate examples, that have a least {} frames'.format(
                                                        bad_strides,
                                                        self.window_size))
            print('_____________________________________')
            print('|***NOTE*** strides must be <= {}  '.format(
                                    min(videos_lengths)//self.window_size))
            print('|for this "windows_size"')
            print('|The minimun video frames length are {}'.format(
                                                    min(videos_lengths)))
            print('_____________________________________')
        if (len(good_strides) > 0) and (len(bad_strides) > 0):
            print('***Re building batches only for good stride/s {}***'.format(
                    good_strides))
            self.strides = good_strides
            self.examples = self.build_all_examples()
            self.batchs = self.batch_maker()
        if len(good_strides) < 1:
            raise Exception('You have no valid *stride* for you data set and '
                            '*window_size* that you provide. Consider that'
                            ' the minimun video length frames are {}'.format(
                                                min(videos_lengths)))
        # - check if if augmentation steps are correct
        for i in self.strides:
            if i <= 0:
                raise Exception('all steps_augmentation must be > 0, you put '
                                '{}'.format(i))
        empty = set()
        for i in self.strides:
            if i in empty:
                raise Exception('every steps in steps_augmentation must be '
                                'unique')
            empty.add(i)

        # if not to train this Datagenerator
        if self.to_train:
            self.on_epoch_end()

        # chek if batch_size >= 1
        if self.batch_size < 1:
            raise Exception('batch_size must be >= 1')

        # check color_mode
        if self.color_mode not in ['rgb', 'gray']:
            raise Exception('The color_mode parameter are incorrect must be'
                            ' "rgb" or "gray"')

        # define clasification output
        if self.sparse_categorical not in [True, False]:
            raise Exception('sparse_categorical vale must be True or False you'
                            ' put {} as value'.format(self.sparse_categorical))

        # set pipe format type for this generator
        if self.data_aug_pipe:
            if self.data_aug_pipe.structure['type'] != 'video':
                self.data_aug_pipe.structure['type'] = 'video'

    def categories(self):
        '''
        Build a dictionary that have all categories with number categories

        Returns
        -------
        NOTHING but!!

        Make class propieties like:

            self.categories_by_inds :
                {#1: biking,
                 #2:, diving,
                 ...,
                 #i, i_name}

            self.categories_by_name :
                {biking: #1,
                 diving: #2,
                 ...,
                 #i_name: #i}
        '''
        categories_by_inds = {}
        categories_by_name = {}
        for i, folder in enumerate(self.dirs):
            categories_by_inds[i] = folder
            categories_by_name[folder] = i

        return categories_by_inds, categories_by_name

    def path_frames_full_paths(self):
        '''
        make a list of all video system paths and check if a data_path is
        vaild

        returns:
        --------
        list type
        '''
        # - check if the path is empnty of categories
        for root, dirs, files in os.walk(self.data_path):
            if len(dirs) == 0:
                raise Exception('The data path is empty, please provide a '
                                'valid data path')
            break

        videos_full_path = []
        for folder in self.dirs:
            for video_name in os.listdir(os.path.join(self.data_path, folder)):
                video_absolute_path = os.path.join(self.data_path, folder,
                                                   video_name)
                videos_full_path.append(video_absolute_path)
        return videos_full_path

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
            print('not steps_augmentation_random technique are used')
            return
        if self.random_steps is None:
            print('automatically setting the random steps equal to'
                  '{}'.format(self.strides))
            self.random_steps = self.strides

        def set_random_steps(list, random_steps):
            '''
            sum number randomly to a list
            '''
            inds = range(len(list))
            for i in inds:
                old_val = list[i]
                new_val = old_val + choice(random_steps)
                list[i] = new_val
                if not i == inds[-1]:
                    list[i + 1] = new_val
                if not i == inds[0]:
                    if list[i] <= list[i - 1]:
                        list[i] = new_val + 1
            return list

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
                for i in frames_inds:
                    string_ind += str(i)
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

    def frames_loader(self, video_path, video_frames, chek=False):
        '''
        take a givenlist of fram of a video

        parameter
        ---------

        video_path : str
            video file path

        video_frames : list{ inds }
            list of int, that 'inds' are a indices in video frames from 0 to N

        return
        ------
            list of numpy arrays
        '''

        # simple check
        def simple_chek(for_chek, video_length, inds):
            if not for_chek:
                return True

            i_ = -1
            for i in inds:
                if i > i_:
                    i_ = i
                else:
                    raise Exception('Inds are bad positioned: {}'.format(inds))

            if inds[-1] <= video_length - 1:
                return True
            else:
                False

        cap = FramesConstructor(video_path)
        video_length = cap.get_frames_count()

        # make test
        if not simple_chek(for_chek=chek, video_length=video_length,
                           inds=video_frames):
            raise Exception('video indeces frames are incorrect: frames {} | '
                            'video total frames '
                            '{}'.format(video_frames, video_length))
        # end test
        natural_frames = cap.get_frames(video_frames)
        frames = []
        for frame in natural_frames:
            if frame is not None:
                if self.color_mode == 'rgb':
                    frame = cv2.resize(frame, self.frame_size)
                    frame = torgb(frame)
                if self.color_mode == 'gray':
                    frame = cv2.resize(frame, self.frame_size)
                    frame = togray(frame)
            else:
                raise Exception('cant load frame from {}'.format(video_path))
            frames.append(frame)

        if len(frames) == self.window_size:
            return frames
        else:
            raise Exception('Not posible get all frames... aborting')

    def generator_auto_test(self, use=False):
        '''
        Make a exaustive testing over the datagen

        This is for chek if our Datagenerator can be generate over all dataset
        before make a model train

        parameters
        ----------
        use : bool
            True if you realy want to check your generator
        '''
        if not use:
            print('This task may take a lot of time, do it with use=True')
            return
        print('Making an AUTO TEST, this may take a while... please wait')
        if self.color_mode == 'rgb':
            hope_shape = (self.window_size, self.frame_size[0],
                          self.frame_size[1], self.chanels)
        if self.color_mode == 'gray':
            hope_shape = (self.window_size, self.frame_size[0],
                          self.frame_size[1])
        for batch in tqdm.tqdm(self.batchs, desc='working on ... '):
            if self.to_train:
                X, Y = self.make_Xy_batch(batch)
            else:
                X = self.make_Xy_batch(batch)

            if X.shape[1:] != hope_shape:
                raise ValueError('generator cant proces well Shape provide'
                                 ' and, shape we hope are: {}, {}'.format(
                                                                X.shape[1:],
                                                                hope_shape)
                                 )
        print('_____________________________________________________________')
        print('Data generator auto test task, has been finished correctly :D')
        print('_____________________________________________________________')

    def batch_maker(self):
        '''
        Make the batches based on examples
        '''

        counter = {}
        for key in self.categories_by_name.keys():
            counter[key] = {'cuantitie': 0, 'example_ind': []}

        for i, example in enumerate(self.examples):
            category = example['category']
            counter[category]['cuantitie'] += 1
            counter[category]['example_ind'].append(i)

        batchs = []
        for category in counter.keys():
            while len(counter[category]['example_ind']) > 0:
                ex_inds = counter[category]['example_ind'][:self.batch_size]
                # NOTE: ex_inds are example
                # indices, NOT frames indices
                del counter[category]['example_ind'][:self.batch_size]

                batch = []
                for i in ex_inds:
                    batch.append(self.examples[i])
                batchs.append(batch)
        return batchs

    def augmenter(self, frames):
        '''
        Listen for image transformation functions
        '''
        if self.data_aug_pipe:
            frames = self.data_aug_pipe.apply(frames)
        return frames

    def plot_random_sample(self):
        '''
        Take a random batch and plot it
        '''
        import matplotlib.pyplot as plt

        if self.to_train:
            X, Y = self.__getitem__(choice(range(self.__len__())))
        else:
            X = self.__getitem__(choice(array(range(self.__len__()))))

        y = choice(range(self.batch_size))
        fig, axs = plt.subplots(nrows=self.window_size, ncols=1,
                                figsize=(10, self.window_size*10))
        for x, ax in zip(range(self.window_size), axs.flat):
            img = X[y, x]
            plt.tight_layout()
            ax.imshow(img)
        plt.show()

    def to_one_hot(self, i):
        '''Do from sparse to one hot'''

        one_hot = self.one_hot_base
        one_hot[i] = 1
        return one_hot

    def y_make(self, Y):

        '''make a one hot encoder clasification output
           or sparse categorical output

           one hot encoder output like --> [[0, 0, 1, 0]]

           sparse categorical output like -- > [3]; that means what the output
           must be [[0, 0, 0, 1]]
           '''

        if self.sparse_categorical:
            return Y
        else:
            one_hot_y = []
            for y in Y:
                one_hot_y.append(self.to_one_hot(y))
            return one_hot_y

    def make_Xy_batch(self, batch):
        '''
        Build data batch based on examples
        '''
        X = []
        Y = []
        for example in batch:
            category = example['category']  # name
            category = self.categories_by_name[category]  # numeric sparse
            video_path = example['path_to_video']
            video_inds_frames = example['video_inds']
            frames = self.frames_loader(video_path, video_inds_frames)
            frames = self.augmenter(frames)
            X.append(frames)
            Y.append(category)
        if self.to_train:
            Y = self.y_make(Y)
            return array(X), array(Y)
        else:
            return array(X)

    def __len__(self):
        return len(self.batchs)

    def on_epoch_end(self):
        if self.to_train:
            shuffle(self.batchs)

    def __getitem__(self, index):
        batch = self.batchs[index]
        if self.to_train:
            batch, category = self.make_Xy_batch(batch=batch)
            return batch, category
        else:
            batch = self.make_Xy_batch(batch=batch)
            return batch
