'''
PIPE

The pipe module has contains pipe augmenters to build a tree, that
helps to build efient pipe lines. The pipes augmenters use multiprocesing,
for do eficients dataaugmentation task.
'''

from multiprocessing import Pool, cpu_count
from cv2 import imread
from abc import abstractmethod, ABC
from numpy.random import choice, random


class Pipe(ABC):
    def __init__(self):
        self.n_cpu = cpu_count()

    def __call__(self, batch_example):
        with Pool(self.n_cpu) as p:
            return list(p.map(self.apply, batch_example))

    @abstractmethod
    def apply(self, example):
        pass

    @abstractmethod
    def add(self, layers):
        pass


class ImagePipe(Pipe):
    def __init__(self, **kwrd):
        self.structure = {}
        self.structure['suported layers'] = ['image']
        self.structure['flow map'] = []
        self.structure['pipe'] = []
        # self.structure['type'] = type
        super().__init__(**kwrd)

    def apply(self, example):
        impath = example['information']['frame_path']

        img = imread(impath)
        bboxs = []
        objects = []
        for key in example['anotations'].keys():
            bbox = example['anotations'][key]['bbox']
            obj = example['anotations'][key]['category']
            objects.append(obj)
            bboxs.append(bbox)

        img, bboxs = self.flow(img, bboxs)
        # the apply augmenter tree
        if bboxs:
            return (img, objects, bboxs)
        else:
            return (img, objects)

    def flow(self, img, bboxs=None):
        for _ in range(len(self.structure['pipe'])):
            layer = choice(self.structure['pipe'])
            if random() <= layer.frec:
                img, bboxs = layer(img=img, bboxs=bboxs)

        if bboxs:
            return img, bboxs
        else:
            img

    def add_layer(self, layer):
        try:
            if layer.type in self.structure['suported layers']:
                self.structure['pipe'].append(layer)
                self.structure['flow map'].append(layer.name)
            else:
                raise Exception(
                                "This layer can't be added, the pipe "
                                "type is 'ImagePipe' and you put a "
                                "layer {} type".format(str(layer.type))
                                )
        except LayerError as err:
            print('Object {}, is not a valid object '
                  'type for fakg Pipe'.format(err))

    def add(self, layers):
        '''
        Add a (or list) of fakgaugmenters
         ___________
        | Parameters|
        +-----------+

        layers : list of tuples or single tuple.
            Each tuple mustbe:
                tuple: (fakg augmenter, unique identifier: str )
        '''
        # iterate over list layers
        if isinstance(layers, list):
            for layer in layers:
                self.add_layer(layer)

        else:
            self.add_layer(layers)
        # add single layer
        # elif isinstance(layers, tuple):

        # try:
        #     if layer.type in self.structure['suported layers']:
        #         self.structure['pipe'].append(layer)
        #         self.structure['flow map'].append(layer.name)
        #     else:
        #         raise Exception(
        #                         "This layer can't be added, the pipe "
        #                         "type is 'ImagePipe' and you put a "
        #                         "layer {} type".format(str(layer.type))
        #                         )
        # except LayerError as err:
        #     print('Object {}, is not a valid object '
        #           'type for fakg Pipe'.format(err))

        # def branchs(self, branch: (list or str)):
        #     '''
        #
        #     Branch are map where the input data must be flow.
        #     Each passed branch must be in the form:
        #
        #         str: "[unique identifier(UI)]#1[OP]UI2[OP]UI#3"
        #
        #         where [OP] must be "&", "-&", "&-" (and, not and, and not)
        #         example # 1
        #         -----------
        #
        #         "A&B&C"
        #
        #         it means that, the Pipe will try to do the "A" augmeter
        #         firs, then it will be try with B, and, then try with "C".
        #
        #
        #         example # 2
        #         -----------
        #
        #         branch = 'A' + '-&' + 'B' + '&-' + 'C'
        #
        #         means that if "A" is applied, not apply "B", and if "B" is
        #         applied not applied "C"
        #     '''
        #     if isinstance(branch, list):
        #         for b in branch:
        #             if isinstance(b, str):
        #                 self.structure['branchs'].append(branch)
        #             else:
        #                 raise ValueError('you must pass list of strings')
        #     else:
        #         self.structure['branchs'].append(branch)
        #     # TODO:
        #     # make the paths and build the function that plot these paths
        #     # to get a visual map flow
        #     tree = []
        #     for branch in self.structure['branchs']:
        #         layers = branch.split('&')
        #         tree.append(list(layers))
        #
        #     layers_and_ids = {}
        #     for layer in self.structure['pipe']:
        #         augmenter, layer = layer
        #         layers_and_ids[layer] = augmeter
        #
        #     for t_branch in reversed(tree):
        #         branch = []
        #         for id in reversed(t_branch):
        #             augmeter = layers_and_ids[id]()
        #             branch.append(augmeter)


class LayerError(Exception):

    # Constructor or Initializer
    def __init__(self, input):
        self.input = type(input)

    # __str__ is to print() the value
    def __str__(self):
        return(repr(self.input))
