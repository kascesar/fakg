from numpy import ndarray, random


class ImagePipe:

    '''
    Pipe builder class

    This constructor receives augmentators to  apply in cascade
     ___________
    | Parameters|
    +-----------+
    type:
        this param is configfureb by generator, because is more natural
        type -> None, may be 'image' or 'video'
    '''

    def __init__(self, type=None):
        self.structure = {}
        self.structure['type'] = type
        self.structure['suported layers'] = ['image']
        self.structure['flow map'] = []
        self.structure['pipe'] = []

    def add(self, layers):
        def do(layers):
            for layer in layers:
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
        if isinstance(layers, list):
            do(layers)
        else:
            try:
                do([layers])
            except ValueError as err:
                print(err)
                print('Augmeter must be passed into a list or single')

    def remove(self, index=0):
        '''
        Remove a layer added before
        _____________
        | Parameters|
        +-----------+

        index -> int, zero based index layers
            as defect 0
        '''
        try:
            del self.structure['pipe'][index]
            del self.structure['flow map'][index]

        except ValueError as err:
            print(err)
            print('Index that you provide is out of index list.\n'
                  'The len of layeres in the pipe are {}, and you passed \n'
                  '{} as index'.format(len(self.structure['pipe']), index))

    def print_workflow(self):
        '''
        show the work flow map
        '''
        str_len = []
        for STR in self.structure['flow map']:
            str_len.append(len(STR))
        pad = ''
        for _ in range(max(str_len)//2):
            pad += ' '

        map = ''

        input_pad = 'Index '
        for _ in range(abs((len('input data') - max(str_len))//2) - 6):
            input_pad += ' '
        map += input_pad + 'Input Data' + '\n'
        map += pad + '|' + pad + '\n'
        map += pad + '|' + pad + '\n'
        map += pad + '*' + pad + '\n'
        for i, STR in enumerate(self.structure['flow map']):
            layer_str = ''
            internal_pad = str(i) + ':'
            for _ in range(abs((len(STR) - max(str_len))//2) - 2):
                internal_pad += '.'
            layer_str += internal_pad
            layer_str += STR
            layer_str += '\n'
            layer_str += pad + '|' + pad + '\n'
            layer_str += pad + '|' + pad + '\n'
            layer_str += pad + '*' + pad + '\n'
            map += layer_str

        output_pad = ''
        for _ in range(abs((len('output data') - max(str_len))//2)):
            output_pad += ' '
        map += output_pad + 'Output Data' + output_pad + '\n'
        return print(map)

    def test(self, data):
        '''
        make a data test before pasing into fakg augmenter layers
        '''
        def is_img(x):
            if isinstance(data, ndarray):
                if (data.shape[0] >= 2) and (data.shape[1] >= 2):
                    True
        if is_img:
            pass
        elif isinstance(data, list):
            for sub_data in data:
                if not is_img(sub_data):
                    raise ValueError('The list passed in this augmenter has an'
                                     ' invalid value: {}'.format(type(sub_data)
                                                                 )
                                     )
        else:
            raise ValueError('You pass a invalid value for this augmenter'
                             '. value are {} and must be "ndarray" or '
                             "list" 'that has "ndarrays" on it'.format(type(
                                                                        data)
                                                                       )
                             )

    def apply(self, data):
        x = data
        self.test(data)
        for layer in self.structure['pipe']:
            if random.rand() <= float(layer.frec):
                if self.structure['type'] == 'video':
                    y = []
                    for img in x:
                        img = layer.trans(img)
                        y.append(img)
                elif self.structure['type'] == 'image':
                    y = self.trans(x)
                else:
                    raise Exception('The pipe object is "{}", and this Pipe '
                                    'are  invalid for this task please, '
                                    'provide a valid fukg Pipe for image '
                                    'augmentation.\n The valid Pipe for this '
                                    'task are "ImagePipe"'
                                    ''.format(self.structure['type']))
                layer.reset_state()
                x = y
            else:
                continue
        return x


class LayerError(Exception):

    # Constructor or Initializer
    def __init__(self, input):
        self.input = type(input)

    # __str__ is to print() the value
    def __str__(self):
        return(repr(self.input))
