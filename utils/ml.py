import pandas as pd
import numpy as np


class OneHotEncoder:
    '''
    Build one hot encoder for object clasification or detection generators

    paramters:
    ----------

    classes : DataFrame
        a data frame with all classes in a DataSet
    '''

    def __init__(self, dataset: pd.DataFrame):
        self.classes = dataset['category']
        self.onehot = None
        self.build_one_hot()

    def build_one_hot(self):
        classes_int = {}
        classes_str = {}
        hot_encoder = np.zeros(len(self.classes.unique()))
        for i, classe in enumerate(self.classes.unique()):
            classes_int[int(i)] = classe
            c_onehot = hot_encoder.copy()
            c_onehot[i] = 1
            classes_str[str(classe)] = c_onehot

        self.classes_int = classes_int
        self.classes_str = classes_str

    def query(self, val: str):
        '''
        make a query to get onehot or label name

        return:
        ------

        * if val are int or onehot, return a label name as string
        * if val are string, return onehot
        '''
        if not ((isinstance(val, str)) or (isinstance(val, int)) or
           (isinstance(val, np.ndarray))):
            raise ValueError('You passed a invalid value, it must be and Int '
                             'or Str. You passes a {}'.format(type(val)))
        if isinstance(val, int):
            return self.classes_int[val]

        if isinstance(val, str):
            return self.classes_str[val]

        if isinstance(val, np.ndarray):
            # convert to logit
            val = int(np.argmax(val))
            return self.classes_int[val]
