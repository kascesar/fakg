import os
import cv2
import numpy as np
from magic import from_file
import re


class FramesConstructor:
    '''
    Image loader based on frames img files

    The structure of file must be:

    |ath_to_-Vid_1.
                  |
                  |-image_name#1
                  |-image_name#2
                  |-...
                  |-image_name#j

    where all frames must be consecutive

    parameters:
    -----------

    path2frames : None as default, path is where are all frames from a same
                  video
    '''
    def __init__(self, path2frames=None):
        self.path2frames = path2frames
        self.test()
        self.frame_list = sorted(os.listdir(self.path2frames))

    def test(self):
        # test if the past exist
        if not os.path.exists(self.path2frames):
            raise ValueError("The path '{}' that yoiu provide\ndon't exist, "
                             "please, provide avalid path"
                             "".format(self.path2frames))

    def get_frames_count(self):
        '''
        return the amount of frames that path contains
        '''
        return len(os.listdir(self.path2frames))

    def get_inds(self):
        '''
        get all inddexes from every frame
        '''
        return np.arange(0, self.get_frames_count(), 1)

    def get_frames(self, inds):
        '''
        Return a frame or set of frame based on inds

        parameters:
        -----------

        inds : integer or list of integer
           '''
        if isinstance(inds, int):
            frame_path = os.path.join(self.path2frames, self.frame_list[inds])
            return cv2.imread(frame_path)
        elif isinstance(inds, list):
            frames = []
            for i in inds:
                frame_path = os.path.join(self.path2frames, self.frame_list[i])
                frames.append(cv2.imread(frame_path))
        else:
            raise ValueError("You passeda '{}' onject as inds.\n"
                             "You must provide a integer or list of integers\n"
                             "as parameter".format(type(inds)))
        return frames


def only_files(path: str):
    '''
    return only files from a directory

    parameters: path to folder

    return: list of string (no absolute paths)
    '''
    folder_list = os.listdir(path)
    only_files = []
    for file in folder_list:
        path2file = os.path.join(path, file)
        if os.path.isfile(path2file):
            only_files.append(file)
    return only_files


def only_imgs(path: str):
    files = only_files(path)
    ext = ['jpeg', 'JPEG', 'jpg', 'JPG', 'png', 'PNG', 'tif', 'TIF']
    only_x = []
    for f in files:
        for e in ext:
            if e in f:
                only_x.append(f)
                break
    return only_x


def only_json(path: str):
    files = only_files(path)
    ext = ['json', 'JSON']
    only_x = []
    for f in files:
        for e in ext:
            if e in f:
                only_x.append(f)
                break
    return only_x


def only_txt(path: str):
    files = only_files(path)
    ext = ['txt', 'TXT']
    only_x = []
    for f in files:
        for e in ext:
            if e in f:
                only_x.append(f)
                break
    return only_x


def only_xml(path: str):
    '''
    return only json files in the target path
    '''
    files = only_files(path)
    ext = ['xml', 'XMLS']
    only_x = []
    for f in files:
        for e in ext:
            if e in f:
                only_x.append(f)
                break
    return only_x


def make_complete_paths(path, list_files_and_dirs):
    '''
    make all complete paths from a list of dirs and files that are in the
    same dir

    param:
    ------
    list_files_and_dirs: list of strings
        list that have all files an dirs of the same dir

    path: string
        strig that are the path of those files and dirs
    '''
    complete_paths = []
    for f in list_files_and_dirs:
        complete = os.path.join(path, f)
        complete_paths.append(complete)
    return complete_paths


def get_im_wh(img_path: str):
    '''
    Get image w,h with out charge it
    '''
    inf = from_file(img_path)
    w, h = list(map(int, re.findall(r'(\d+)x(\d+)', inf)[-1]))
    return w, h
