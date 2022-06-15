import ffmpeg
import os


def video2frames(path2video: str = None,
                 path2frames: str = None,
                 ext: str = 'png', fps=10,
                 *args, **kwargs):
    ''' conver a video2frames
    params:
    ======'''
    output_path = os.path.join(path2frames, '%04d.{}'.format(ext))
    ffmpeg.input('input.mp4') 
    (ffmpeg.input(path2video)
     .filter('fps', fps=fps)
     .output(output_path).
     run())


def vids2frames_clf(PATH: str, fps=10):
    ''' convert video dataset clasification to a
    frames clasification dataset

    NOTE: Needs ffmpeg-python and ffmpeg installed'''
    root = os.path.dirname(PATH)
    frames_base_path = PATH + '_frames'

    def mkdir(PATH):
        if os.path.isdir(PATH):
            return
        os.mkdir(PATH)
    mkdir(os.path.join(root, PATH + '_frames'))

    for clas in os.listdir(PATH):
        clas_path = os.path.join(PATH, clas)
        if not os.path.isdir(clas_path):
            continue
        base_clas_path = os.path.join(root, frames_base_path, clas)
        mkdir(base_clas_path)
        for video in os.listdir(clas_path):
            if not video.endswith('avi'):
                continue
            base_frames_path = os.path.join(base_clas_path, video.split('.')[0])
            mkdir(base_frames_path)
            video_path = os.path.join(clas_path, video)
            video2frames(video_path, base_frames_path, fps=fps)
