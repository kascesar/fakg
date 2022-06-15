import cv2
from numpy import ndarray


def imgtest(img):
    '''Make sure if the argument passes are correct'''

    if not isinstance(img, ndarray):
        raise Exception('The argument passes is not a numpy.ndarray object')
    if (img.shape[0] <= 1) or (img.shape[1] <= 1):
        raise Exception('There is no valid argument, the Image passes has {}'
                        ' shape'.format(img.shape))


def torgb(img):
    ''' Convert gray images into rgb, convert brg images into rgb
    convert rgb img into rgb
    '''
    imgtest(img)

    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def togray(img):
    ''' convert bgr images into gray images and rgb images into gray.
    '''
    imgtest(img)

    if len(img.shape) == 2:
        return img
    elif len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)


def draw_object(img: ndarray, bbox: dict, object: str, color=(255, 0, 255)):
    '''
    '''
    xmin, ymin = bbox['xmin'], bbox['ymin']
    xmax, ymax = bbox['xmax'], bbox['ymax']
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

    font_scale = 1
    font = cv2.FONT_HERSHEY_TRIPLEX
    # set the rectangle background to white
    rectangle_bgr = color
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(object, font,
                                                fontScale=font_scale,
                                                thickness=1)[0]
    text_offset_x = xmin
    text_offset_y = ymin if not ymin < 20 else ymax - 4
    box_coords = ((text_offset_x, text_offset_y),
                  (text_offset_x + text_width + 2,
                   text_offset_y - text_height - 2))

    cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr,
                  cv2.FILLED)
    cv2.putText(img, object, (text_offset_x, text_offset_y), font,
                fontScale=font_scale, color=(255, 255, 255),
                thickness=1)

    return img


def draw_points(img, points, color=(0, 255, 0)):
    for point in points:
        cv2.circle(img, tuple(point), 5, color, -1)
    return img


def plot_example(dataset=None):

    # randomnly choice a example
    from numpy.random import choice

    rand_idx = choice(range(len(dataset)))
    example = dataset[rand_idx]
    type = example['information']['type']
    # plot example based in it type

    if type == 'object detection':
        img = example['information']['frame_path']
        img = cv2.imread(img)
        imgtest(img)
        for key in example['anotations'].keys():
            bbox = example['anotations'][key]['bbox']
            object = example['anotations'][key]['object']
            img = draw_object(img, bbox, object, (255, 0, 255))

        cv2.imshow('fakg plot for object detection', img)
        k = cv2.waitKey(0)
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()


def load_img(img_path, new_shape=None):
    img = cv2.imread(img_path)
    imgtest(img)
    if not new_shape:
        return img
    if new_shape:
        return cv2.resize(img, dsize=new_shape)


def show_img(img):
    cv2.imshow('img', img)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

        
def reshape_img(img: ndarray, new_shape=None):
    if not new_shape:
        return img
    imgtest(img)
    if new_shape:
        return cv2.resize(img, dsize=new_shape)