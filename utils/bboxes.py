import numpy as np
'''
this module has every function for generycal bbox operations

1. resize_bboxes
'''


def decode_bbox(bbox):
    '''
    given a bbox in fakg forma return their cordinates

    parameters:
    -----------
    bbox : dict
        bbox in fakg format

    return: xmin, ymin, xmax, ymax
    '''
    xmin, ymin = bbox['xmin'], bbox['ymin']
    xmax, ymax = bbox['xmax'], bbox['ymax']
    return xmin, ymin, xmax, ymax


def resize_boxes(bboxs: list, old_shape: tuple, new_shape: tuple,
                 x=0, y=0):
    '''
    reshape bboxes to adapt it to new image shape, and return only bboes
    with area non zero
    _____________
    | Parameters|
    +-----------+

    * bboxes -> bbox in fakg style
    * old_shape -> input shape
    * new_shape -> output shape
    * x, y, give it only if img was croped
    '''
    beta = new_shape[0] / old_shape[0]
    alpha = new_shape[1] / old_shape[1]
    bboxes = []
    for bbox in bboxs:
        xmin, ymin, xmax, ymax = decode_bbox(bbox)
        bboxes.append([xmin, ymin, xmax, ymax])

    bboxes = np.array(bboxes, dtype=np.int32)

    bboxes[:, 0] = np.int32(bboxes[:, 0]*alpha)
    bboxes[:, 1] = np.int32(bboxes[:, 1]*beta)
    bboxes[:, 2] = np.int32(bboxes[:, 2]*alpha)
    bboxes[:, 3] = np.int32(bboxes[:, 3]*beta)

    xmins = (bboxes[:, 0] - x).reshape((-1, 1))
    xmaxs = (bboxes[:, 2] - x).reshape((-1, 1))
    ymins = (bboxes[:, 1] - y).reshape((-1, 1))
    ymaxs = (bboxes[:, 3] - y).reshape((-1, 1))

    xmins[xmins < 0] = 0
    xmaxs[xmaxs < 0] = 0

    ymins[ymins < 0] = 0
    ymaxs[ymaxs < 0] = 0

    # check for hight range (remember that shape are (h, w))
    xmins[xmins > old_shape[1]] = old_shape[1]
    xmaxs[xmaxs > old_shape[1]] = old_shape[1]

    ymins[ymins > old_shape[0]] = old_shape[0]
    ymaxs[ymaxs > old_shape[0]] = old_shape[0]

    # # chek for zero area bboxs

    bboxes = np.hstack((xmins, ymins, xmaxs, ymaxs))
    areas = (xmaxs - xmins) * (ymaxs - ymins)
    non_area_cond = areas != 0
    non_area_cond = np.tile(non_area_cond, (1, 4))

    bboxes = bboxes[non_area_cond].reshape((-1, 4))

    # rebuild dicts
    bboxs = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        bbox = {}
        bbox['xmin'] = xmin
        bbox['ymin'] = ymin
        bbox['xmax'] = xmax
        bbox['ymax'] = ymax
        bboxs.append(bbox)
    if len(bboxs) == 0:
        return False
    return bboxs


def bboxs2points(bboxes):
    '''
    giveb boxes in fakg DataSet format:
    list of dictionaries
        [{xmin: val, ymin: val, xmax: val, ymax: val}, ...]
    return a arrat of of points with shape (all, 3)
    '''
    points = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = decode_bbox(bbox)
        bbox_points = [[xmin, ymin, 1],
                       [xmin, ymax, 1],
                       [xmax, ymin, 1],
                       [xmax, ymax, 1]]
        points.append(bbox_points)
    return points


def points2bboxs(points, transformed_shape=None):
    bboxs = []
    for bbox_points in points:
        bbox_points = np.array(bbox_points)
        bbox_points = np.reshape(bbox_points, (-1, 2))
        # fix boxes where are points outs of the image
        # remember thar img_shape is (h, w)

        xs = bbox_points[:, 0]
        ys = bbox_points[:, 1]

        # fix negatives values
        xs[xs < 0] = 0
        ys[ys < 0] = 0

        # fix out of shape values
        xs[xs > transformed_shape[1]] = transformed_shape[1]
        ys[ys > transformed_shape[0]] = transformed_shape[0]
        xmin, xmax = np.min(xs), np.max(xs)
        ymin, ymax = np.min(ys), np.max(ys)

        # check for cero area bbox
        if np.any([[xmax - xmin == 0], [ymax - ymin == 0]]):
            continue

        bbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}

        bboxs.append(bbox)

    if len(bboxs) > 0:
        return bboxs
    else:
        return False
