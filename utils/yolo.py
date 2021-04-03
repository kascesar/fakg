import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame


def yolov2_anchors_study(df, n_clusters, input_shape=(416, 416), plot=False):
    '''
    Function that compute, and plot all anchors between 1 to n_clusters

    this function computed k-means algoritm basesd on yolo 9000's papers
    and compute de mean IoU anchors vs all boxs in the dataset

    parameters:
    -----------

    df : DataFrame
        this DataFrame are maked with the dataset_frame function's fakg

    n_clusters : int
        max cluster to search anchors

    input_shape : tuple
        this is because the images not necessarily have the same shape, so
        to make sure that all box are teatred in equal condition, convert all
        boxes that if it has a shape 416, 416 or aribiter given shapoe
    '''

    df['xmin'] = (input_shape[0] / df['im_width']) * df['xmin']
    df['xmax'] = (input_shape[0] / df['im_width']) * df['xmax']
    df['ymin'] = (input_shape[1] / df['im_height']) * df['ymin']
    df['ymax'] = (input_shape[1] / df['im_height']) * df['ymax']
    df[['xmin', 'ymin', 'xmax', 'ymax']] = df[['xmin', 'ymin',
                                               'xmax', 'ymax']].astype('int')
    boxs_w = df['xmax'] - df['xmin']
    boxs_h = df['ymax'] - df['ymin']
    x = boxs_w.to_numpy()
    y = boxs_h.to_numpy()
    data = np.reshape(np.array([x, y]), (-1, 2))
    # search kmean

    # compute mean IoU with the closet center
    mean_iou = []
    for n_cluster in range(1, n_clusters + 1):
        comulative_iou = []
        clusters = yolo_kmeans(data, n_cluster)
        for key in clusters.keys():
            k = clusters[key]['center']
            wh = clusters[key]['wh']
            iou = np.apply_along_axis(lambda box: center_iou(box, k),
                                      1, wh)
            comulative_iou.extend(iou)
        mean_iou.append(np.mean(comulative_iou))
    if plot:  # plot is that desired
        plt.figure(figsize=(10, 7))
        # plot graph mIoU v n clusters
        plt.scatter(range(1, n_clusters + 1), mean_iou)
        plt.xticks(range(1, n_clusters + 1))
        plt.grid(b=True, linestyle='--')
        plt.title('Mean IoU vs numbers of anchors', fontsize=16)
        # show plot
        plt.show()


def anchor_distance(wh: list, k: list):
    '''
    Defined the distance metric for kmeans anchros clustering
    in Yolo2 anchors k-means anchors selection
    '''
    return 1 - center_iou(wh, k)


def yolo_kmeans(x, n_clusters):
    '''
    that is my own k-means for yolo implementation, that need more speed
    execution but, i work on it in the near future

    Thas function is neede that yolo kmeans anchors need a diferent metric for
    compute the distances when clustering:

    d = 1 - iou(cemter, box)

    instead euclidean distance, that is commonly used in the major of
    algorithms

    d = sqrt( sum (x_i ** 2) )
    '''
    ############################
    # Randomly choose clusters #
    ############################
    centers_idx = set()                         # thi may mek sure that pick
    while len(centers_idx) < n_clusters:        # uniques centers
        idx = np.random.choice(range(len(x)))
        if idx not in centers_idx:
            centers_idx.add(idx)

    centers = []
    for idx in centers_idx:
        centers.append(x[idx])

    clusters = {}
    for i, c in enumerate(centers):
        clusters[i] = {}
        clusters[i]['center'] = c  # asiciated clusters
        clusters[i]['wh'] = []     # asociated data

    ####################
    # start clustering #
    ####################
    while True:

        # copy old_centers
        old_centers = []
        for key in clusters.keys():
            old_centers.append(clusters[key]['center'])

        # clean all data
        for key in clusters:
            clusters[key]['wh'] = []

        # Assign labels based on closest center
        for wh in x:
            distances = []
            for key in clusters.keys():
                cluster = clusters[key]['center']
                d = anchor_distance(wh, cluster)
                distances.append([d, wh, key])
            distances = np.array(distances)
            idx = np.argmin(distances[..., 0])
            best_distance, wh, key = distances[idx]
            clusters[key]['wh'].append(wh)

        #########################################
        # Find new centers from means of points #
        #########################################
        new_centers = []
        for key in clusters.keys():
            wh = clusters[key]['wh']
            wh = np.reshape(wh, (-1, 2))
            new_center = np.mean(wh, axis=0)
            clusters[key]['center'] = new_center
            new_centers.append(new_center)

        new_centers = np.array(new_centers)
        old_centers = np.array(old_centers)

        ##################################
        # check for terminate clustering #
        ##################################
        if np.array_equal(new_centers, old_centers):
            for key in clusters.keys():  # fix data structure
                wh = clusters[key]['wh']
                clusters[key]['wh'] = np.reshape(wh, (-1, 2))
            break

    return clusters


def center_iou(wh1, wh2):
    '''
    compute iou asumet that x1, y2 of each bbox are the same
    '''

    w1, h1 = int(wh1[0]), int(wh1[1])
    w2, h2 = int(wh2[0]), int(wh2[1])

    # inter
    w = np.minimum(w1, w2)
    h = np.minimum(h1, h2)
    intersection = w * h

    # union
    a1 = w1 * h1
    a2 = w2 * h2
    union = a1 + a2 - intersection
    iou = intersection / union
    return iou


def yolov2_batch_gen(batch, anchors):
    pass


def yolov2_groundtruth(img_path, anchors, bbox, one_hot, im_w, im_h):
    pass


def bbox2yolo(bbox):
    '''
    given a decoded and reescaled bbox from anotation, convert it to yolo
    format

    parameters:
    -----------

    bbox : dict
        bbox fakg bbox style

    return :
        converted bbox
    '''
    xmin, ymin, xmax, ymax = bbox
    xcenter = (xmin + xmax)/2  # mean of distance
    ycenter = (ymin + ymax)/2  # mean of distance
    w, h = xmax - xmin, ymax - ymin
    return xcenter, ycenter, w, h


class YoloV2Anchors:
    '''
    class for search anchors based on k-means

    this class computed k-means algoritm basesd on yolo 9000's papers
    and compute de mean IoU anchors vs all boxs in the dataset

    parameters:
    -----------

    dataset : DataFrame
        this DataFrame are maked with the DataSet class's fakg

    n_clusters : int
        max cluster to search anchors

    input_shape : tuple
        this is because the images not necessarily have the same shape, so
        to make sure that all box are teatred in equal condition, convert all
        boxes that if it has a shape 416, 416 or aribiter given shapoe
    '''
    def __init__(self, dataset: DataFrame, input_shape: tuple):
        self.dataset = dataset
        self.input_shape = input_shape
        self.data = self.make_data()

    def make_data(self):
        self.dataset['xmin'] = ((self.input_shape[0]/self.dataset['im_width'])
                                * self.dataset['xmin'])
        self.dataset['xmax'] = ((self.input_shape[0]/self.dataset['im_width'])
                                * self.dataset['xmax'])
        self.dataset['ymin'] = ((self.input_shape[1]/self.dataset['im_height'])
                                * self.dataset['ymin'])
        self.dataset['ymax'] = ((self.input_shape[1]/self.dataset['im_height'])
                                * self.dataset['ymax'])
        self.dataset[['xmin', 'ymin', 'xmax', 'ymax']] = self.dataset[
                                                    ['xmin', 'ymin',
                                                     'xmax', 'ymax']
                                                     ].astype('int')
        boxs_w = self.dataset['xmax'] - self.dataset['xmin']
        boxs_h = self.dataset['ymax'] - self.dataset['ymin']
        x = boxs_w.to_numpy()
        y = boxs_h.to_numpy()
        return np.reshape(np.array([x, y]), (-1, 2))

    def build_anchors(self, n_anhcors):
        '''
        build n anchors based on k-means with yolo algorithm
        (distance d = 1 - iou)
        '''
        kmeans = yolo_kmeans(self.data, n_anhcors)
        anchors = []
        for key in kmeans.keys():
            anchor = kmeans[key]['center']
            anchors.extend(anchor)
        return anchors

    def anchors_study(self, max_clusters=11, use=False):
        if not use:
            print('This may take a lot of time, use it with param use=True if '
                  'you wnat to do it at all')
        else:
            yolov2_anchors_study(df=self.dataset, n_clusters=max_clusters,
                                 input_shape=self.input_shape, plot=True)
