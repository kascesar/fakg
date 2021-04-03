import cv2
import numpy as np
from abc import ABC, abstractmethod
from fakg.utils.bboxes import (resize_boxes, bboxs2points, points2bboxs,
                               decode_bbox)


def bboxs_check(func, min_area=0.3, max_trys=100):
    '''
    this decorator is to check the bboxs are correct after transformed
    if it not, it try again
    '''
    def wraper(*args, **kargs):
        bboxs = kargs['bboxs']
        self, img = args
        # check if for img only
        if not bboxs:
            return func(self, img, bboxs=bboxs)

        trys = 0
        while True:
            trys += 1
            area_val = True
            len_val = True
            img_, bboxs_ = func(self, img, bboxs=bboxs)

            # check if the transformation return no bboxs
            if not bboxs_:
                continue
            for bbox_, bbox in zip(bboxs_, bboxs):
                xmin, ymin, xmax, ymax = decode_bbox(bbox)
                xmin_, ymin_, xmax_, ymax_ = decode_bbox(bbox_)
                w, h = xmax - xmin, ymax - ymin
                w_, h_ = xmax_ - xmin_, ymax_ - ymin_

                # check if the area are >= 0.3
                if (w_ * h_)/(w*h) <= min_area:
                    area_val = False

            # check if de n bbox returned are equal
            if len(bboxs) != len(bboxs_):
                len_val = False

            if (area_val) and (len_val):
                return img_, bboxs_

            if trys == max_trys:
                return img, bboxs

    return wraper


class ImageLayer(ABC):
    '''
    Base layer for video augmenter data in FAKG
    _____________
    | Parameters|
    +-----------+

    * frec -> float between[0, 1]: with what frec the augmenter is apply this
                                   transformation

    * type -> type fot this augmenter (can be apply in videos or images)
    ___________________________________________________________________________
    ________________
    | Esential def |
    +--------------+

    * reset_state : reset all random state in the augmenter

    * apply       : Apply the transformation in all video frames that it recibe

    * names       : Must give, this is for plot workflow in fakg pipe objetcs
    ___________________________________________________________________________
    '''
    def __init__(self, frec=0.5):
        self.frec = frec
        self.type = 'image'
        self.sub_type
        self.name
        self.border_options = (cv2.WARP_INVERSE_MAP,
                               cv2.BORDER_WRAP,
                               cv2.BORDER_REPLICATE)

    def __call__(self, img, bboxs=None):
        return self.trans(img, bboxs=bboxs)

    @abstractmethod
    def trans(self, img, bboxes=None):
        pass

    def reset_state(self):
        pass

    def color_type(self, img):
        if len(img.shape) == 3:
            return 'rgb'
        else:
            return 'gray'

    def to_hsv(self, img):
        if self.color_type(img) == 'rgb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        else:
            return img

    def hsv2rbg(self, img):
        if len(img.shape) >= 3:
            return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        else:
            return img

    def trans_matrix_adjust(self, M, w, h):
        if self.sub_type == 'spatial':
            cx = w//2
            cy = h//2
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            M[0, 2] += (nW / 2) - cx
            M[1, 2] += (nH / 2) - cy
            return M, nW, nH

    def mt_bbox_points(self, transformation_matrix, bbox_points):
        '''
        Apply transformation matrix (mt)to a points bbox
        _____________
        | Parameters|
        +-----------+

        * mt -> matrix transformation
        * bboxes -> bbox in fakg style
        '''

        points = np.apply_along_axis(lambda point: np.matmul(
                                    transformation_matrix, np.array(point).T),
                                                    1, bbox_points)
        points = [point[:2] for point in points]
        points = np.array(points, dtype=np.int)
        return points

    def trans_bboxs(self, mt, bboxs, img_shape):
        '''
        apply matrix transformation to all bboxes
        _____________
        | Parameters|
        +-----------+

        * mt -> matrix transformation
        * bboxes -> list of bboxes dict
        '''
        if self.sub_type == 'spatial':
            bboxs_points = bboxs2points(bboxs)
            out = []
            for bbox_points in bboxs_points:
                points = self.mt_bbox_points(mt, bbox_points)
                out.append(points)
            return points2bboxs(out, transformed_shape=img_shape)
        else:
            return bboxs

    def resize_img_and_boxs(self, img, bboxs, old_shape, new_shape):
        img = cv2.resize(img, (new_shape[1], new_shape[0]))
        bboxs = resize_boxes(bboxs, old_shape=old_shape, new_shape=new_shape)
        return img, bboxs


class Vflip(ImageLayer):
    '''
    Make a vertical flip to a video frames
    _____________
    | Parameters|
    +-----------+
    Nothing, that layer apply Horizontal flip as random
    '''
    def __init__(self, **kwrd):
        self.tm = np.float32([[1, 0, 0], [0, -1, -1], [0, 0, 1]])
        self.name = 'Vertical flip'
        self.sub_type = 'spatial'
        super().__init__(**kwrd)

    def trans(self, img, bboxs=None):
        self.h, self.w = img.shape[:2]
        self.tm[1, 2] += self.h

        if bboxs:
            img = cv2.warpPerspective(img, self.tm, (self.w, self.h),
                                      *self.border_options
                                      )
            bboxes = self.trans_bboxs(self.tm, bboxs, img.shape[:2])
            if bboxes:
                img, bboxes = self.resize_img_and_boxs(img, bboxes,
                                                       img.shape[:2],
                                                       (self.h, self.w),
                                                       )
                return img, bboxes
            else:
                return img, False
        else:
            return cv2.warpPerspective(img, self.tm, (self.w, self.h),
                                       *self.border_options
                                       )


class Hflip(ImageLayer):
    '''
    Make a vertical flip to a video frames
    _____________
    | Parameters|
    +-----------+

    Nothing, thar layer apply Vertical flip as random
    '''
    def __init__(self, **kwrd):
        self.tm = np.float32([[-1, 0, -1], [0, 1, 0], [0, 0, 1]])
        self.sub_type = 'spatial'
        self.name = 'Horizontal flip'
        super().__init__(**kwrd)

    def trans(self, img, bboxs=None):
        if self.tm[0, 2] == -1:
            shape = img.shape
            self.h = shape[0]
            self.w = shape[1]
            self.tm[0, 2] += self.w

        if bboxs:
            img = cv2.warpPerspective(img, self.tm, (self.w, self.h),
                                      *self.border_options
                                      )
            bboxes = self.trans_bboxs(self.tm, bboxs, img.shape[:2])
            if bboxes:
                # return bboxs only if the are bboxs inside the transformed
                # image
                img, bboxes = self.resize_img_and_boxs(img, bboxes,
                                                       img.shape[:2],
                                                       (self.h, self.w),
                                                       )
                return img, bboxes
            else:
                return img, False
        else:
            return cv2.warpPerspective(img, self.tm, (self.w, self.h),
                                       *self.border_options
                                       )


class RandomRotation(ImageLayer):
    '''
    randomly rotate an image between -angle, angle
    _____________
    | Parameters|
    +-----------+

    * angle -> int: angle of rotation.
    '''
    def __init__(self, angle=30, **kwrd):
        self.sub_type = 'spatial'
        self.val = angle
        self.angle = np.random.randint(-self.val, self.val)
        self.h = None
        self.w = None
        self.tm = None
        self.name = 'Random rotation by +- {} angle'.format(abs(self.angle))
        super().__init__(**kwrd)

    def reset_state(self):
        self.angle = np.random.randint(-self.val, self.val)
        self.name = 'Random rotation by +- {} angle'.format(abs(self.angle))

        self.tm = cv2.getRotationMatrix2D((self.w//2, self.h//2),
                                          self.angle, 1.0)

    @bboxs_check
    def trans(self, img, bboxs=None):
        if self.tm is None:
            self.h, self.w = img.shape[:2]
            self.tm = cv2.getRotationMatrix2D((self.w//2, self.h//2),
                                              self.angle, 1.0)

        if bboxs:
            img = cv2.warpAffine(img, self.tm, (self.w, self.h),
                                 *self.border_options
                                 )
            bboxes = self.trans_bboxs(self.tm, bboxs, img.shape[:2])
            if bboxes:

                return img, bboxes
            else:
                return img, False
        else:
            return cv2.warpAffine(img, self.tm, (self.w, self.h),
                                  *self.border_options
                                  )


class RandomCrop(ImageLayer):
    '''
    Randomly crop an image
    _____________
    | Parameters|
    +-----------+

    * porcentage -> float [0, 1[ : amount of "zoom"
    '''
    def __init__(self, magnitude=0.2, **kwrd):
        self.sub_type = 'spatial'
        self.porcentage = magnitude
        self.first_run = True
        self.w = None
        self.h = None
        self.name = 'Random Crop by {} porcentage'.format(self.porcentage)
        super().__init__(**kwrd)

    def reset_state(self):
        pass

    @bboxs_check
    def trans(self, img, bboxs=None):
        self.h, self.w = img.shape[:2]
        old_shape = img.shape[:2]
        if self.first_run:
            self.x = np.random.randint(0, int(self.w * self.porcentage))
            self.y = np.random.randint(0, int(self.h * self.porcentage))
            self.first_run = False

        new_shape = (int(self.w * self.porcentage + self.w),
                     int(self.h * self.porcentage + self.h))

        resized_img = cv2.resize(img, new_shape)
        new_shape = resized_img.shape[:2]

        x_range = (self.x, self.x + self.w)
        y_range = (self.y, self.y + self.h)
        img = resized_img[y_range[0]:y_range[1], x_range[0]:x_range[1]]

        if bboxs:
            bboxs = resize_boxes(bboxs,
                                 old_shape=old_shape,
                                 new_shape=new_shape,
                                 x=self.x,
                                 y=self.y)
            if bboxs:
                return img, bboxs
            else:
                return img, False
        else:
            return img


class RandomShift(ImageLayer):
    '''
    randomly shift an image moving it to x, y randon direction

    _____________
    | Parameters|
    +-----------+

    * porcentage -> float between[0, 1]: porcentage in relation with img shape
    '''
    def __init__(self, magnitude=0.2, **kwrd):
        self.sub_type = 'spatial'
        self.porcentage = magnitude
        self.first_run = True
        self.w = None
        self.h = None
        self.tm = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.name = 'Random shift by {} porcentage'.format(self.porcentage)
        super().__init__(**kwrd)

    def reset_state(self):
        self.dx = np.random.randint(-self.x_range, self.x_range)
        self.dy = np.random.randint(-self.y_range, self.y_range)
        self.tm[0, 2] = self.dx
        self.tm[1, 2] = self.dy

    @bboxs_check
    def trans(self, img, bboxs=None):
        self.h, self.w = img.shape[:2]
        if self.first_run:
            self.x_range = int(self.w * self.porcentage)
            self.y_range = int(self.h * self.porcentage)
            self.dx = np.random.randint(-self.x_range, self.x_range)
            self.dy = np.random.randint(-self.y_range, self.y_range)

        self.tm[0, 2] = self.dx
        self.tm[1, 2] = self.dy

        if bboxs:
            img = cv2.warpPerspective(img, self.tm, (self.w, self.h),
                                      *self.border_options
                                      )
            bboxes = self.trans_bboxs(self.tm, bboxs, img.shape[:2])
            if bboxes:
                img, bboxes = self.resize_img_and_boxs(img, bboxes,
                                                       img.shape[:2],
                                                       (self.h, self.w))
                return img, bboxes
            else:
                return img, False
        else:
            return cv2.warpPerspective(img, self.tm, (self.w, self.h),
                                       *self.border_options
                                       )


class RandomShear(ImageLayer):
    '''
    Randomly shear frames

    this effect is the same for all frames
    _____________
    | Parameters|
    +-----------+
    * magnitude -> float: "intensity" shear
    '''
    def __init__(self, magnitude=0.3, **kwrd):
        self.sub_type = 'spatial'
        self.magnitude = magnitude
        self.dxdy = np.random.uniform(-self.magnitude, self.magnitude)
        self.tm = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.name = 'Random Shear img by {} magnitude'.format(self.magnitude)
        self.h = None
        self.w = None
        super().__init__(**kwrd)

    def reset_state(self):
        self.dxdy = np.random.uniform(-self.magnitude, self.magnitude)
        self.tm[0, 1] = self.dxdy
        self.tm[1, 0] = self.dxdy

    @bboxs_check
    def trans(self, img, bboxs=None):

        self.h, self.w = img.shape[:2]
        self.tm[0, 1] = self.dxdy
        self.tm[1, 0] = self.dxdy

        if bboxs:
            img = cv2.warpPerspective(img, self.tm, (self.w, self.h),
                                      *self.border_options
                                      )
            bboxes = self.trans_bboxs(self.tm, bboxs, img.shape[:2])
            if bboxes:
                img, bboxes = self.resize_img_and_boxs(img, bboxes,
                                                       img.shape[:2],
                                                       (self.h, self.w),
                                                       )
                return img, bboxes
            else:
                return img, False
        else:
            return cv2.warpPerspective(img, self.tm, (self.w, self.h),
                                       *self.border_options
                                       )


class CutOut(ImageLayer):
    '''
    Randomly cut portion of an image based on the paper:

        `Improved Regularization of Convolutional Neural Networks with Cutout`
                et al Terrance DeVries and Graham W. Taylor

    for clasification, this will cut a p[ortion of an image
    for object detection, thius will cut a portion of img under bbox
    _____________
    | Parameters|
    +-----------+
    * magnitude -> float [0, 1[: amount of the image (or bbox) to cut

    return: img, or img, and bboxs if the bboxes are given
    '''

    def __init__(self, magnitude=0.2, **kwrd):
        self.sub_type = 'spatial'
        self.name = 'Cut out'
        self.magnitude = magnitude
        super().__init__(**kwrd)

    def cut_cordinates(self, img, bbox):
        '''
        giben a bbox in fakg format return a cutout image in this bbox
        '''
        xmin, ymin, xmax, ymax = decode_bbox(bbox)
        w, h = xmax - xmin, ymax - ymin
        w_cut, h_cut = int(w*self.magnitude), int(h*self.magnitude)
        ymax, xmax = ymax - h_cut, xmax - w_cut
        h, w = np.random.randint(ymin, ymax), np.random.randint(xmin, xmax)
        img[h:h + h_cut, w:w + w_cut] = [0, 0, 0]

        return img

    def trans(self, img, bboxs=None):

        if bboxs:
            for bbox in bboxs:
                img = self.cut_cordinates(img, bbox)
            return img, bboxs

        else:
            self.magnitude = 0.04
            h, w = img.shape[:2]
            h_cut, w_cut = int(h*self.magnitude), int(w*self.magnitude)
            h, w = h - h_cut, w - w_cut

            trays = np.random.randint(10, 20)
            for _ in range(trays):
                h_ = np.random.randint(0, h)
                w_ = np.random.randint(0, w)

                img[h_:h_ + h_cut, w_:w_ + w_cut] = [0, 0, 0]
            return img


class GaussianFlitering(ImageLayer):
    '''
    Add gaussian blur in frames

     ___________
    | Parameters|
    +-----------+
    * k_size -> int: kernel size

    * k_sigma -> int: "intensity" blur
    '''
    def __init__(self, k_size=5, k_sigma=10, **kwrd):
        self.sub_type = 'color'
        self.k_size = k_size
        self.k_sigma = k_sigma
        self.kernel = cv2.getGaussianKernel(self.k_size, self.k_sigma)
        self.name = 'Gaussian blurring'
        super().__init__(**kwrd)

    def trans(self, img, bboxs=None):
        if bboxs:
            return cv2.sepFilter2D(img, -1, self.kernel, self.kernel), bboxs
        else:
            return cv2.sepFilter2D(img, -1, self.kernel, self.kernel)


class KMeansCC(ImageLayer):
    '''
    Returnern a segmented image, like a cartoonized img
     ___________
    | Parameters|
    +-----------+
    ncluster -> 10, numbers of clusters
    '''
    def __init__(self, ncluster=10, **kwrd):
        self.sub_type = 'color'
        # Stop Criteria
        self.sc = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # number of attempts
        self.na = 10
        # centroid initialization strategy
        self.cis = cv2.KMEANS_RANDOM_CENTERS
        self.ncluster = ncluster
        self.name = 'k-means clustering color with {} clusters'.format(
                                                                self.ncluster)
        super().__init__(**kwrd)

    def trans(self, img, bboxs=None):
        pixels = img.reshape((-1, 3))
        pixels = np.float32(pixels)

        _, labels, centers = cv2.kmeans(pixels, self.ncluster, None, self.sc,
                                        self.na, self.cis)

        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        if bboxs:
            return segmented_data.reshape(img.shape), bboxs
        else:
            return segmented_data.reshape(img.shape)


class BrightnessJitter(ImageLayer):
    '''
    Return images with random brightness shift
     ___________
    | Parameters|
    +-----------+

    magnitude -> None,  must be float if it given.
                 other way, the value will be taked randomly between from
                 0.5 to 1.5
    '''
    def __init__(self, magnitude=None, **kwrd):
        self.sub_type = 'color'
        self.magnitude = magnitude
        if self.magnitude:
            self.name = ('Brigghtness shift by magnitude of {}'
                         ''.format(self.magnitude))
            self.val = self.magnitude
        else:
            self.name = 'random Brigghtness jitter'
            self.val = np.random.uniform(0.5, 1.5)
        super().__init__(**kwrd)

    def reset_state(self):
        if self.magnitude is None:
            self.magnitude = np.random.uniform(0.5, 1.5)
        else:
            self.val = self.magnitude

    # @bboxs_check
    def trans(self, img, bboxs=None):
        if self.color_type(img) == 'rgb':
            hsv = self.to_hsv(img)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[..., 1] = hsv[..., 1] * self.val
            hsv[..., 1][hsv[..., 1] > 255] = 255
            hsv[..., 2] = hsv[..., 2] * self.val
            hsv[..., 2][hsv[..., 2] > 255] = 255
            hsv = np.array(hsv, dtype=np.uint8)
            hsv = np.array(hsv, dtype=np.uint8)
            if bboxs:
                return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), bboxs
            else:
                return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        else:  # gray image
            img = np.array(img, dtype=np.float64) * self.val
            img[img > 255] = 255
            if bboxs:
                return np.array(img, dtype=np.uint8), bboxs
            else:
                return np.array(img, dtype=np.uint8)


class SaturationJitter(ImageLayer):
    '''
    Return images with random brightness shift
     ___________
    | Parameters|
    +-----------+

    magnitude -> None,  must be integer if it given.
                 other way, the value will be taked randomly between from
                 0 to 60  (recomended not touch this)
    '''
    def __init__(self, magnitude=None,  **kwrd):
        self.sub_type = 'color'
        self.magnitude = magnitude
        self.val = self.magnitude
        if not self.magnitude:
            self.name = 'Random Saturation jitter'
            self.val = np.random.randint(0, 60)
        else:
            self.val = self.magnitude
            self.name = ('Saturation jitter by magnitude of {}'
                         ''.format(self.magnitude))
        super().__init__(**kwrd)

    def reset_state(self):
        if not self.magnitude:
            self.val = np.random.randint(0, 60)
        else:
            pass

    def trans(self, img, bboxs=None):
        hsv = self.to_hsv(img)
        h, s, v = np.dsplit(hsv, img.shape[-1])
        coin = np.random.choice(('x', 'o'))
        if coin == 'x':
            lim = 255 - self.val
            s[s > lim] = 255
            s[s <= lim] += self.val
        else:
            s[s < self.val] = 0
            s[s >= self.val] -= self.val

        hsv = np.concatenate((h, s, v), -1)

        img = self.hsv2rbg(hsv)
        if bboxs:
            return img, bboxs
        else:
            return img


class ContrastJitter(ImageLayer):
    def __init__(self, magnitude=None,  **kwrd):
        self.sub_type = 'color'
        self.magnitude = magnitude
        self.val = self.magnitude
        if not self.magnitude:
            self.name = 'Random Contrast jitter'
            self.val = np.random.randint(40, 100)
        else:
            self.val = self.magnitude
            self.name = ('Contrast jitter by magnitude of {}'
                         ''.format(self.magnitude))
        super().__init__(**kwrd)

    def reset_state(self):
        if not self.magnitude:
            self.val = np.random.randint(40, 100)
        else:
            pass

    def trans(self, img, bboxs=None):
        brightness = 10
        dummy = np.int16(img)
        dummy = dummy * (self.val/127+1) - self.val + brightness
        dummy = np.clip(dummy, 0, 255)
        img = np.uint8(dummy)
        if bboxs:
            return img, bboxs
        else:
            img

# ----------------------- BETA AUGMENTERS ----------------------------------- #
# This section are for beta augmenters, this need revision an rebuild


class Equalize(ImageLayer):
    '''
    Equalize histogram of color make it more disperse
     ___________
    | Parameters|
    +-----------+
    ---
    '''
    def __init__(self, **kwrd):
        self.sub_type = 'color'
        self.name = 'color equalizer'
        super().__init__(**kwrd)

    def trans(self, img, bboxs=None):
        container = []
        for chanel in cv2.split(img):
            container.append(cv2.equalizeHist(chanel))
        if bboxs:
            return cv2.merge(tuple(container)), bboxs
        else:
            return cv2.merge(tuple(container))


class WaveDistortion(ImageLayer):
    '''
    Returnern a waveled distorted image based on 'sin' and 'cos' grid
    _____________
    | Parameters|
    +-----------+
    rand_magnitude -> True, if the augmenter take a diferent magnitud each time
                            that wee useit (in videos, all frame have same
                            transformation)
    magnitude -> 30, as defect, this augmenter take mgnitude val between-30, 30
                     (only if rand_magnitude is 'True'), other way, magnitude
                     val are always 30
    '''
    def __init__(self, rand_magnitude=True, magnitude=30, **kwrd):
        self.sub_type = 'spatial'
        self.rand_magnitude = rand_magnitude
        self.magnitude = magnitude
        self.mag_val = None
        self.xmap = None
        self.ymap = None
        self.h = None
        self.w = None
        if (self.mag_val is None) and (self.rand_magnitude):
            self.mag_val = np.random.randint(-self.magnitude, self.magnitude)
        if (self.mag_val is None) and not (self.rand_magnitude):
            self.mag_val = self.magnitude
        if rand_magnitude:
            self.name = 'Wave Distortion with random magnitude between ' + \
                        '+- {}'.format(self.magnitude)
        else:
            self.name = 'Wave Distortion with magnitude {}'.format(
                        self.magnitude)
        super().__init__(**kwrd)

    def reset_state(self):
        if self.rand_magnitude:
            self.mag_val = np.random.randint(-self.magnitude, self.magnitude)
            self.make_map()

    def make_map(self):
        map = np.indices((self.h, self.w))
        xmap = map[1] + self.mag_val*np.cos(20*map[1]/self.w)
        ymap = map[0] + self.mag_val*np.sin(20*map[0]/self.h)
        self.xmap = np.float32(xmap)
        self.ymap = np.float32(ymap)

    def trans(self, img, bboxs=None):
        if self.xmap is None:
            self.h, self.w = img.shape[:2]
            self.make_map()

        if bboxs:
            return (cv2.remap(img, self.xmap, self.ymap,
                              cv2.INTER_LINEAR, None, cv2.BORDER_REPLICATE),
                    bboxs)
        else:
            return cv2.remap(img, self.xmap, self.ymap,
                             cv2.INTER_LINEAR, None, cv2.BORDER_REPLICATE)


class BarrelPincushionDis(ImageLayer):
    '''
    Return a image with Barrel or Pincushion distortion
    _____________
    | Parameters|
    +-----------+
    rand_magnitude -> True, if the augmenter take a diferent magnitud each time
                            that wee useit (in videos, all frame have same
                            transformation)
    magnitude -> 1.5, as defect, for this augmente its recomended to pass
                     magnitude between 0.5 to 1.5, more or less may be destroy
                     the data(only if rand_magnitude is 'False'), other way,
                     magnitude vals come between 0.5 and 1.5

    rand_magnitude -> True, Take a random magnitude between .5, 1.5
    rand_pos -> True
    '''
    def __init__(self, rand_pos=True, magnitude=1.5, rand_magnitude=True,
                 **kwrd):

        self.sub_type = 'spatial'
        self.name = 'Barrel/Pincushion Distortion'
        self.mag_val = None
        self.rand_pos = rand_pos
        self.rand_mag = rand_magnitude
        self.h = None
        self.w = None
        self.alpha = None
        self.beta = None
        self.xmap = None
        self.ymap = None
        super().__init__(**kwrd)

    def reset_state(self):
        if not self.rand_pos:
            self.alpha = 0.5
            self.beta = 0.5
        else:
            self.alpha = 0.5 + np.random.uniform(-.1, .1)
            self.beta = 0.5 + np.random.uniform(-.1, .1)
        if self.rand_mag:
            self.magnitude = np.random.uniform(.5, 1.5)
        self.make_map()

    def make_map(self):
        cv, ch = self.alpha*self.w, self.beta*self.h
        a = np.indices((self.h, self.w))
        a[0] = cv - a[0]
        a[1] = a[1] - ch
        r = np.sqrt(a[0]*a[0] + a[1]*a[1])
        r = (r ** self.magnitude)/(cv**(self.magnitude-1))
        t = np.arctan2(a[0], a[1])
        xmap = r*np.cos(t)
        xmap = xmap + ch
        ymap = r*np.sin(t)
        ymap = cv - ymap
        xmap = np.float32(xmap)
        ymap = np.float32(ymap)
        self.xmap = xmap
        self.ymap = ymap

    def trans(self, img, bboxs=None):
        if self.xmap is None:
            self.h, self.w = img.shape[:2]

            self.reset_state()
        if bboxs:
            return cv2.remap(img, self.xmap, self.ymap,
                             cv2.INTER_LINEAR, 0, cv2.BORDER_CONSTANT), bboxs
        else:
            return cv2.remap(img, self.xmap, self.ymap,
                             cv2.INTER_LINEAR, 0, cv2.BORDER_CONSTANT)
