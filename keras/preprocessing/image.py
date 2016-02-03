from __future__ import absolute_import

import numpy as np
import re
from scipy import ndimage
from scipy import linalg

from os import listdir
from os.path import isfile, join
import random
import math
from six.moves import range
import threading

from math import cos, sin, radians

'''Fairly basic set of tools for realtime data augmentation on image data.
Can easily be extended to include new transformations, new preprocessing methods, etc...
'''

def coordinatesToLearnAreStillInScope(coordinates, abscissaScale, ordinateScale, margin):

    #get extremum points coordinates
    _abcissa_max = coordinates[np.argmax(coordinates[:, 0])]
    _ordinate_max = coordinates[np.argmax(coordinates[:, 1])]
    _abcissa_min = coordinates[np.argmin(coordinates[:, 0])]
    _ordinate_min = coordinates[np.argmin(coordinates[:, 1])]

    #if extremum are outside rotation is cancelled
    if _abcissa_max[0] > abscissaScale[1] - margin:
        return False

    if _ordinate_max[1] > ordinateScale[1] - margin:
        return False

    if _abcissa_min[0] < abscissaScale[0] + margin:
        return False

    if _ordinate_min[1] < ordinateScale[0] + margin:
        return False

    return True   

def random_rotation(x, y, rg, fill_mode="nearest", cval=0., y_abscissaScale=(-1,1), y_ordinateScale=(-1,1), margin=0):
    angle = random.uniform(-rg, rg)

    _y = None

    if y is not None:
        #@see https://en.wikipedia.org/wiki/Rotation_matrix
        _y = np.transpose(np.dot(np.array([ [cos(radians(-angle)), -sin(radians(-angle))], [sin(radians(-angle)), cos(radians(-angle))] ]), np.transpose(y)))

        if not coordinatesToLearnAreStillInScope(_y, y_abscissaScale, y_ordinateScale, margin):
            return (x,y)

    x = ndimage.interpolation.rotate(x, angle,
                                     axes=(1, 2),
                                     reshape=False,
                                     mode=fill_mode,
                                     cval=cval)
    return (x,_y)




def random_shift(x, y, wrg, hrg, fill_mode="nearest", cval=0., y_abscissaScale=(-1,1), y_ordinateScale=(-1,1), margin=0):
    crop_left_pixels = 0
    crop_top_pixels = 0
    _y = None

    if wrg:
        crop = random.uniform(0., wrg)
        split = random.uniform(0, 1)
        crop_left_pixels = int(split*crop*x.shape[1])

    if hrg:
        crop = random.uniform(0., hrg)
        split = random.uniform(0, 1)
        crop_top_pixels = int(split*crop*x.shape[2])

    if y is not None:
        _y = np.add(y, [[crop_top_pixels/float(x.shape[2]) * (y_ordinateScale[1] - y_ordinateScale[0]), crop_left_pixels/float(x.shape[1]) * (y_abscissaScale[1] - y_abscissaScale[0])]])

        if not coordinatesToLearnAreStillInScope(_y, y_abscissaScale, y_ordinateScale, margin):
            return (x,y)

    x = ndimage.interpolation.shift(x, (0, crop_left_pixels, crop_top_pixels),
                                    order=0,
                                    mode=fill_mode,
                                    cval=cval)
    return (x,_y)



def horizontal_flip(x,y, swapingIndex=[]):
    if y is not None:
        y[:,0] = y[:,0] * -1.0

        for (i,j) in swapingIndex:
            _temp = np.copy(y[i])
            y[i] = y[j]
            y[j] = _temp
        
    for i in range(x.shape[0]):
        x[i] = np.fliplr(x[i])
    return (x,y)


def vertical_flip(x,y, swapingIndex=[]):
    if y is not None:
        y[:,1] = y[:,1] * -1.0

        for (i,j) in swapingIndex:
            _temp = np.copy(y[i])
            y[i] = y[j]
            y[j] = _temp
        
    for i in range(x.shape[0]):
        x[i] = np.fliplr(x[i])
    return (x,y)


def random_barrel_transform(x, intensity):
    # TODO
    pass


def random_shear(x, intensity, fill_mode="nearest", cval=0.):
    shear = random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1.0, -math.sin(shear), 0.0],
                            [0.0, math.cos(shear), 0.0],
                            [0.0, 0.0, 1.0]])
    x = ndimage.interpolation.affine_transform(x, shear_matrix,
                                               mode=fill_mode,
                                               order=3,
                                               cval=cval)
    return x


def random_channel_shift(x, rg):
    # TODO
    pass


def random_zoom(x, rg, fill_mode="nearest", cval=0.):
    zoom_w = random.uniform(1.-rg, 1.)
    zoom_h = random.uniform(1.-rg, 1.)
    x = ndimage.interpolation.zoom(x, zoom=(1., zoom_w, zoom_h),
                                   mode=fill_mode,
                                   cval=cval)
    return x  # shape of result will be different from shape of input!


def array_to_img(x, y=None, scale=True):
    from PIL import Image, ImageDraw

    x = x.transpose(1, 2, 0)
    if scale:
        x += max(-np.min(x), 0)
        x /= np.max(x)
        x *= 255
    if x.shape[2] == 3:
        # RGB
        image = Image.fromarray(x.astype("uint8"), "RGB")
    else:
        # grayscale
        image = Image.fromarray(x[:, :, 0].astype("uint8"), "L")

    if y is not None:
        y = y.reshape(-1,2)
        y[:,0] *=  x.shape[0]/2
        y[:,0] +=  x.shape[0]/2
        y[:,1] *=  x.shape[1]/2
        y[:,1] +=  x.shape[1]/2
        ImageDraw.Draw(image).point(map(tuple, y ))

    return image


def img_to_array(img):
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        # RGB: height, width, channel -> channel, height, width
        x = x.transpose(2, 0, 1)
    else:
        # grayscale: height, width -> channel, height, width
        x = x.reshape((1, x.shape[0], x.shape[1]))
    return x


def load_img(path, grayscale=False):
    from PIL import Image
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    return img


def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [join(directory, f) for f in listdir(directory)
            if isfile(join(directory, f)) and re.match('([\w]+\.(?:' + ext + '))', f)]


class ImageDataGenerator(object):
    '''Generate minibatches with
    realtime data augmentation.
    '''
    def __init__(self,
                 featurewise_center=True,  # set input mean to 0 over the dataset
                 samplewise_center=False,  # set each sample mean to 0
                 featurewise_std_normalization=True,  # divide inputs by std of the dataset
                 samplewise_std_normalization=False,  # divide each input by its std
                 zca_whitening=False,  # apply ZCA whitening
                 rotation_range=0.,  # degrees (0 to 180)
                 width_shift_range=0.,  # fraction of total width
                 height_shift_range=0.,  # fraction of total height
                 shear_range=0.,  # shear intensity (shear angle in radians)
                 horizontal_flip=False,
                 vertical_flip=False):

        self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None
        self.lock = threading.Lock()

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        b = 0
        total_b = 0
        while 1:
            if b == 0:
                if seed is not None:
                    np.random.seed(seed + total_b)

                if shuffle:
                    index_array = np.random.permutation(N)
                else:
                    index_array = np.arange(N)

            current_index = (b * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
            else:
                current_batch_size = N - current_index

            if current_batch_size == batch_size:
                b += 1
            else:
                b = 0
            total_b += 1
            yield index_array[current_index: current_index + current_batch_size], current_index, current_batch_size

    def flow(self, X, y, batch_size=32, shuffle=False, seed=None,
             save_to_dir=None, save_prefix="", save_format="jpeg", horizontalSwapingIndex = None, verticalSwapingIndex = None):
        '''If you want to also change y value during random_transform, y must have (_,2) shaped and be normalized between [-1;1]
        For now only shift, rotation and flip change y values.
        horizontalSwaping is used to swap y indexes (to handle leftmost/rightmost)
        verticalSwaping is used to swap y indexes (to handle upper/lower)
        '''
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.flow_generator = self._flow_index(X.shape[0], batch_size, shuffle, seed)
        self.horizontalSwaping = horizontalSwapingIndex
        self.verticalSwaping = verticalSwapingIndex
        return self

    def __iter__(self):
        # needed if we want to do something like for x,y in data_gen.flow(...):
        return self

    def next(self):
        # for python 2.x
        # Keep under lock only the mechainsem which advance the indexing of each batch
        # see # http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.flow_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        bX = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        bY = np.zeros((current_batch_size, len(self.y[0].flatten())))

        #test if data to learn from are coordinates that need to be transformed with raw image
        if self.y[0].ndim == 2:
            for i, j in enumerate(index_array):
                x = self.X[j]
                x, bY[i] = self.random_transform(x.astype("float32"), self.y[j])
                x = self.standardize(x)
                bX[i] = x
        else:
            for i, j in enumerate(index_array):
                x = self.X[j]
                x, _ = self.random_transform(x.astype("float32"))
                x = self.standardize(x)
                bX[i] = x           
        if self.save_to_dir:
            if self.y[0].ndim == 2:
                for i in range(current_batch_size):
                    img = array_to_img(bX[i], bY[i], scale=True)
                    img.save(self.save_to_dir + "/" + self.save_prefix + "_" + str(current_index + i) + "." + self.save_format)
            else:
                for i in range(current_batch_size):
                    img = array_to_img(bX[i], scale=True)
                    img.save(self.save_to_dir + "/" + self.save_prefix + "_" + str(current_index + i) + "." + self.save_format)
        return bX, bY

    def __next__(self):
        # for python 3.x
        return self.next()

    def standardize(self, x):
        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= self.std

        if self.zca_whitening:
            flatx = np.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2]))
            whitex = np.dot(flatx, self.principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))

        if self.samplewise_center:
            x -= np.mean(x)
        if self.samplewise_std_normalization:
            x /= np.std(x)

        return x

    def random_transform(self, x, y = None):
        if self.rotation_range:
            x, y = random_rotation(x, y, self.rotation_range)
        if self.width_shift_range or self.height_shift_range:
            x, y = random_shift(x, y, self.width_shift_range, self.height_shift_range)
        if self.horizontal_flip:
            if random.random() < 0.5:
                x, y = horizontal_flip(x, y, self.horizontalSwaping)
        if self.vertical_flip:
            if random.random() < 0.5:
                x, y = vertical_flip(x, y, self.verticalSwaping)
        if self.shear_range:
            x = random_shear(x,self.shear_range)
        # TODO:
        # zoom
        # barrel/fisheye
        # shearing
        # channel shifting
        if y is None:
            return (x, None)
        else:
            return (x, y.flatten())

    def fit(self, X,
            augment=False,  # fit on randomly augmented samples
            rounds=1,  # if augment, how many augmentation passes over the data do we use
            seed=None):
        '''Required for featurewise_center, featurewise_std_normalization and zca_whitening.
        '''
        X = np.copy(X)
        if augment:
            aX = np.zeros(tuple([rounds*X.shape[0]]+list(X.shape)[1:]))
            for r in range(rounds):
                for i in range(X.shape[0]):
                    img = array_to_img(X[i])
                    img = self.random_transform(img)
                    aX[i+r*X.shape[0]] = img_to_array(img)
            X = aX

        if self.featurewise_center:
            self.mean = np.mean(X, axis=0)
            X -= self.mean
        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)
            X /= self.std

        if self.zca_whitening:
            flatX = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]))
            fudge = 10e-6
            sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + fudge))), U.T)


class GraphImageDataGenerator(ImageDataGenerator):
    '''Example of how to build a generator for a Graph model
    '''

    def next(self):
        bX, bY = super(GraphImageDataGenerator, self).next()
        return {'input': bX, 'output': bY}
