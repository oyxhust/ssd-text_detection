from __future__ import print_function
import os
import numpy as np
from imdb import Imdb
import cv2
import re
import scipy.io as sio
import h5py


class SynthText(Imdb):
    """
    Implementation of Imdb for Pascal VOC datasets

    Parameters:
    ----------
    image_set : str
        set to be used, can be train, val, trainval, test
    year : str
        year of dataset, can be 2007, 2010, 2012...
    devkit_path : str
        devkit path of VOC dataset
    shuffle : boolean
        whether to initial shuffle the image list
    is_train : boolean
        if true, will load annotations
    """
    def __init__(self, image_set, devkit_path, shuffle=False, is_train=False):
        super(SynthText, self).__init__(image_set)
        self.image_set = image_set
        self.devkit_path = devkit_path
        self.data_path = os.path.join(devkit_path, 'SynthText')
        #self.extension = '.jpg'
        self.is_train = is_train

        self.classes = ['text', ]

        self.config = {#'use_difficult': True,
                       'comp_id': 'comp4',
                       'padding': 63}
        self.num_classes = len(self.classes)
        self.dataset_index, self.gt = self._load_dateset_index(shuffle)
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        if self.is_train:
            self.labels = self._load_image_labels()
    '''
    @property
    def cache_path(self):
        """
        make a directory to store all caches

        Returns:
        ---------
            cache path
        """
        cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path
    '''
    def _load_dateset_index(self, shuffle):
        """
        find out which indexes correspond to given image set (train or val)

        Parameters:
        ----------
        shuffle : boolean
            whether to shuffle the image list
        Returns:
        ----------
        entire list of images specified in the setting
        """
        dataset_index_file = os.path.join(self.data_path, self.image_set, 'gt.mat')
        assert os.path.exists(dataset_index_file), 'Path does not exist: {}'.format(dataset_index_file)
        gt = sio.loadmat(dataset_index_file)
        dataset_index = range(gt['imnames'].shape[1])
        if shuffle:
            np.random.shuffle(dataset_index)
        return dataset_index, gt

    def _load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)

        Parameters:
        ----------
        shuffle : boolean
            whether to shuffle the image list
        Returns:
        ----------
        entire list of images specified in the setting
        """
        image_set_index = []
        for idx in self.dataset_index:
            image_set_index.append(str(self.gt['imnames'][0, idx][0]))
        return image_set_index
    
    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = os.path.join(self.data_path, self.image_set, name)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file
    '''
    def _image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.gt is not None, "Dataset not initialized"
        name = str(self.gt['imnames'][0, index][0])
        image_file = os.path.join(self.data_path, self.image_set, name)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file
    '''
    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index, :, :]

    def _label_path_from_index(self, index):
        """
        given image index, find out annotation path

        Parameters:
        ----------
        index: int
            index of a specific image

        Returns:
        ----------
        full path of annotation file
        """
        label_array = self.gt['wordBB'][0, index]
        #print(label_array.shape)
        #assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)
        return label_array

    def _load_image_labels(self):
        """
        preprocess all ground-truths

        Returns:
        ----------
        labels packed in [num_images x max_num_objects x 5] tensor
        """
        temp = []
        max_objects = 0
        size_file_path = os.path.join(self.data_path, self.image_set, 'size.h5')
        with h5py.File(size_file_path, 'r') as f:
            size = f['size'][:]

        # load ground-truth from xml annotations
        for idx in self.dataset_index:
            width = float(size[1, idx])
            height = float(size[0, idx])
            label = []
            label_array = self._label_path_from_index(idx)
            if label_array.ndim == 3:
                for i in xrange(label_array.shape[2]):
                    xmin = float(np.min(label_array[0,:,i])) / width
                    ymin = float(np.min(label_array[1,:,i])) / height
                    xmax = float(np.max(label_array[0,:,i])) / width
                    ymax = float(np.max(label_array[1,:,i])) / height
                    cls_id = self.classes.index('text')
                    label.append([cls_id, xmin, ymin, xmax, ymax])
            else:
                xmin = float(np.min(label_array[0,:])) / width
                ymin = float(np.min(label_array[1,:])) / height
                xmax = float(np.max(label_array[0,:])) / width
                ymax = float(np.max(label_array[1,:])) / height
                cls_id = self.classes.index('text')
                label.append([cls_id, xmin, ymin, xmax, ymax])

            temp.append(np.array(label))
            max_objects = max(max_objects, len(label))

        #print(max_objects)

        # add padding to labels so that the dimensions match in each batch
        # TODO: design a better way to handle label padding
        assert max_objects > 0, "No objects found for any of the images"
        assert max_objects <= self.config['padding'], "# obj exceed padding"
        self.padding = self.config['padding']
        labels = []
        for label in temp:
            label = np.lib.pad(label, ((0, self.padding-label.shape[0]), (0,0)), \
                               'constant', constant_values=(-1, -1))
            labels.append(label)

        return np.array(labels)

'''
    def evaluate_detections(self, detections):
        """
        top level evaluations
        Parameters:
        ----------
        detections: list
            result list, each entry is a matrix of detections
        Returns:
        ----------
            None
        """
        # make all these folders for results
        result_dir = os.path.join(self.devkit_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        year_folder = os.path.join(self.devkit_path, 'results', 'VOC' + self.year)
        if not os.path.exists(year_folder):
            os.mkdir(year_folder)
        res_file_folder = os.path.join(self.devkit_path, 'results', 'VOC' + self.year, 'Main')
        if not os.path.exists(res_file_folder):
            os.mkdir(res_file_folder)

        self.write_pascal_results(detections)
        self.do_python_eval()

    def get_result_file_template(self):
        """
        this is a template
        VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt

        Returns:
        ----------
            a string template
        """
        res_file_folder = os.path.join(self.devkit_path, 'results', 'VOC' + self.year, 'Main')
        comp_id = self.config['comp_id']
        filename = comp_id + '_det_' + self.image_set + '_{:s}.txt'
        path = os.path.join(res_file_folder, filename)
        return path

    def write_pascal_results(self, all_boxes):
        """
        write results files in pascal devkit path
        Parameters:
        ----------
        all_boxes: list
            boxes to be processed [bbox, confidence]
        Returns:
        ----------
        None
        """
        for cls_ind, cls in enumerate(self.classes):
            print('Writing {} VOC results file'.format(cls))
            filename = self.get_result_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_set_index):
                    dets = all_boxes[im_ind]
                    if dets.shape[0] < 1:
                        continue
                    h, w = self._get_imsize(self.image_path_from_index(im_ind))
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        if (int(dets[k, 0]) == cls_ind):
                            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                    format(index, dets[k, 1],
                                           int(dets[k, 2] * w) + 1, int(dets[k, 3] * h) + 1,
                                           int(dets[k, 4] * w) + 1, int(dets[k, 5] * h) + 1))

    def do_python_eval(self):
        """
        python evaluation wrapper

        Returns:
        ----------
        None
        """
        annopath = os.path.join(self.data_path, 'Annotations', '{:s}.xml')
        imageset_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
        cache_dir = os.path.join(self.cache_path, self.name)
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self.year) < 2010 else False
        print('VOC07 metric? ' + ('Y' if use_07_metric else 'No'))
        for cls_ind, cls in enumerate(self.classes):
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imageset_file, cls, cache_dir,
                                     ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
        print('Mean AP = {:.4f}'.format(np.mean(aps)))

    def _get_imsize(self, im_name):
        """
        get image size info
        Returns:
        ----------
        tuple of (height, width)
        """
        img = cv2.imread(im_name)
        return (img.shape[0], img.shape[1])
'''
