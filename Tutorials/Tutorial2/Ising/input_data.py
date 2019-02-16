"""Functions for downloading and reading MNIST data."""
import gzip
import os
import urllib
import numpy
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms, utils
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def maybe_download(filename, work_directory):
    filepath = os.path.join(work_directory, filename)
    return filepath

def extract_images(filename,lx):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename,'aaaaaa')
    
    data=numpy.loadtxt(filename,dtype='int64')
    dim=data.shape[0]
    data=data.reshape(dim, lx, lx, 1) 
    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    data = data.reshape(data.shape[0],
                        data.shape[1] * data.shape[2])
    # Convert from [0, 255] -> [0.0, 1.0].
    data = data.astype(numpy.float64)
    # images = numpy.multiply(images, 1.0 / 255.0) # commented since it is ising variables
    data = numpy.multiply(data, 1.0 ) # multiply by one, instead
    print(data.shape)
    return data

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def extract_labels(nlabels,filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename,'bbbccicicicicib')

    labels=numpy.loadtxt(filename,dtype='int64')
      
    if one_hot:
       print("LABELS ONE HOT")
       print(labels.shape)
       XXX=dense_to_one_hot(labels,nlabels)
       print(XXX.shape)
       return dense_to_one_hot(labels,nlabels)
    print("LABELS")
    print(labels.shape)
    return labels


class CustomDataSet(Dataset):
    def __init__(self, images, labels, lx, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        # self.to_tensor = transforms.Compose([
        #                         transforms.Resize(lx*lx),
        #                         transforms.ToTensor()])

    def __getitem__(self, index):
        single_image = self._images[index]
        single_image_label = self._labels[index]

        # img_as_tensor = self.to_tensor(single_image)

        return (single_image, single_image_label)

    def __len__(self):
        return self._num_examples

def read_data_sets(nlabels, lx, train_dir, fake_data=False, one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
        data_sets.train = CustomDataSet([], [], lx, fake_data=True)
        data_sets.validation = CustomDataSet([], [], lx, fake_data=True)
        data_sets.test = CustomDataSet([], [], lx, fake_data=True)
        return data_sets
    TRAIN_IMAGES = 'Xtrain.txt'
    TRAIN_LABELS = 'ytrain.txt'
    TEST_IMAGES = 'Xtest.txt'
    TEST_LABELS = 'ytest.txt'
    VALIDATION_SIZE = 0
    local_file = maybe_download(TRAIN_IMAGES, train_dir)
    train_images = extract_images(local_file,lx)
    local_file = maybe_download(TRAIN_LABELS, train_dir)
    train_labels = extract_labels(nlabels,local_file, one_hot=one_hot)
    local_file = maybe_download(TEST_IMAGES, train_dir)
    test_images = extract_images(local_file,lx)
    local_file = maybe_download(TEST_LABELS, train_dir)
    test_labels = extract_labels(nlabels,local_file, one_hot=one_hot)
    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]
    data_sets.train = TensorDataset(torch.tensor(train_images), torch.tensor(train_labels))
    data_sets.test = TensorDataset(torch.tensor(test_images), torch.tensor(test_labels))
    if (VALIDATION_SIZE!=0):
        data_sets.validation = TensorDataset(torch.tensor(validation_images), torch.tensor(validation_labels))
    # data_sets.train = CustomDataSet(train_images, train_labels, lx)
    # data_sets.test = CustomDataSet(test_images, test_labels, lx)
    # if (VALIDATION_SIZE!=0):
    #     data_sets.validation = CustomDataSet(validation_images, validation_labels, lx)
    return data_sets
