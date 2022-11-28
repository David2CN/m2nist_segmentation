import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def read_image_and_annotation(image, annotation):
    '''
    Casts the image and annotation to their expected data type and
    normalizes the input image so that each pixel is in the range [-1, 1]

    Args:
      image (numpy array) -- input image
      annotation (numpy array) -- ground truth label map

    Returns:
      preprocessed image-annotation pair
    '''

    image = tf.cast(image, dtype=tf.float32)
    image = tf.reshape(image, (image.shape[0], image.shape[1], 1,))
    annotation = tf.cast(annotation, dtype=tf.int32)
    image = image / 127.5
    image -= 1

    return image, annotation


def get_training_dataset(images, annos, batch_size=32):
    '''
    Prepares shuffled batches of the training set.

    Args:
      images (list of strings) -- paths to each image file in the train set
      annos (list of strings) -- paths to each label map in the train set

    Returns:
      tf Dataset containing the preprocessed train set
    '''
    training_dataset = tf.data.Dataset.from_tensor_slices((images, annos))
    training_dataset = training_dataset.map(read_image_and_annotation)

    training_dataset = training_dataset.shuffle(512, reshuffle_each_iteration=True)
    training_dataset = training_dataset.batch(batch_size)
    training_dataset = training_dataset.repeat()
    training_dataset = training_dataset.prefetch(-1)

    return training_dataset


def get_validation_dataset(images, annos, batch_size=32):
    '''
    Prepares batches of the validation set.

    Args:
      images (list of strings) -- paths to each image file in the val set
      annos (list of strings) -- paths to each label map in the val set

    Returns:
      tf Dataset containing the preprocessed validation set
    '''
    validation_dataset = tf.data.Dataset.from_tensor_slices((images, annos))
    validation_dataset = validation_dataset.map(read_image_and_annotation)
    validation_dataset = validation_dataset.batch(batch_size)
    validation_dataset = validation_dataset.repeat()

    return validation_dataset


def get_test_dataset(images, annos, batch_size=32):
    '''
    Prepares batches of the test set.

    Args:
      images (list of strings) -- paths to each image file in the test set
      annos (list of strings) -- paths to each label map in the test set

    Returns:
      tf Dataset containing the preprocessed validation set
    '''
    test_dataset = tf.data.Dataset.from_tensor_slices((images, annos))
    test_dataset = test_dataset.map(read_image_and_annotation)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

    return test_dataset


def load_images_and_segments(data_path):
    '''
    Loads the images and segments as numpy arrays from npy files 
    and makes splits for training, validation and test datasets.

    Returns:
      3 tuples containing the train, val, and test splits
    '''

    #Loads images and segmentation masks.
    images = np.load(f'{data_path}/combined.npy')
    segments = np.load(f'{data_path}/segmented.npy')

    #Makes training, validation, test splits from loaded images and segmentation masks.
    train_images, val_images, train_annos, val_annos = train_test_split(images, segments, test_size=0.2, shuffle=True)
    val_images, test_images, val_annos, test_annos = train_test_split(val_images, val_annos, test_size=0.2, shuffle=True)

    return (train_images, train_annos), (val_images, val_annos), (test_images, test_annos)


