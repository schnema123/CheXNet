import os
import csv
import random

import tensorflow as tf

IMAGES_PATH = "../dataset/"

FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
            "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
NUM_FINDINGS = len(FINDINGS)


def _read_image(filepath, label):

    image_string = tf.read_file(filepath)

    image_decoded = tf.image.decode_png(image_string, channels=3)
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)

    # TODO: What about normalization on the whole dataset?
    # TODO: pytorch has a RandomResizedCrop function. Look into that?
    # TODO: It does seem that tensorflow acts weird on resizing images
    # https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35
    image_decoded = tf.image.resize_images(image_decoded, [224, 224])
    image_decoded = tf.image.per_image_standardization(image_decoded)
    image_decoded = tf.image.random_flip_left_right(image_decoded)

    # image_decoded = tf.reshape(image, [-1, 224, 224])
    # image_decoded = tf.stack([image, image, image], axis=3)
    image_decoded.set_shape([224, 224, 3])

    return image_decoded, label

def _read_txt(filename):

    images = []
    labels = []

    num_no_finding = 0
    num_finding = dict()
    for finding in FINDINGS:
        num_finding[finding] = 0

    with open(filename) as txt_file:

        line = True
        while line:

            line = txt_file.readline()
            if line:

                lineItems = line.split()

                imagePath = os.path.join(IMAGES_PATH, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]

                for idx, item in enumerate(imageLabel):
                    if item == 1:
                        finding = FINDINGS[idx]
                        num_finding[finding] += 1 

                images.append(imagePath)
                labels.append(imageLabel)

    print("[*] {} labels and {} images in {}".format(len(labels), len(labels), filename))
    print("[*] Number of findings:")
    print("[*] No Finding: {}".format(num_no_finding))
    for finding in FINDINGS:
        print("[*] {}: {}".format(finding, num_finding[finding]))

    return images, labels


def create_dataset(mode):

    # Get csv data
    if (mode == tf.estimator.ModeKeys.TRAIN):
        filename = "../dataset/train_1.txt"
    elif (mode == tf.estimator.ModeKeys.EVAL):
        filename = "../dataset/val_1.txt"
    elif (mode == tf.estimator.ModeKeys.PREDICT):
        filename = "../dataset/test_1.txt"

    # Reads image file names and labels
    images, labels = _read_txt(filename)   

    images = tf.convert_to_tensor(images)
    labels= tf.convert_to_tensor(labels)

    ds = tf.contrib.data.Dataset.from_tensor_slices((images, labels))

    # if mode == tf.estimator.ModeKeys.TRAIN:
    #    ds = ds.repeat(2)

    ds = ds.shuffle(100000)
    ds = ds.map(_read_image, num_threads=20, output_buffer_size=50)
    ds = ds.batch(16)

    return ds


def input_fn(mode):
    """An input function for training"""
    ds = create_dataset(mode)
    features, labels = ds.make_one_shot_iterator().get_next()
    return {"input_1": features}, labels