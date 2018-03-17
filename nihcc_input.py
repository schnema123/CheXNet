import os
import csv

import tensorflow as tf

IMAGE_FIELD = 0
FINDINGS_FIELD = 1

FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
            "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
NUM_FINDINGS = len(FINDINGS)


def _read_image(filename, label):

    images_path = tf.constant("../images_scaled/")
    file_path = tf.string_join([images_path, filename])
    image_string = tf.read_file(file_path)

    image_decoded = tf.image.decode_image(image_string)
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)

    image_decoded = tf.image.per_image_standardization(image_decoded)
    image_decoded = tf.image.random_flip_left_right(image_decoded)

    # image_decoded = tf.reshape(image, [-1, 224, 224])
    # image_decoded = tf.stack([image, image, image], axis=3)
    image_decoded.set_shape([224, 224, 1])

    return image_decoded, label


def _read_csv(filename):

    images = []
    labels = []

    with open(filename) as csv_file:

        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            images.append(row["Image Index"])

            # Make vector of labels
            findings = row["Finding Labels"].split("|")

            findings_vector = [0 for _ in range(NUM_FINDINGS)]
            for finding in findings:
                if finding != "No Finding":
                    findings_vector[FINDINGS.index(finding)] = 1

            labels.append(findings_vector)

    print("[*] {} labels and {} images in {}".format(len(labels), len(labels), filename))
    return images, labels


def create_dataset(mode):

    # Get csv data
    if (mode == tf.estimator.ModeKeys.TRAIN):
        filename = "../dataset/Data_Train.csv"
    elif (mode == tf.estimator.ModeKeys.EVAL):
        filename = "../dataset/Data_Eval.csv"
    elif (mode == tf.estimator.ModeKeys.PREDICT):
        filename = "../dataset/Data_Test.csv"

    # Reads image file names and labels
    images, labels = _read_csv(filename)   

    images = tf.convert_to_tensor(images)
    labels= tf.convert_to_tensor(labels)

    ds = tf.contrib.data.Dataset.from_tensor_slices((images, labels))

    # TODO: Look into shuffling and then repeating vs vice versa
    # Shuffle the dataset

    if (mode == tf.estimator.ModeKeys.TRAIN):
      ds = ds.repeat(2)

    ds = ds.shuffle(100000)
    ds = ds.map(_read_image, num_threads=10, output_buffer_size=10)
    ds = ds.batch(16)

    return ds


def input_fn(mode):
    """An input function for training"""
    ds = create_dataset(mode)
    features, labels = ds.make_one_shot_iterator().get_next()
    return {"input_1": features}, labels