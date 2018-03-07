import os
import csv


import tensorflow as tf

DATA_PATH = "../nihcc/"

TRAIN_DATA = "data_train.csv"
EVAL_DATA = "data_eval.csv"
TEST_DATA = "data_test.csv"

IMAGE_FIELD = 0
FINDINGS_FIELD = 1

FIELDS_DEFAULT = [[""] * 11]

FINDINGS = ["No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
            "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
NUM_FINDINGS = len(FINDINGS)


def _read_image(filename, label):

    images_path = tf.constant("J:/BA/nihcc/images_scaled/")
    file_path = tf.string_join([images_path, filename])
    image_string = tf.read_file(file_path)
    image_decoded = tf.image.decode_image(image_string)
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_decoded.set_shape((224, 224, 1))

    return image_decoded, label


def _read_csv():

    images = []
    labels = []

    csv_path = os.path.join(DATA_PATH, "Data_Entry_2017_small.csv")
    with open(csv_path) as csv_file:

        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            images.append(row["Image Index"])

            # Make vector of labels
            findings = row["Finding Labels"].split("|")

            findings_vector = [0 for _ in range(NUM_FINDINGS)]
            for finding in findings:
                findings_vector[FINDINGS.index(finding)] = 1

            assert(1 in findings_vector)
                
            labels.append(findings_vector)

    return tf.convert_to_tensor(images), tf.convert_to_tensor(labels)


def create_dataset():

    # Get csv data
    images, labels = _read_csv()

    print(images)
    print(labels)

    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(_read_image)

    return ds


create_dataset()