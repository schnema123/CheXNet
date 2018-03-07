import csv
import random
import tensorflow as tf
import scipy as sp

NIHCC_PATH = "../nihcc"
NIHCC_IMAGE_PATH = "../nihcc/images_scaled"

FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
            "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
NUM_FINDINGS = len(FINDINGS)

# Convert images to TFRecords

train_writer = tf.python_io.TFRecordWriter(NIHCC_PATH + "/tfrecords/train.tfrecords")
eval_writer = tf.python_io.TFRecordWriter(NIHCC_PATH + "/tfrecords/eval.tfrecords")
test_writer = tf.python_io.TFRecordWriter(NIHCC_PATH + "/tfrecords/test.tfrecords")

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int_list_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def parse_rows(rows):
    random.shuffle(rows)

    total_row_number = len(rows)
    current_row_number = 0
    for row in rows:
        # Get image file name
        image_file = row["Image Index"]

        # Get findings
        findings = row["Finding Labels"]

        findings_vector = []
        for finding in findings.split("|"):
            if (finding != "No Finding"):
                findings_vector.append(FINDINGS.index(finding))

        example = tf.train.Example(features=tf.train.Features(feature=
        {
            "image": _bytes_feature(tf.compat.as_bytes(image_file)),
            "findings": _int_list_feature(findings_vector)
        }))

        if (current_row_number < total_row_number * 0.6):
            train_writer.write(example.SerializeToString())
        elif (current_row_number < total_row_number * 0.8):
            eval_writer.write(example.SerializeToString())
        else:
            test_writer.write(example.SerializeToString())

        current_row_number = current_row_number + 1

    train_writer.close()
    eval_writer.close()
    test_writer.close()


with open(NIHCC_PATH + "/Data_Entry_2017.csv") as csv_file:
    rows = []

    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        rows.append(row)

    parse_rows(rows)