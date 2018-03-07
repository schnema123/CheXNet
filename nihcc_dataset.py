import tensorflow as tf

DATA_PATH = "../nihcc/"
IMAGE_PATH = DATA_PATH + "image_all/"

TRAIN_DATA = "data_train.csv"
EVAL_DATA = "data_eval.csv"
TEST_DATA = "data_test.csv"

IMAGE_FIELD = 0
FINDINGS_FIELD = 1

FIELDS_DEFAULT = [[""] * 11]

FINDINGS = ["No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
            "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

def _parse_tfrecord(tfrecord):

    features = {
        'image': tf.FixedLenFeature([], tf.String),
        'findings': tf.FixedLenFeature([15], tf.int64)
    }
    record = tf.parse_single_example(tfrecord)

    images_path = tf.constant("../nihcc/images_scaled/")
    image_full_path = tf.string_join([images_path, record[image]])

    image_string = tf.read_file(image_full_path)
    image_decoded = tf.image.decode_image(image_string)

    return image_decoded, record["findings"]


def create_dataset():

    tf_record_path = "../nihcc/tfrecords/train.tfrecord"

    # Pass this in as a flag
    ds = tf.data.TFRecordDataset(tf_record_path)
    ds = ds.map(_parse_tfrecord)

