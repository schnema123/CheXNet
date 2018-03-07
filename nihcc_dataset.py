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

def _parse_line(line):

    csv_line = tf.decode_csv(line, FIELDS_DEFAULT)

    # Read image
    image_name = csv_line[IMAGE_FIELD]

    findings = csv_line[FINDINGS_FIELD]
    findings_array = findings



def create_dataset():

    # Pass this in as a flag
    ds = tf.data.TextLineDataSet(DATA_PATH + TRAIN_DATA).skip(1)
    ds = ds.map(_parse_line)

