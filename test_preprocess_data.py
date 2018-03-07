import tensorflow as tf

tfrecord_train = "../nihcc/tfrecords/train.tfrecords"
tfrecord_train_iterator = tf.python_io.tf_record_iterator(path=tfrecord_train)

num_records = 0

for record in tfrecord_train_iterator:

    num_records = num_records + 1

    example = tf.train.Example()
    example.ParseFromString(record)

    img_file_name_raw = example.features.feature["image"].bytes_list.value[0]
    img_file_name = tf.compat.as_text(img_file_name_raw)

    labels = example.features.feature["findings"].int64_list.value

    print(img_file_name)
    print(labels)

print(num_records)
