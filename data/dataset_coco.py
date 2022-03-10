"""
Read the CoCo Dataset in form of TFRecord
Create tensorflow dataset and do the augmentation

ref:https://jkjung-avt.github.io/tfrecords-for-keras/
ref:https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py
"""
import os
import tensorflow as tf
from data import parser


# Todo encapsulate it as a class, here is the place to get dataset(train, eval, test)
def prepare_dataloader(img_h, img_w, tfrecord_dir, batch_size, subset="train"):
    parse = parser.Parser(output_size=[img_h, img_w], # (h,w)
                            match_threshold=0.5,
                            unmatched_threshold=0.5,
                            mode=subset)
    files = tf.io.matching_files(os.path.join(tfrecord_dir, "*.*"))
    num_shards = tf.cast(tf.shape(files)[0], tf.int64)
    shards = tf.data.Dataset.from_tensor_slices(files)
    shards = shards.shuffle(num_shards)
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset,
                                cycle_length=num_shards,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=2048)
    dataset = dataset.map(map_func=parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
