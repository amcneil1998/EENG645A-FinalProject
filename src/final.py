import numpy as np
import os
import pandas as pd
import re
from typing import List

import tensorflow as tf
import cv2
import typing
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, BatchNormalization, Concatenate, Activation
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.metrics import mean_absolute_error
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.keras import TqdmCallback


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte., thanks Joe"""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_tf_records(input_loc,
                    output_loc,
                    save_loc,
                    max_per_record=4500):
    '''
    :param input_loc: os.path to the input data to be used for this record set
    :param output_loc: os.path to the output data to be used for this record set
    :param max_per_record:
    :return: None, just writes tf record to disk for future use.
    '''
    # lets kick off with making a list of input and output files
    input_files = []
    sub_dirs = os.listdir(input_loc)
    for i in range(len(sub_dirs)):
        cur_dir = os.path.join(input_loc, sub_dirs[i])
        dir_files = os.listdir(cur_dir)
        for j in range(len(dir_files)):
            # be safe and make sure that we only grab pictures
            if dir_files[j].endswith(".png"):
                input_files.append(os.path.join(cur_dir, dir_files[j]))
    output_files = []
    sub_dirs = os.listdir(output_loc)
    for i in range(len(sub_dirs)):
        cur_dir = os.path.join(output_loc, sub_dirs[i])
        dir_files = os.listdir(cur_dir)
        for j in range(len(dir_files)):
            # we only want the labelIds ones, these take up less space and make it easier to encode
            if dir_files[j].endswith("labelIds.png"):
                output_files.append(os.path.join(cur_dir, dir_files[j]))

    if len(input_files) != len(output_files):
        print("Failed to create with input: " + input_loc + "and output: " + output_loc)
        print("Num input files: " + str(len(input_files)))
        print("Num output files: " + str(len(output_files)))
        print("Throwing a bad assert error now")
        assert len(input_files) == len(output_files)

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    # now its time to load the images
    num_images = len(input_files)
    num_records = int(num_images / max_per_record) + 1
    tf_record_names = [save_loc + f'data{idx}' for idx in range(num_records)]
    record_index = 0

    # loop over each tf record, i=tf record
    for i in range(num_records):
        with tf.io.TFRecordWriter(str(tf_record_names[i])) as writer:
            for j in range(record_index, record_index + max_per_record):
                # load the input image
                in_img = cv2.imread(input_files[j])
                # the labels are stored identically across all 3 channels on these images
                # we can use sparse catigorical entropy
                out_img = cv2.imread(output_files[j])[:, :, 0]
                in_bytes = in_img.tostring()
                out_bytes = out_img.tostring()
                data = {
                    'input': _bytes_feature(in_bytes),
                    'output': _bytes_feature(out_bytes)
                }
                feature = tf.train.Feature(feature=data)
                example = tf.train.Example(feature=feature)
                serialized = example.SerializeToString()
                writer.write(serialized)
        # this gets us the correct index and we only have to think in terms of i and j
        record_index += 4500

def load_tf_records(records_path,
                    image_res,
                    batch_size,
                    shuffle_size=5):
    '''
    This is the function for loading the tfrecords into a tf datasest
    :param records_path: the path to the directory containing the needed TF record files.
    :param image_res: the resolution that we are resizing the images to.
    :param batch_size: the batch size that we are using for this dataset
    :param shuffle_size: the amount that we shuffle through the dataset, the higher the better
    :return: TFRecordDataSet with standardized inputs and outputs for catigorical sparse entropy loss
    '''
    def parser(serialized):
        '''
        This function takes the serialized data and turns it into the proper inputs/outputs
        :param serialized:
        :return:
        '''
        features = {
            'input': tf.io.FixedLenFeature([], tf.string),
            'output': tf.io.FixedLenFeature([], tf.string)
        }
        parsed = tf.io.parse_single_example(serialized=serialized, features=features)
        x_samp_bytes = parsed['input']
        x_raw = tf.io.decode_raw(x_samp_bytes, np.uint8)
        # next we standardize and cast the x to full res
        x_full_res = tf.cast(tf.reshape(x_raw, (1024, 2048, 3)), tf.float32)/255.0
        x_done = tf.image.resize(x_full_res, image_res)
        y_sample = parsed['input']
        y_raw = tf.io.decode_raw(y_sample, np.uint8)
        y_full_res = tf.cast(tf.reshape(y_raw, (1024, 2048)), tf.float32)
        y_done = tf.image.resize(y_full_res, image_res)
        return x_done, y_done
    filenames_list = os.listdir(records_path)
    filename_strings = [os.path.join(records_path, filename) for filename in filenames_list]
    dataset = tf.data.TFRecordDataset(filenames=filename_strings,
                                      compression_type=None,
                                      buffer_size=None,
                                      num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(map_func=parser,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=batch_size * shuffle_size)
    dataset = dataset.batch(batch_size=batch_size).repeat()
    return dataset


def main():
    # the constants that we use, and need to go with from the start
    batch_size = 64
    # (Width, Height)
    input_image_res = (1024, 2048)
    data_root = os.path.join("/opt", "data")
    record_root = os.path.join(data_root, "CityScapes")
    train_model = True
    make_records = True
    coarse_train = True
    fine_train = False
    use_test = False
    base_model_name = "model.h5"
    force_model_name = "FinalModel.h5"

    if make_records:
        raw_data_root = os.path.join(data_root, "FinalProject", "CityScapes")
        # the testing takes place on the finely labeled images
        test_in_loc = os.path.join(raw_data_root, "leftimg8bit", "test")
        test_out_loc = os.path.join(raw_data_root, "gtFine", "test")
        # the first round of training takes place on the coarse images
        first_train_in_loc = os.path.join(raw_data_root, "leftimg8bit", "train_extra")
        first_train_out_loc = os.path.join(raw_data_root, "gtCoarse", "train_extra")
        first_val_in_loc = os.path.join(raw_data_root, "leftimg8bit", "val")
        first_val_out_loc = os.path.join(raw_data_root, "gtCoarse", "val")
        second_train_in_loc = os.path.join(raw_data_root, "leftimg8bit", "train")
        second_train_out_loc = os.path.join(raw_data_root, "gtFine", "train")
        second_val_in_loc = os.path.join(raw_data_root, "leftimg8bit", "val")
        second_val_out_loc = os.path.join(raw_data_root, "gtFine", "val")
        # test data
        make_tf_records(input_loc=test_in_loc,
                        output_loc=test_out_loc,
                        save_loc=os.path.join(record_root, "test"))
        # train 1 (gt coarse)
        make_tf_records(input_loc=first_train_in_loc,
                        output_loc=first_train_out_loc,
                        save_loc=os.path.join(record_root, "train1"))
        # val 1 (gt coarse)
        make_tf_records(input_loc=first_val_in_loc,
                        output_loc=first_val_out_loc,
                        save_loc=os.path.join(record_root, "val1"))
        # train 2 (gt fine)
        make_tf_records(input_loc=second_train_in_loc,
                        output_loc=second_train_out_loc,
                        save_loc=os.path.join(record_root, "train2"))
        # val 2 (gt fine)
        make_tf_records(input_loc=second_val_in_loc,
                        output_loc=second_val_out_loc,
                        save_loc=os.path.join(record_root, "val2"))

    if coarse_train and train_model:
        # now its time to assemble the data from 1st train with the gt coarse labels for training and validation data
        train_ds = load_tf_records(records_path=os.path.join(record_root, 'train1'),
                                     image_res=input_image_res,
                                     batch_size=batch_size)
        val_ds = load_tf_records(records_path=os.path.join(record_root, 'val1'),
                                   image_res=input_image_res,
                                   batch_size=batch_size)
        # all testing will be done with the test dataset a.k.a val dataset for these trains
        test_ds = val_ds
        # model construction here

        model.save("coarse_" + base_model_name)
    if fine_train and train_model:
        train_ds = load_tf_records(records_path=os.path.join(record_root, 'train2'),
                                   image_res=input_image_res,
                                   batch_size=batch_size)
        val_ds = load_tf_records(records_path=os.path.join(record_root, 'val2'),
                                 image_res=input_image_res,
                                 batch_size=batch_size)
        if use_test:
            test_ds = load_tf_records(records_path=os.path.join(record_root, "test"),
                                      image_res=input_image_res,
                                      batch_size=batch_size)
            # extra training code here



            model.save("coarse_" + base_model_name)
        else:
            test_ds = val_ds
    if not train_model:
        model = Model.load_weights(force_model_name)
        if use_test:
            test_ds = load_tf_records(records_path=os.path.join(record_root, "test"),
                                      image_res=input_image_res,
                                      batch_size=batch_size)
            model.save("coarse_" + base_model_name)
        else:
            test_ds = val_ds






if __name__ == "__main__":
    main()
