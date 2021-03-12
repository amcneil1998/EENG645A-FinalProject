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


def make_tf_records(input_loc,
                    output_loc,
                    record_loc,
                    res,
                    batch_size,
                    max_per_record=4500):
    '''
    :param input_loc: os.path to the input data to be used for this record set
    :param output_loc: os.path to the output data to be used for this record set
    :param record_loc: os.path to the location that the tf record will be stored, will create if it isn't there
    :param res: the resolution of the images that we want the network to process.
    :param batch_size: the batch size of the network.
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

    if not os.path.exists(output_loc):
        os.makedirs(output_loc)
    # now its time to load the images
    num_images = len(input_files)
    num_records = int(num_images / max_per_record) + 1
    tf_record_names = [output_loc + f'data{idx}' for idx in range(num_records)]
    record_index = 0

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # loop over each tf record, i=tf record
    for i in range(num_records):
        with tf.io.TFRecordWriter(str(tf_record_names[i])) as writer:
            for j in range(record_index, record_index + max_per_record):
                # load the input image
                in_img = cv2.loadimg(input_files[j])
                out_img = cv2.loadimg(output_files[j])
                in_bytes = in_img.tostring()
                # TODO think if we should one hot here or not here
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



def main():
    # the constants that we use, and need to go with from the start
    batch_size = 64
    # (Width, Height)
    input_image_res = (1024, 2048)
    data_root = os.path.join("/opt", "data")
    record_root = os.path.join(data_root, "CityScapes")
    make_records = True
    if make_records:
        raw_data_root = os.path.join(data_root, "FinalProject", "CityScapes")
        # the testing takes place on the finely labeled images
        test_out_loc = os.path.join(raw_data_root, "gtFine", "test")
        test_in_loc = os.path.join(raw_data_root, "leftimg8bit", "test")
        # the first round of training takes place on the coarse images
        first_train_in_loc = os.path.join(raw_data_root, "leftimg8bit", "train_extra")
        first_train_out_loc = os.path.join(raw_data_root, "gtCoarse", "train_extra")
        first_val_in_loc = os.path.join(raw_data_root, "leftimg8bit", "val")
        first_val_out_loc = os.path.join(raw_data_root, "gtCoarse", "val")
        second_train_in_loc = os.path.join(raw_data_root, "leftimg8bit", "train")
        second_train_out_loc = os.path.join(raw_data_root, "gtFine", "train")
        second_val_in_loc = os.path.join(raw_data_root, "leftimg8bit", "val")
        second_val_out_loc = os.path.join(raw_data_root, "gtFine", "val")
        make_tf_records(test_in_loc,
                        test_out_loc,
                        os.path.join(record_root, "test"),
                        input_image_res,
                        batch_size)
        make_tf_records(first_train_in_loc,
                        first_train_out_loc,
                        os.path.join(record_root, "train1"),
                        input_image_res,
                        batch_size)
        make_tf_records(first_val_in_loc,
                        first_val_out_loc,
                        os.path.join(record_root, "val1"),
                        input_image_res,
                        batch_size)
        make_tf_records(second_train_in_loc,
                        second_train_out_loc,
                        os.path.join(record_root, "train2"),
                        input_image_res,
                        batch_size)
        make_tf_records(second_val_in_loc,
                        second_val_out_loc,
                        os.path.join(record_root, "val2"),
                        input_image_res,
                        batch_size)




if __name__ == "__main__":
    main()
