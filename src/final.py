import numpy as np
import itertools
import os
import tensorflow as tf
import cv2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Concatenate, Conv2DTranspose
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
import sklearn.metrics

# my plot history function from last lab
def plot_history(hist):
    '''
    Plots Accuracy and Loss for Model training
    :param hist:
    :return:
    '''
    plt.figure()
    plt.plot(hist.history['loss'], label="Train")
    plt.plot(hist.history['val_loss'], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Losses")
    plt.grid()
    plt.legend()
    plt.figure()
    plt.plot(hist.history['accuracy'], label="Train")
    plt.plot(hist.history['val_accuracy'], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracies")
    plt.grid()
    plt.legend()
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte., thanks Joe"""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_tf_records(input_loc,
                    output_loc,
                    save_loc,
                    evaluate=False,
                    max_per_record=4500):
    '''
    :param input_loc: os.path to the input data to be used for this record set
    :param output_loc: os.path to the output data to be used for this record set
    :param save_loc: os.path to where we are putting the tf records
    :param evaluate: lets you see the images going into the dataset, will not make a dataset
    :param max_per_record:
    :return: None, just writes tf record to disk for future use.
    '''
    # lets kick off with making a list of input and output files
    input_files = []
    sub_dirs = os.listdir(input_loc)
    for i in range(len(sub_dirs)):
        cur_dir = os.path.join(input_loc, sub_dirs[i])
        dir_files = os.listdir(cur_dir)
        dir_files.sort() # <- the first part of the stupid line that cost me 2 days
        for j in range(len(dir_files)):
            # be safe and make sure that we only grab pictures
            if dir_files[j].endswith(".png"):
                input_files.append(os.path.join(cur_dir, dir_files[j]))
    output_files = []
    sub_dirs = os.listdir(output_loc)
    for i in range(len(sub_dirs)):
        cur_dir = os.path.join(output_loc, sub_dirs[i])
        dir_files = os.listdir(cur_dir)
        dir_files.sort() # <- the stupid line that cost me two days
        for j in range(len(dir_files)):
            # we only want the labelIds ones, these take up less space and make it easier to encode
            if dir_files[j].endswith("labelIds.png"):
                output_files.append(os.path.join(cur_dir, dir_files[j]))
    if evaluate:
        print(input_files[0])
        print(output_files[0])
        print(input_files[8953])
        print(output_files[8953])
        print(input_files[-1])
        print(output_files[-1])
        in_img = cv2.imread(input_files[999])
        out_img = cv2.imread(output_files[999])
        img = np.hstack((in_img, out_img))
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()
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
    tf_record_names = [save_loc + f'/data{idx}' for idx in range(num_records)]
    record_index = 0
    print("Starting to process images to: " + save_loc)
    # loop over each tf record, i=tf record
    for i in range(num_records):
        with tf.io.TFRecordWriter(str(tf_record_names[i])) as writer:
            end_point = record_index + max_per_record
            if end_point > len(input_files):
                end_point = len(input_files)
            for j in range(record_index, end_point):
                # load the input image
                in_img = cv2.imread(input_files[j])
                # the labels are stored identically across all 3 channels on these images
                # we can use sparse catigorical entropy
                out_img = cv2.imread(output_files[j])
                in_bytes = in_img.tostring()
                out_bytes = out_img.tostring()
                data = {
                    'input': _bytes_feature(in_bytes),
                    'output': _bytes_feature(out_bytes)
                }
                feature = tf.train.Features(feature=data)
                example = tf.train.Example(features=feature)
                serialized = example.SerializeToString()
                writer.write(serialized)
        # this gets us the correct index and we only have to think in terms of i and j
        record_index += 4500
        print("Processed: " + str(record_index) + " images")

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
        y_full_res = tf.reshape(y_raw, (1024, 2048, 3))
        y_vals = tf.cast(tf.image.resize(y_full_res, image_res)[:, :, 0], tf.int32)
        y_done = tf.one_hot(indices=y_vals,
                            depth=34)
        return x_done, y_done
    filenames_list = os.listdir(records_path)
    filename_strings = [os.path.join(records_path, filename) for filename in filenames_list]
    dataset = tf.data.TFRecordDataset(filenames=filename_strings,
                                      compression_type=None,
                                      buffer_size=None,
                                      num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(map_func=parser,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = dataset.shuffle(buffer_size=batch_size * shuffle_size)S
    dataset = dataset.batch(batch_size=batch_size).repeat()
    return dataset

def main():
    # the constants that we use, and need to go with from the start
    batch_size = 32
    num_classes = 34
    num_epochs = 300
    steps_per_epoch = 16
    num_epochs_fine = 20
    # (Width, Height)
    input_image_res = (192, 384)
    data_root = os.path.join("/opt", "data")
    record_root = os.path.join(data_root, "CityScapes")
    make_records = True
    coarse_train = True
    fine_train = True
    use_test = True
    test_num = 256
    base_model_name = "model.h5"
    force_model_name = "coarse_model.h5"

    if make_records:
        raw_data_root = os.path.join(data_root, "FinalProject", "CityScapes")
        # the testing takes place on the finely labeled images
        test_in_loc = os.path.join(raw_data_root, "leftImg8bit", "test")
        test_out_loc = os.path.join(raw_data_root, "gtFine", "test")
        # the first round of training takes place on the coarse images
        first_train_in_loc = os.path.join(raw_data_root, "leftImg8bit", "train_extra")
        first_train_out_loc = os.path.join(raw_data_root, "gtCoarse", "train_extra")
        first_val_in_loc = os.path.join(raw_data_root, "leftImg8bit", "val")
        first_val_out_loc = os.path.join(raw_data_root, "gtCoarse", "val")
        second_train_in_loc = os.path.join(raw_data_root, "leftImg8bit", "train")
        second_train_out_loc = os.path.join(raw_data_root, "gtFine", "train")
        second_val_in_loc = os.path.join(raw_data_root, "leftImg8bit", "val")
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

    if coarse_train:
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
        in_layer = Input((input_image_res[0], input_image_res[1], 3))
        # Start making U net, all things will be notated as c for convolve, p for MaxPooling, and U for upscale/concat
        # the number notates the layer of the U that it is on
        # we are doing a 5 layer U
        # first layer L
        c1 = Conv2D(36,
                    kernel_size=(3, 3),
                    kernel_initializer='he_normal',
                    padding='same')(in_layer)
        c1 = BatchNormalization()(c1)
        c1 = Conv2D(36,
                    kernel_size=(3, 3),
                    kernel_initializer='he_normal',
                    padding='same')(c1)
        p1 = MaxPooling2D((2, 2), strides=2)(c1)
        # second layer L
        c2 = Conv2D(72,
                    kernel_size=(3, 3),
                    kernel_initializer='he_normal',
                    padding='same')(p1)
        c2 = BatchNormalization()(c2)
        c2 = Conv2D(72,
                    kernel_size=(3, 3),
                    kernel_initializer='he_normal',
                    padding='same')(c2)
        p2 = MaxPooling2D((2, 2), strides=2)(c2)
        # third layer L
        c3 = Conv2D(144,
                    kernel_size=(3, 3),
                    kernel_initializer='he_normal',
                    padding='same')(p2)
        c3 = BatchNormalization()(c3)
        c3 = Conv2D(144,
                    kernel_size=(3, 3),
                    kernel_initializer='he_normal',
                    padding='same')(c3)
        p3 = MaxPooling2D((2, 2), strides=2)(c3)
        # fourth layer L
        c4 = Conv2D(288,
                    kernel_size=(3, 3),
                    kernel_initializer='he_normal',
                    padding='same')(p3)
        c4 = BatchNormalization()(c4)
        c4 = Conv2D(288,
                    kernel_size=(3, 3),
                    kernel_initializer='he_normal',
                    padding='same')(c4)
        p4 = MaxPooling2D((2, 2), strides=2)(c4)
        # bottom layer
        c5 = Conv2D(576,
                    kernel_size=(3, 3),
                    kernel_initializer='he_normal',
                    padding='same')(p4)
        c5 = BatchNormalization()(c5)
        c5 = Conv2D(576,
                    kernel_size=(3, 3),
                    kernel_initializer='he_normal',
                    padding='same')(c5)
        # now start moving up
        # going up involves upscaling below, concat across, then putting through 2 convolutional layers.
        # 4th layer on the return
        u4 = Conv2DTranspose(288,
                             kernel_size=(3, 3),
                             strides=(2, 2),
                             padding='same')(c5)
        u4 = Concatenate()([u4, c4])
        c4 = Conv2D(288,
                    kernel_size=(3, 3),
                    kernel_initializer='he_normal',
                    padding='same')(u4)
        c4 = BatchNormalization()(c4)
        c4 = Conv2D(288,
                    kernel_size=(3, 3),
                    kernel_initializer='he_normal',
                    padding='same')(c4)
        # 3rd layer on the return
        u3 = Conv2DTranspose(144,
                             kernel_size=(3, 3),
                             strides=(2, 2),
                             padding='same')(c4)
        u3 = Concatenate()([u3, c3])
        c3 = Conv2D(144,
                    kernel_size=(3, 3),
                    kernel_initializer='he_normal',
                    padding='same')(u3)
        c3 = BatchNormalization()(c3)
        c3 = Conv2D(144,
                    kernel_size=(3, 3),
                    kernel_initializer='he_normal',
                    padding='same')(c3)
        # 2nd layer on the return
        u2 = Conv2DTranspose(72,
                             kernel_size=(3, 3),
                             strides=(2, 2),
                             padding='same')(c3)
        u2 = Concatenate()([u2, c2])
        c2 = Conv2D(72,
                    kernel_size=(3, 3),
                    kernel_initializer='he_normal',
                    padding='same')(u2)
        c2 = BatchNormalization()(c2)
        c2 = Conv2D(72,
                    kernel_size=(3, 3),
                    kernel_initializer='he_normal',
                    padding='same')(c2)
        # 1st layer on the return
        u1 = Conv2DTranspose(36,
                             kernel_size=(3, 3),
                             strides=(2, 2),
                             padding='same')(c2)
        u1 = Concatenate()([u1, c1])
        c1 = Conv2D(36,
                    kernel_size=(3, 3),
                    kernel_initializer='he_normal',
                    padding='same')(u1)
        c1 = BatchNormalization()(c1)
        c1 = Conv2D(36,
                    kernel_size=(3, 3),
                    kernel_initializer='he_normal',
                    padding='same')(c1)
        out = Conv2D(34,
                     kernel_size=(1, 1),
                     activation='softmax')(c1)

        # End making U net
        model = Model(in_layer, out)
        model.compile(loss=CategoricalCrossentropy(from_logits=True),
                      optimizer=Adam(lr=0.001),
                      metrics=['accuracy'])
        reduce_lr = ReduceLROnPlateau(monitor='loss',
                                      factor=0.85,
                                      patience=6,
                                      verbose=1,
                                      mode='auto',
                                      min_delta=0.01,
                                      cooldown=2,
                                      min_lr=0.0000001)
        model.summary()
        hist = model.fit(train_ds,
                         steps_per_epoch=steps_per_epoch,
                         batch_size=batch_size,
                         epochs=num_epochs,
                         validation_data=val_ds,
                         validation_steps=1,
                         use_multiprocessing=True,
                         workers=8,
                         callbacks=[reduce_lr])
        model.save("coarse_" + base_model_name)
        plot_history(hist)
        if not fine_train and not use_test:
            test_ds = val_ds
    # now that we have a model trained on the rough images, lets do the smaller ds of the finely labeled images
    if fine_train:
        # check if we are only doing a fine train of a pre defined coarse trained model
        if not coarse_train:
            model = load_model("coarse_" + base_model_name)
        train_ds = load_tf_records(records_path=os.path.join(record_root, 'train2'),
                                   image_res=input_image_res,
                                   batch_size=batch_size)
        val_ds = load_tf_records(records_path=os.path.join(record_root, 'val2'),
                                 image_res=input_image_res,
                                 batch_size=batch_size)
        # build the model now with a much lower learning rate
        model.compile(loss=CategoricalCrossentropy(from_logits=True),
                      optimizer=Adam(lr=0.0001),
                      metrics=['accuracy'])
        reduce_lr = ReduceLROnPlateau(monitor='loss',
                                      factor=0.75,
                                      patience=6,
                                      verbose=1,
                                      mode='auto',
                                      min_delta=0.01,
                                      cooldown=2,
                                      min_lr=0.000001)
        hist = model.fit(train_ds,
                         initial_epoch=num_epochs,
                         steps_per_epoch=steps_per_epoch,
                         batch_size=batch_size,
                         epochs=num_epochs_fine + num_epochs,
                         validation_data=val_ds,
                         validation_steps=1,
                         use_multiprocessing=True,
                         workers=8,
                         callbacks=[reduce_lr])
        model.save("fine_" + base_model_name)
        plot_history(hist)
        if use_test:
            test_ds = load_tf_records(records_path=os.path.join(record_root, "test"),
                                      image_res=input_image_res,
                                      batch_size=batch_size)
        else:
            test_ds = val_ds
    # we are just evaluating the model that we have trained
    if not fine_train and not coarse_train:
        model = load_model(force_model_name)
        if use_test:
            test_ds = load_tf_records(records_path=os.path.join(record_root, "test"),
                                      image_res=input_image_res,
                                      batch_size=batch_size)
        else:
            test_ds = load_tf_records(records_path=os.path.join(record_root, "val2"),
                                      image_res=input_image_res,
                                      batch_size=batch_size)

    # time to evaluate the model on our test data
    x_visualize = []
    y_visualize = []
    total_samps = 0
    for img_batch, label_batch in test_ds.as_numpy_iterator():
        total_samps += img_batch.shape[0]
        x_visualize.append(img_batch)
        y_visualize.append(label_batch)
        if total_samps > test_num:
            break
    x_eval = np.vstack(x_visualize)
    y_eval = np.vstack(y_visualize)
    y_pred = model.predict(x_eval, batch_size=1)
    cv2.destroyAllWindows()
    y_eval = np.argmax(y_eval, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    ex = np.hstack((x_eval[0],
                    np.stack((y_eval[0], y_eval[0], y_eval[0]), axis=-1)/35.0,
                    np.stack((y_pred[0], y_pred[0], y_pred[0]), axis=-1)/35.0))
    cv2.imshow('image', ex)
    cv2.waitKey(0)
    y_eval = y_eval.reshape(y_eval.shape[0] * y_eval.shape[1] * y_eval.shape[2])
    y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1] * y_pred.shape[2])
    names = ['unlabeled', 'ego_vehicle', 'rectification_border', 'out_of_roi', 'static', 'dynamic', 'ground',
             'road', 'sidewalk', 'parking', 'rail_track', 'building', 'wall', 'fence', 'guard_rail', 'bridge',
             'tunnel', 'pole', 'polegroup', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky',
             'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']
    print(sklearn.metrics.classification_report(y_eval, y_pred, target_names=names))
    confusion_matrix = sklearn.metrics.confusion_matrix(y_pred=y_pred,
                                                        y_true=y_eval)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=names,
                          title='Confusion matrix, without normalization')
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=names, normalize=True,
                          title='Normalized confusion matrix')
    plt.show()



if __name__ == "__main__":
    main()
