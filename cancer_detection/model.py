import pandas as pd
from glob import glob
import sys
from sklearn.model_selection import train_test_split

from keras.layers import GlobalAveragePooling2D, Flatten
from keras.layers import Dense, Input, Dropout, Concatenate, GlobalMaxPooling2D
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.applications.nasnet import NASNetMobile
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from data_utils import DataGenerator

DATA_PATH = '../input/'
TRAIN_DIR = DATA_PATH + 'train/'
TEST_DIR = DATA_PATH + 'test/'

CROP_SIZE = 96
ORIGINAL_SIZE = 96


def auc(y_true, y_pred):
    """Calculate the AUC-ROC score for predictions

    Arguments:
        y_true {tf.Tensor} -- true labels of samples
        y_pred {tf.Tensor} -- predicted labels of samples

    Returns:
        tf.Tensor -- scalar tf.Tensor with the AUC-ROC score
    """
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def map_id_label(df_train):
    """creates a mapping between the ID of an image and its label

    Arguments:
        df_train {pd.DataFrame} -- DataFrame whose rows are the id and the
                                   label for each image

    Returns:
        dict -- mapping between images IDs and labels
    """

    id_label_map = {k: v for k, v in zip(df_train.id.values,
                                         df_train.label.values)}

    return id_label_map


def extract_id(img_path):
    """Extract image ID from filename

    Arguments:
        img_path {str} -- path to image file

    Returns:
        str -- image id
    """

    img_id = img_path.split('/')[-1].split('.')[0]

    return img_id


def get_model_classif_nasnet():
    """Creates model architecture

    Returns:
        keras.model.Model -- compiled model
    """
    inputs = Input((CROP_SIZE, CROP_SIZE, 3))
    base_model = NASNetMobile(include_top=False,
                              input_tensor=inputs, weights='imagenet')
    x = base_model(inputs)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(1, activation="sigmoid", name="3_")(out)
    model = Model(inputs, out)
    model.compile(optimizer=Adam(0.0001),
                  loss=binary_crossentropy, metrics=['acc', auc])
    model.summary()

    return model


def generate_prediction(model, test_gen, test_files):
    """Generates predictions for submissions

    Arguments:
        model {keras.models.Model} -- trained model
        test_gen {DataGenerator} -- data generator with test set
        test_files {list} -- image filenames for test set

    Returns:
        pd.DataFrame -- DataFrame with image ID and predicted labels
    """

    predictions = model.predict_generator(test_gen, verbose=1)
    df_preds = pd.DataFrame(predictions,
                            columns=['has_tumor_tissue'])

    df_preds['file_names'] = test_files
    df_preds['id'] = df_preds['file_names'].str.split('/')\
                                           .str.get(-1)\
                                           .str.split('.').str.get(0)

    submission_df = pd.DataFrame({'id': df_preds['id'],
                                  'label': df_preds['has_tumor_tissue'],
                                  }).set_index('id')
    return submission_df


def train_model(submission_filename):
    """Entire pipeline to train and generate predictions

    Arguments:
        submission_filename {str} -- filename and path for submission file
    """

    labeled_files = glob('../input/train/*.tif')
    test_files = glob('../input/test/*.tif')

    partition = {}
    train, val = train_test_split(labeled_files, test_size=0.1,
                                  random_state=42)

    partition['train'] = list(map(extract_id, train))
    partition['val'] = list(map(extract_id, val))
    partition['test'] = list(map(extract_id, test_files))

    df_train = pd.read_csv("../input/train_labels.csv")
    labels = map_id_label(df_train)

    print('Creating Generators')
    train_gen = DataGenerator(partition['train'], labels, TRAIN_DIR,
                              dim=(CROP_SIZE, CROP_SIZE),
                              n_channels=3, n_classes=1, shuffle=True)
    val_gen = DataGenerator(partition['val'], labels, TRAIN_DIR,
                            dim=(CROP_SIZE,  CROP_SIZE),
                            n_channels=3, n_classes=1, shuffle=True)
    datagen = ImageDataGenerator(rescale=1.0/255)
    test_gen = datagen.flow_from_directory(TEST_DIR,
                                           target_size=(CROP_SIZE, CROP_SIZE),
                                           batch_size=512,
                                           class_mode='categorical',
                                           shuffle=False)

    model = get_model_classif_nasnet()
    model_filepath = 'model.h5'
    checkpoint = ModelCheckpoint(model_filepath, monitor='val_acc',
                                 save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2,
                                  mode='max', min_lr=0.00001)
    print('Fitting Model')
    history = model.fit_generator(generator=train_gen,
                                  validation_data=val_gen,
                                  epochs=2,
                                  callbacks=[checkpoint, reduce_lr],
                                  use_multiprocessing=True,
                                  workers=2
                                  )

    model.load_weights('model.h5')

    print('Creating Submission Predictions')
    submission = generate_prediction(model, test_gen, test_files)

    submission.to_csv('submission_files/' + submission_filename)


if __name__ == '__main__':
    submission_name = sys.argv[1]
    train_model(submission_name)
