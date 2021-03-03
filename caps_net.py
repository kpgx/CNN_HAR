# Compatibility layer between Python 2 and Python 3
from __future__ import print_function

import sqlite3

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
# from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils
#
# from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
# from keras import layers, models, optimizers
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as keras
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask



def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma


def show_confusion_matrix(validations, predictions):
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=activity_map.values(),
                yticklabels=activity_map.values(),
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def show_basic_dataframe_info(dataframe, preview_rows=20):
    """
    This function shows basic information for the given dataframe
    Args:
        dataframe: A Pandas DataFrame expected to contain data
        preview_rows: An integer value of how many rows to preview
    Returns:
        Nothing
    """

    # Shape and how many rows and columns
    print("Number of columns in the dataframe: %i" % (dataframe.shape[1]))
    print("Number of rows in the dataframe: %i\n" % (dataframe.shape[0]))
    print("First 20 rows of the dataframe:\n")
    # Show first 20 rows
    print(dataframe.head(preview_rows))
    print("\nDescription of dataframe:\n")
    # Describe dataset like mean, min, max, etc.
    # print(dataframe.describe()


def read_data(file_path):
    """
    This function reads the accelerometer data from a file
    Args:
        file_path: URL pointing to the CSV file
    Returns:
        A pandas dataframe
    """

    column_names = ['user-id',
                    'activity',
                    'timestamp',
                    'x-axis',
                    'y-axis',
                    'z-axis']
    df = pd.read_csv(file_path,
                     header=None,
                     names=column_names)
    # Last column has a ";" character which must be removed ...
    df['z-axis'].replace(regex=True,
      inplace=True,
      to_replace=r';',
      value=r'')
    # ... and then this column must be transformed to float explicitly
    df['z-axis'] = df['z-axis'].apply(convert_to_float)
    # This is very important otherwise the model will not fit and loss
    # will show up as NAN
    df.dropna(axis=0, how='any', inplace=True)

    return df


def read_data_new(file_path, activity_list, table):
    """
    This function reads the accelerometer data from a db
    Args:
        file_path: URL pointing to the DB file
    Returns:
        A pandas dataframe
    """

    conn = sqlite3.connect(file_path)
    activity_list_str=",".join(['"'+x+'"' for x in activity_list])
    query = '''select * from %s where "activity_label" in (%s) and "user_id" < 1651'''%(table, activity_list_str)
    print(query)
    df = pd.read_sql(query, conn)
    # Last column has a ";" character which must be removed ...
    df['z'].replace(regex=True,
      inplace=True,
      to_replace=r';',
      value=r'')
    # ... and then this column must be transformed to float explicitly
    df['z'] = df['z'].apply(convert_to_float)
    # This is very important otherwise the model will not fit and loss
    # will show up as NAN
    df.dropna(axis=0, how='any', inplace=True)
    df.rename(inplace=True, columns={"user_id":"user-id",
                       "activity_label":"activity",
                       "x":"x-axis",
                       "y":"y-axis",
                       "z":"z-axis"})

    return df


def convert_to_float(x):
    try:
        return np.float(x)
    except:
        return np.nan


def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,
         figsize=(15, 10),
         sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


def create_segments_and_labels(df, time_steps, step, label_name):
    """
    This function receives a dataframe and returns the reshaped segments
    of x,y,z acceleration as well as the corresponding labels
    Args:
        df: Dataframe in the expected format
        time_steps: Integer value of the length of a segment that is created
    Returns:
        reshaped_segments
        labels:
    """

    # x, y, z acceleration as features
    N_FEATURES = 3
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['x-axis'].values[i: i + time_steps]
        ys = df['y-axis'].values[i: i + time_steps]
        zs = df['z-axis'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df[label_name][i: i + time_steps])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels


def CapsNet(input_shape, n_class, routings, batch_size):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv1D(filters=256, kernel_size=12, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=5, n_channels=20, kernel_size=12, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=10, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    dec_3 = decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid', input_dim=10 * n_class))
    #    dec_1 = decoder.add(layers.Dense(16, activation='relu', input_dim=10*n_class))
    #    dec_2 = decoder.add(layers.Dense(16, activation='relu'))
    #    decoder.add(layers.Dropout(0.40))
    #    dec_3 = decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    dec_4 = decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked)])  # masked_by_y
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 10))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * keras.square(keras.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * keras.square(keras.maximum(0., y_pred - 0.1))

    return keras.mean(keras.sum(L, 1))


def train(model, X_train, y_train, X_test, y_test, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr, amsgrad=True),
                  loss=[margin_loss, 'categorical_crossentropy'],
                  loss_weights=[1, args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    #    path = r'C:\Users\Karush\.spyder-py3\CapsNet_Weights5'
    #    for j in range(1,11):
    #        model.load_weights(path+r"\weights-improvement-"+str(j)+".hdf5")
    #        filters = model.layers[8].get_weights()[0]
    #        np.shape(filters)
    #        filters = np.reshape(filters, (int(200*4),int(9324/4)))
    #        plt.figure()
    #        plt.imshow(filters)
    #        plt.savefig(r'C:\Users\Karush\.spyder-py3\CapsNet_Activations5\c'+str(j)+'.png')

    # model.load_weights('new_capsnet20.h5')
    # Training without data augmentation:
    hist = model.fit([X_train, y_train], [y_train, X_train], batch_size=args.batch_size, epochs=args.epochs,
                     shuffle=True)

    model.save_weights('new_capsnet30.h5')
    ## Saving model to JSON
    #    model_json = model.to_json()
    #    with open("5capsnet_50.json", "w") as json_file:
    #        json_file.write(model_json)
    #
    #    ## Saving model data to JSON
    #    with open('5capsnet_50.json', 'w') as f:
    #        json.dump(hist.history, f)
    #
    #    model.save_weights('5capsnet_50.h5')

    return model


def test(model, X_test, y_test, args):
    y_pred, x_recon = model.predict(X_test, batch_size=20)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])
    return y_pred
# ------- THE PROGRAM TO LOAD DATA AND TRAIN THE MODEL -------

# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format
sns.set() # Default seaborn look and feel
plt.style.use('ggplot')
# print('keras version ', keras.__version__)

# LABELS = ["Downstairs",
#           "Jogging",
#           "Sitting",
#           "Standing",
#           "Upstairs",
#           "Walking"]
activity_map = {
    "A": "walking",
    "B": "jogging",
    "C": "stairs",
    # "D": "sitting",
    # "E": "standing",
    # "F": "typing",
    # "G": "teeth",
    # "H": "soup",
    # "I": "chips",
    # "J": "pasta",
    # "K": "drinking",
    # "L": "sandwich",
    # "M": "kicking",
    # "O": "catch",
    # "P": "dribbling",
    # "Q": "writing",
    # "R": "clapping",
    # "S": "folding"
}

LABELS = activity_map.keys()

# The number of steps within one time segment
TIME_PERIODS = 80
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 40

print("\n--- Load, inspect and transform data ---\n")

# Load data set containing all the data from csv
WATCH_TABLE_NAME = 'watch_accl'
watch_df = read_data_new('/Users/kasun/projects/activity_capsule/activity.db', activity_map.keys(), WATCH_TABLE_NAME)
PHONE_TABLE_NAME = 'phone_accl'
phone_df = read_data_new('/Users/kasun/projects/activity_capsule/activity.db', activity_map.keys(), PHONE_TABLE_NAME)

# Describe the data
# show_basic_dataframe_info(df, 20)

# df['activity'].value_counts().plot(kind='bar',
#                                    title='Training Examples by Activity Type')
# plt.show()

# df['user-id'].value_counts().plot(kind='bar',
#                                   title='Training Examples by User')
# plt.show()

# plot the accl signals
# for activity in np.unique(df["activity"]):
#     subset = df[df["activity"] == activity][:180]
#     plot_activity(activity, subset)

# Define column name of the label vector
LABEL = "ActivityEncoded"
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
watch_df[LABEL] = le.fit_transform(watch_df["activity"].values.ravel())
phone_df[LABEL] = le.fit_transform(phone_df["activity"].values.ravel())

# %%

print("\n--- Reshape the data into segments ---\n")

# Differentiate between test set and training set
df_test = phone_df[phone_df['user-id'] <= 1610]
df_train = watch_df[watch_df['user-id'] > 1610]

# Normalize features for training data set
df_train['x-axis'] = feature_normalize(watch_df['x-axis'])
df_train['y-axis'] = feature_normalize(watch_df['y-axis'])
df_train['z-axis'] = feature_normalize(watch_df['z-axis'])
# Round in order to comply to NSNumber from iOS
df_train = df_train.round({'x-axis': 6, 'y-axis': 6, 'z-axis': 6})

# Reshape the training data into segments
# so that they can be processed by the network
x_train, y_train = create_segments_and_labels(df_train,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)

# %%

# Inspect x data
print('x_train shape: ', x_train.shape)
# Displays (20869, 40, 3)
print(x_train.shape[0], 'training samples')
# Displays 20869 train samples

# Inspect y data
print('y_train shape: ', y_train.shape)
# Displays (20869,)

# Set input & output dimensions
num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_classes = le.classes_.size
print(list(le.classes_))

# Set input_shape / reshape for Keras
# Remark: acceleration data is concatenated in one array in order to feed
# it properly into coreml later, the preferred matrix of shape [80,3]
# cannot be read in with the current version of coreml (see also reshape
# layer as the first layer in the keras model)
input_shape = x_train[1,:,:].shape
# x_train = x_train.reshape(x_train.shape[0], input_shape)

print('x_train shape:', x_train.shape)
# x_train shape: (20869, 120)
print('input_shape:', input_shape)
# input_shape: (120)

# Convert type for Keras otherwise Keras cannot process the data
x_train = x_train.astype("float32")
y_train = y_train.astype("float32")

# %%

# One-hot encoding of y_train labels (only execute once!)
y_train = np_utils.to_categorical(y_train, num_classes)
print('New y_train shape: ', y_train.shape)
# (4173, 6)

# %%

print("\n--- Create capsule network model ---\n")

model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape,
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=3,
                                                  batch_size=100)
model.summary()

# 1D Capsule neural network
# model_m = Sequential()
# model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
# model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
# model_m.add(PrimaryCap(dim_capsule=5, n_channels=20, kernel_size=12, strides=2, padding='valid'))
# model_m.add(CapsuleLayer(num_capsule=n_class, dim_capsule=10, num_routing=routings,
#                              name='digitcaps'))
# model_m.add(Conv1D(160, 10, activation='relu'))
# model_m.add(Conv1D(160, 10, activation='relu'))
# model_m.add(GlobalAveragePooling1D())
# model_m.add(Dropout(0.5))
# model_m.add(Dense(num_classes, activation='softmax'))
print("\n--- Train model summary ---\n")
print(eval_model.summary())

print("\n--- Eval model summary ---\n")
print(eval_model.summary())
# Accuracy on training data: 99%
# Accuracy on test data: 91%

# %%

print("\n--- Fit the model ---\n")

# The EarlyStopping callback monitors training accuracy:
# if it fails to improve for two consecutive epochs,
# training stops early
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
]

eval_model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

# Hyper-parameters
BATCH_SIZE = 400
EPOCHS = 50

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history = eval_model.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)

# %%

print("\n--- Learning curve of model training ---\n")

# summarize history for accuracy and loss
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], "g--", label="Accuracy of training data")
plt.plot(history.history['val_accuracy'], "g", label="Accuracy of validation data")
plt.plot(history.history['loss'], "r--", label="Loss of training data")
plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

#%%

print("\n--- Check against test data ---\n")

# Normalize features for testing data set
df_test['x-axis'] = feature_normalize(df_test['x-axis'])
df_test['y-axis'] = feature_normalize(df_test['y-axis'])
df_test['z-axis'] = feature_normalize(df_test['z-axis'])

df_test = df_test.round({'x-axis': 6, 'y-axis': 6, 'z-axis': 6})

x_test, y_test = create_segments_and_labels(df_test,
                                            TIME_PERIODS,
                                            STEP_DISTANCE,
                                            LABEL)

# Set input_shape / reshape for Keras
# x_test = x_test.reshape(x_test.shape[0], input_shape)

x_test = x_test.astype("float32")
y_test = y_test.astype("float32")

y_test = np_utils.to_categorical(y_test, num_classes)

score = eval_model.evaluate(x_test, y_test, verbose=1)

print("\nAccuracy on test data: %0.2f" % score[1])
print("\nLoss on test data: %0.2f" % score[0])

# %%

print("\n--- Confusion matrix for test data ---\n")

y_pred_test = eval_model.predict(x_test)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

show_confusion_matrix(max_y_test, max_y_pred_test)

# %%

print("\n--- Classification report for test data ---\n")

print(classification_report(max_y_test, max_y_pred_test))