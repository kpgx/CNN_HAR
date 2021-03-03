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

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM
from keras.utils import np_utils


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma


def show_confusion_matrix(validations, predictions):
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(10, 8))
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


def read_data_pamap(file_path, activity_list, column_list, limit=1000000000):
    """
    This function reads the accelerometer data from a db
    Args:
        file_path: URL pointing to the DB file
    Returns:
        A pandas dataframe
    """

    conn = sqlite3.connect(file_path)
    # column_list_str = ", ".join(['"' + x +'"' for x in column_list])
    training_records_list=[]
    testing_records_list=[]

    for activity_id in activity_list:
        query = '''select %s from main where "activity_id"=%d and "sub_id" = 1 order by "index" limit %d'''%(column_list, activity_id, limit)
        print(query)
        df = pd.read_sql(query, conn)
        mark_80 = int(len(df) * .8)
        training_records = df.iloc[:mark_80, :]
        testing_records = df.iloc[mark_80:, :]
        training_records_list.append(training_records)
        testing_records_list.append(testing_records)


    training_return_df = pd.concat(training_records_list)
    testing_return_df = pd.concat(testing_records_list)

    # This is very important otherwise the model will not fit and loss
    # will show up as NAN
    training_return_df.dropna(axis=0, how='any', inplace=True)
    testing_return_df.dropna(axis=0, how='any', inplace=True)

    return training_return_df, testing_return_df


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


def create_segments_and_labels(df, m_feature_list, time_steps, step, label_name):
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
    N_FEATURES = 9
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        segment = []
        for feature in m_feature_list:
            segment.append(df[feature].values[i: i + time_steps])
            # xs = df['x-axis'].values[i: i + time_steps]
            # ys = df['y-axis'].values[i: i + time_steps]
            # zs = df['z-axis'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df[label_name][i: i + time_steps])[0][0]
        segments.append(segment)
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels


# ------- THE PROGRAM TO LOAD DATA AND TRAIN THE MODEL -------

# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format
sns.set() # Default seaborn look and feel
plt.style.use('ggplot')
print('keras version ', keras.__version__)

# LABELS = ["Downstairs",
#           "Jogging",
#           "Sitting",
#           "Standing",
#           "Upstairs",
#           "Walking"]
activity_map = {
    # 0: 'transient',
    1:'lying',
    2:'sitting',
    3:'standing',
    4:'walking',
    5:'running',
    6:'cycling',
    7:'Nordic walking',
    9:'watching TV',
    # 10:'computer work',
    11:'car driving',
    12:'ascending stairs',
    13:'descending stairs',
    # 16:'vacuum cleaning',
    17:'ironing',
    18:'folding laundry',
    19:'house cleaning',
    # 20:'playing soccer',
    24:'rope jumping'
}

LABELS = activity_map.keys()

# The number of steps within one time segment
TIME_PERIODS = 200#80
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 100#40

print("\n--- Load, inspect and transform data ---\n")

# Load data set containing all the data from csv
db_path = '/Users/kasun/projects/activity_capsule/data/pmap2.db'

feature_list_hand = ["acc_16_01_hand",
                "acc_16_02_hand",
                "acc_16_03_hand",
                "gyr_01_hand",
                "gyr_02_hand",
                "gyr_03_hand",
                "mag_01_hand",
                "mag_02_hand",
                "mag_03_hand",
                     ]

feature_list_chest = ["acc_16_01_chest",
                "acc_16_02_chest",
                "acc_16_03_chest",
                "gyr_01_chest",
                "gyr_02_chest",
                "gyr_03_chest",
                "mag_01_chest",
                "mag_02_chest",
                "mag_03_chest",
                      ]

feature_list_train = feature_list_hand
feature_list_chest = feature_list_hand


df_hand_train, df_hand_test = read_data_pamap(db_path, activity_map.keys(),"sub_id, activity_id," +", ".join(feature_list_hand), 10899)
df_chest_train, df_chest_test = read_data_pamap(db_path,activity_map.keys(),"sub_id, activity_id,"+", ".join(feature_list_chest), 10899)

df_train = df_chest_train
df_test = df_chest_test
# df_test = df_test2

# df_train.rename(inplace=True, columns={"sub_id": "user-id",
#                                                 "activity_id": "activity",
#                                                 "acc_16_01_hand": "x-axis",
#                                                 "acc_16_02_hand": "y-axis",
#                                                 "acc_16_03_hand": "z-axis"})
# df_test.rename(inplace=True, columns={"sub_id": "user-id",
#                                                 "activity_id": "activity",
#                                                 "acc_16_01_hand": "x-axis",
#                                                 "acc_16_02_hand": "y-axis",
#                                                 "acc_16_03_hand": "z-axis"})
# ankle_df = read_data_pamap(db_path,[4,5,12],"sub_id, activity_id, acc_16_01_ankle, acc_16_02_ankle, acc_16_03_ankle", 85000)

# chest_query = '''select %s from main where "activity_id" in (4,5,12)'''%("sub_id, activity_id, acc_16_01_chest, acc_16_02_chest, acc_16_03_chest")
# print(chest_query)
# chest_df = pd.read_sql(chest_query, PMAP_DB)
# # This is very important otherwise the model will not fit and loss
# # will show up as NAN
# chest_df.dropna(axis=0, how='any', inplace=True)
# chest_df.rename(inplace=True, columns={"sub_id":"user-id",
#                        "activity_id":"activity",
#                        "acc_16_01_chest":"x-axis",
#                        "acc_16_02_chest":"y-axis",
#                        "acc_16_03_chest":"z-axis"})


# ankle_query = '''select %s from main where "activity_id" in (4,5,12)'''%("sub_id, activity_id, acc_16_01_ankle, acc_16_02_ankle, acc_16_03_ankle")
# print(ankle_query)
# ankle_df = pd.read_sql(ankle_query, PMAP_DB)
# # This is very important otherwise the model will not fit and loss
# # will show up as NAN
# ankle_df.dropna(axis=0, how='any', inplace=True)
# ankle_df.rename(inplace=True, columns={"sub_id":"user-id",
#                        "activity_id":"activity",
#                        "acc_16_01_ankle":"x-axis",
#                        "acc_16_02_ankle":"y-axis",
#                        "acc_16_03_ankle":"z-axis"})


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
df_train[LABEL] = le.fit_transform(df_train["activity_id"].values.ravel())
df_test[LABEL] = le.fit_transform(df_test["activity_id"].values.ravel())

# chest_df[LABEL] = le.fit_transform(chest_df["activity"].values.ravel())
# ankle_df[LABEL] = le.fit_transform(ankle_df["activity"].values.ravel())



# %%

# current_df = wrist_df



print("\n--- Reshape the data into segments ---\n")

# Differentiate between test set and training set

#
# df_train['activity_id'].value_counts().plot(kind='bar',
#                                    title='Training Examples by Activity Type')
# plt.show()

# df_train['user-id'].value_counts().plot(kind='bar',
#                                   title='Training Examples by User')
# plt.show()
#
# df_test['activity_id'].value_counts().plot(kind='bar',
#                                    title='testing Examples by Activity Type')
# plt.show()
#
# df_test['user-id'].value_counts().plot(kind='bar',
#                                   title='testing Examples by User')
# plt.show()



# df_train = watch_df[watch_df['user-id'] > 1610]  #1611 - 1651
# df_test = phone_df[phone_df['user-id'] <= 1610]  #1600 - 1610
#
# df_train = pd.concat([phone_df[phone_df['user-id'] > 1610], watch_df[watch_df['user-id'] > 1610]])  #1611 - 1651
# df_test = pd.concat([phone_df[phone_df['user-id'] <= 1610], watch_df[watch_df['user-id'] <= 1610]])  #1600 - 1610


# Normalize features for training data set
for feature in feature_list_hand:
    df_train[feature] = feature_normalize(df_train[feature])
for feature in feature_list_hand:
    df_test[feature] = feature_normalize(df_test[feature])


# Round in order to comply to NSNumber from iOS
# df_train = df_train.round({'x-axis': 6, 'y-axis': 6, 'z-axis': 6})

# Reshape the training data into segments
# so that they can be processed by the network
x_train, y_train = create_segments_and_labels(df_train,
                                              feature_list_hand,
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
input_shape = (num_time_periods*num_sensors)
x_train = x_train.reshape(x_train.shape[0], input_shape)

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

print("\n--- Create neural network model ---\n")

# 1D CNN neural network
model_m = Sequential()
model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
# model_m.add(LSTM(1))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.8))
model_m.add(Dense(len(activity_map.keys()), activation='softmax'))
print(model_m.summary())
model_m.summary()

# model_m = Sequential()
# model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
# model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
# model_m.add(MaxPooling1D(3))
# model_m.add(Conv1D(160, 10, activation='relu'))
# model_m.add(MaxPooling1D(3))
# model_m.add(LSTM(1))
# # model_m.add(Conv1D(160, 10, activation='relu'))
# # model_m.add(GlobalAveragePooling1D())
# model_m.add(Dropout(0.8))
# model_m.add(Dense(len(activity_map.keys()), activation='softmax'))
# print(model_m.summary())
# model_m.summary()
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

model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])



# Hyper-parameters
BATCH_SIZE = 100
EPOCHS = 20

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      # callbacks=callbacks_list,
                      # validation_split=0.2,
                      verbose=1)

# %%

print("\n--- Learning curve of model training ---\n")

# summarize history for accuracy and loss
# plt.figure(figsize=(6, 4))
# plt.plot(history.history['accuracy'], "g--", label="Accuracy of training data")
# plt.plot(history.history['val_accuracy'], "g", label="Accuracy of validation data")
# plt.plot(history.history['loss'], "r--", label="Loss of training data")
# plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
# plt.title('Model Accuracy and Loss')
# plt.ylabel('Accuracy and Loss')
# plt.xlabel('Training Epoch')
# plt.ylim(0)
# plt.legend()
# plt.show()

#%%

print("\n--- Check against test data ---\n")

# Normalize features for testing data set
# df_test['x-axis'] = feature_normalize(df_test['x-axis'])
# df_test['y-axis'] = feature_normalize(df_test['y-axis'])
# df_test['z-axis'] = feature_normalize(df_test['z-axis'])
#
# df_test = df_test.round({'x-axis': 6, 'y-axis': 6, 'z-axis': 6})

x_test, y_test = create_segments_and_labels(df_test,
                                            feature_list_hand,
                                            TIME_PERIODS,
                                            STEP_DISTANCE,
                                            LABEL)

# Set input_shape / reshape for Keras
x_test = x_test.reshape(x_test.shape[0], input_shape)

x_test = x_test.astype("float32")
y_test = y_test.astype("float32")

y_test = np_utils.to_categorical(y_test, num_classes)

score = model_m.evaluate(x_test, y_test, verbose=1)

print(model_m.metrics_names)
print(score)

print("\nAccuracy on test data: %0.2f" % score[1])
print("\nLoss on test data: %0.2f" % score[0])

# %%

print("\n--- Confusion matrix for test data ---\n")

y_pred_test = model_m.predict(x_test)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

show_confusion_matrix(max_y_test, max_y_pred_test)

# %%

print("\n--- Classification report for test data ---\n")

print(classification_report(max_y_test, max_y_pred_test, target_names=activity_map.values()))