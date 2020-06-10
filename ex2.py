# Alex Danieli 317618718
# Gil Shamay 033076324

import time
import pandas as pd
import glob
import zipfile
from io import StringIO
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from keras_pandas import lib
from keras_pandas.Automater import Automater
from sklearn.model_selection import train_test_split
import pickle

###########################################################################
# the DATA
# Data columns (total 23 columns):
#  #   Column                   Non-Null Count   Dtype
# ---  ------                   --------------   -----
#  0   page_view_start_time     462734 non-null  int64
#  1   user_id_hash             462734 non-null  object
#  2   target_id_hash           462734 non-null  object
#  3   syndicator_id_hash       462734 non-null  object //the customer
#  4   campaign_id_hash         462734 non-null  object
#  5   empiric_calibrated_recs  462734 non-null  float64 //num of clicks target got - calibrated low/high - float //
#  6   empiric_clicks           462734 non-null  float64 //num of clicks target got - actual  - int
#  7   target_item_taxonomy     462734 non-null  object //BUSINESS/SPORT/...
#  8   placement_id_hash        462734 non-null  object //affect the calibration
#  9   user_recs                462734 non-null  float64 // user actual saw
#  10  user_clicks              462734 non-null  float64 // user actual clicked
#  11  user_target_recs         462734 non-null  float64 //how many he saw this
#  12  publisher_id_hash        462734 non-null  object //the website
#  13  source_id_hash           462734 non-null  object //web actual page
#  14  source_item_type         462734 non-null  object //type of page
#  15  browser_platform         462734 non-null  object //OS
#  16  os_family                462734 non-null  int64
#  17  country_code             462727 non-null  object
#  18  region                   462724 non-null  object
#  19  day_of_week              462734 non-null  int64
#  20  time_of_day              462734 non-null  int64
#  21  gmt_offset               462734 non-null  int64
#  22  is_click                 462734 non-null  float64
# dtypes: float64(6), int64(5), object(12)

###########################################
stringToPrintToFile = ""


def printDebug(str):
    global stringToPrintToFile
    print(str)
    stringToPrintToFile = stringToPrintToFile + str + "\r"


def printToFile(fileName):
    global stringToPrintToFile
    file1 = open(fileName, "a")
    file1.write("\r************************\r")
    file1.write(stringToPrintToFile)
    stringToPrintToFile = ""
    file1.close()


def saveModelToFile(fileName):
    ##########################################
    pickle.dump(model, fileName)
    ##########################################
    # Load the Model to be reused - sample code
    # mySvdload = None
    # with open(dumpFileFullPath, 'rb') as fp:
    #     mySvdload = pickle.load(fp)


def processDataChunk(dataChunk):
    # todo: check if needed
    # dataChunk.drop(dataChunk.columns[[0]], axis=1, inplace=True)
    return dataChunk


def loadUncompressed(path):
    chunksNum = 0
    beginTime = time.time()
    data = None
    pd.read_csv(path, chunksize=20000)
    for dataChunk in pd.read_csv(path, chunksize=20000):
        dataChunk = processDataChunk(dataChunk)
        if (data is None):
            data = dataChunk
        else:
            data = data.append(dataChunk, ignore_index=True)

        if (chunksNum % 10 == 0):
            took = time.time() - beginTime
            # printDebug(str(chunksNum) + " " + str(took))
            # break  # TODO: DEBUG DEBUG DEBUG - FOR FAST TESTS ONLY

        chunksNum += 1

    took = time.time() - beginTime
    printDebug("LOAD: chunksNum[" + str(chunksNum) + "]took[" + str(took) + "]data[" + str(len(data)) + "]")
    return data


# helper data for the domain
# user -> targets + num seen #user that cliced more then once --> may click it again
# target -> total seen + time # taret that is popular (and seen lately ? )
# user -> clics ratio # is teh user clicking/ can we expect him to click
# target + data --> Target CF
# session based mechanism
# date/time  --> date features


def readAndRunUncompressedFiles():
    csvFiles = glob.glob("./data/*.csv");
    for csvfile in csvFiles:
        df = loadUncompressed(csvfile)
        # printDebug(df)
        handleDataChunk(df)


def readAndRunZipFiles():
    runStartTime = time.time()
    archive = zipfile.ZipFile('./data/bgu-rs.zip', 'r')
    totalLines = 0
    numOffiles = 0
    limitNumOfFiles = 4
    readBeginTime = time.time()
    for file in archive.filelist:
        if ("part-" in file.filename and ".csv" in file.filename):
            fileData = archive.read(file.filename)
            printDebug("read Zip file took [" + str(time.time() - readBeginTime) + "][" + str(numOffiles) + "]")
            numOffiles = numOffiles + 1
            s = str(fileData, 'utf-8')
            data = StringIO(s)
            df = pd.read_csv(data)
            handleDataChunk(df)
            totalLines = totalLines + df.__len__()
            printDebug(
                "lines[" + str(df.__len__()) + "]total[" + str(totalLines) + "]numOffiles[" + str(numOffiles) + "]")
            saveModelToFile("./models/run" + str(runStartTime) + "_part" + str(numOffiles) + ".dump")
    printToFile("./models/run" + str(runStartTime) + "_lines" + str(totalLines) + ".log")


def handleDataChunk(df):
    # keep statistical data
    # currentUsers = df['user_id_hash'].unique()
    # currentTargets = df['target_id_hash'].unique()

    df = df.dropna()
    # printDebug(str(df.info()))

    # X, y = transformDataToX_Y_Automater(df)
    X, y = transformDataToX_Y(df)

    # fit Model with chunk Data
    epochs = 4
    batch_size = 10
    fitBeginTime = time.time()
    printDebug("start fit dataChunk epochs[" + str(epochs) + "]batch_size[" + str(batch_size) + "]")
    model.fit(X, epochs=epochs, use_multiprocessing=True, verbose=2, workers=3)
    printDebug("fit dataChunk took[" + str(time.time() - fitBeginTime) + "]")

    loss, accuracy = model.evaluate(X, y)
    printDebug('Accuracy: %.2f' % (accuracy * 100))


output_var = 'is_click'
data_type_dict = {
    # numerical cause # ValueError: all the input array dimensions for the concatenation axis must match exactly,
    #       but along dimension 0, the array at index 0 has size 462734 and the array at index 8 has size 462735

    # 'numerical': [
    #     # 'page_view_start_time',
    #     # 'empiric_calibrated_recs',
    #     # 'empiric_clicks',
    #     # 'user_recs',
    #     # 'user_clicks',
    #     # 'user_target_recs',
    #     # 'time_of_day',
    #     # 'gmt_offset'
    # ],
    'categorical': [
        'user_id_hash',
        'target_id_hash',
        'syndicator_id_hash',
        'campaign_id_hash',
        'target_item_taxonomy',
        'placement_id_hash',
        'publisher_id_hash',
        'source_id_hash',
        'source_item_type',
        'browser_platform',
        'os_family',
        'country_code',
        'region',
        'day_of_week',
        'is_click']}


def transformDataToX_Y_Automater(df):
    # Transform the data set, using keras_pandas
    fitBeginTime = time.time()
    # Create and fit Automater
    automater = Automater(data_type_dict=data_type_dict, output_var=output_var)
    printDebug("start Automater.fit")
    automater.fit(df)
    printDebug("Automater.fit took[" + str(time.time() - fitBeginTime) + "]")
    printDebug("start Automater.transform")
    fitBeginTime = time.time()
    X, y = automater.transform(df, df_out=False)
    printDebug("Automater.transform took[" + str(time.time() - fitBeginTime) + "]")
    return X, y


def transformDataToX_Y(df):
    # https://www.tensorflow.org/tutorials/load_data/pandas_dataframe
    # Convert column which is an object in the dataframe to a discrete numerical value.
    # todo:     A value is trying to be set on a copy of a slice from a DataFrame.  # Try using .loc[row_indexer,col_indexer] = value instead
    fitBeginTime = time.time()
    df['user_id_hash'] = pd.Categorical(df['user_id_hash'])
    df['user_id_hash'] = df.user_id_hash.cat.codes
    df['target_id_hash'] = pd.Categorical(df['target_id_hash'])
    df['target_id_hash'] = df.target_id_hash.cat.codes
    df['syndicator_id_hash'] = pd.Categorical(df['syndicator_id_hash'])
    df['syndicator_id_hash'] = df.syndicator_id_hash.cat.codes
    df['campaign_id_hash'] = pd.Categorical(df['campaign_id_hash'])
    df['campaign_id_hash'] = df.campaign_id_hash.cat.codes
    df['target_item_taxonomy'] = pd.Categorical(df['target_item_taxonomy'])
    df['target_item_taxonomy'] = df.target_item_taxonomy.cat.codes
    df['placement_id_hash'] = pd.Categorical(df['placement_id_hash'])
    df['placement_id_hash'] = df.placement_id_hash.cat.codes
    df['publisher_id_hash'] = pd.Categorical(df['publisher_id_hash'])
    df['publisher_id_hash'] = df.publisher_id_hash.cat.codes
    df['source_id_hash'] = pd.Categorical(df['source_id_hash'])
    df['source_id_hash'] = df.source_id_hash.cat.codes
    df['source_item_type'] = pd.Categorical(df['source_item_type'])
    df['source_item_type'] = df.source_item_type.cat.codes
    df['browser_platform'] = pd.Categorical(df['browser_platform'])
    df['browser_platform'] = df.browser_platform.cat.codes
    df['country_code'] = pd.Categorical(df['country_code'])
    df['country_code'] = df.country_code.cat.codes
    df['region'] = pd.Categorical(df['region'])
    df['region'] = df.region.cat.codes
    # printDebug(str(df.info()))
    target = df.pop('is_click')
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
    train_dataset = dataset.shuffle(len(df)).batch(1)
    printDebug("transformDataToX_Y took[" + str(time.time() - fitBeginTime) + "]")
    return train_dataset, target


# define the model
model = Sequential()
model.add(Dense(5, input_dim=22, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# read the data and fit
readAndRunZipFiles()

# todo: 2020-06-10 18:45:36.536587: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
