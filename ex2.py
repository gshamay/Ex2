# Alex Danieli 317618718
# Gil Shamay 033076324
seed = 415

import time
import pandas as pd
import zipfile
from io import StringIO
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from scipy import stats
import pickle
import numpy as np

from surprise import NMF
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy

# todo: 2020-06-10 18:45:36.536587: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# (tf.version.GIT_VERSION, tf.version.VERSION) --> # v2.0.0-rc2-26-g64c3d382ca 2.0.0
###########################################################################
# todo: Plan / feature engeneering\
#  DONE - consider remove first time value
#  DONE - time + gmt_offset --> new time ; remove GMT // (save time.. ?)
#  Done - calculate user click rate #  user -> targets + num seen #user that cliced more then once --> may click it again
#  DONE - find how to use this data w/o harming the accuracy - try to remove user / target / anything that is not common
#  Done - Creare SVD - user - item using the user recs to the target
#  Add SVD as input to the ANN (Hybrid)
#  Add SVD ensamble
#  SVD of more categorials
#  one HOT to some of the parameters
#  create a shared category between the different file batch so the same values will be used for all batches
#  consider use LSTM
#  Change the target to categorial and not numerical
#  helper data for the domain
#  target -> total seen + time # taret that is popular (and seen lately ? )
#  target + data --> Target CF
#  session based mechanism
#  date/time  --> date features
#  emphesize the connection between    user_recs    user_clicks    user_target_recs
#  one may be removed browser_platform and os_family // (save time..?)
###########################################################################
# parameters for Debug
# Todo: in Debug - change here
numOfTests = 1  # 5
epochs = 1  # 2
epochsOfBatch = 1  # 10
test = True
# test = False
limitNumOfFilesInTest = 4
loadPercentageTest = 0.95
basePath = "C:\\Users\\gshamay.DALET\\PycharmProjects\\RS\\Ex2\\models\\"
layers = [14, 9, 4]

bEnableSVD = True
K = 400
lam = 0.005
delta = 0.02
input_dim = 15
bAddSvdToAnn = False

if (bEnableSVD):
    if (bAddSvdToAnn):
        input_dim = input_dim + 1
    epochs = epochs + 1  # first epoch is used for SVD train
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
#  11  user_target_recs         462734 non-null  float64 //how many he saw this [0->100]
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
# globals
stringToPrintToFile = ""
numOffiles = 0
numOffilesInEpoch = 0
model = None
SVDModel = None
epochNum = 0
testNum = 0
totalLines = 0
modelFitted = False
bSvdTrained = False


###########################################
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


def saveModelToFile(dumpFileFullPath):
    with open(dumpFileFullPath, 'wb') as fp:
        pickle.dump(model, fp)


##############################################################


def train_svd(train_set):
    global SVDModel, bSvdTrained
    if (not bEnableSVD):
        printDebug("SVD Disabled")
        return
    printDebug("SVDModel fit start ; file [" + str(numOffilesInEpoch) + "]")
    beginTime = time.time()
    SVDModel.fit(train_set)
    bSvdTrained = True
    printDebug("SVDModel fit took[" + str(time.time() - beginTime) + "]")


##############################################################
# def readAndRunUncompressedFiles():
#     csvFiles = glob.glob("./data/*.csv");
#     for csvfile in csvFiles:
#         df = loadUncompressed(csvfile)
#         # printDebug(df)
#         handleDataChunk(df)

def loadUncompressed(path):
    chunksNum = 0
    beginTime = time.time()
    data = None
    pd.read_csv(path, chunksize=20000)
    for dataChunk in pd.read_csv(path, chunksize=20000):
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


def readAndRunZipFiles():
    global model
    global numOffiles
    global numOffilesInEpoch
    global totalLines
    epochBeginTime = time.time()
    numOffilesInEpoch = 0
    archive = zipfile.ZipFile('./data/bgu-rs.zip', 'r')
    totalLines = 0
    trainX = None
    trainY = None
    testX = None
    testY = None
    for file in archive.filelist:
        if ("part-" in file.filename and ".csv" in file.filename):
            fileBeginTime = time.time()
            df = readCSVFromZip(archive, file)
            printDebug("user_target_recs max[" + str(df['user_target_recs'].max(axis=0, skipna=True)) + "]")  # debug
            bLearnOnChunk = (not test) or ((limitNumOfFilesInTest > 1) and (
                    limitNumOfFilesInTest <= numOffiles + 1))  # don't learn on the last file in test mode - keep if for evaluation
            if (bLearnOnChunk):
                testX, testY, trainX, trainY = handleASingleDFChunk(
                    df, numOffiles, numOffilesInEpoch, testX, testY, trainX, trainY)
                printDebug("file handle time[" + str(time.time() - fileBeginTime)
                           + "]epochNum[" + str(epochNum)
                           + "]numOffilesInEpoch[" + str(numOffilesInEpoch)
                           + "]")
            else:
                printDebug("this file is used for evaluation")

            if (test):
                # calcullate error on the validation data
                # Evaluate the model with a partial part of the incoming data
                # can have wrong values between the epochs if different entries are selected for the test (enries taht the model was trained on)

                evaluateModel(model, testX, testY)  # eval on every file

                # test using a few files only
                if ((limitNumOfFilesInTest > 0) and (limitNumOfFilesInTest <= numOffiles)):
                    break

    # test each epoch - using the last read file any way - in test mode this file is not being trained on and in production it is
    if (totalLines > 0):
        # after every epoch - evaluate teh last trained file
        xCopy = df.copy()
        yCopy = xCopy.pop('is_click')
        evaluateModel(model, xCopy, yCopy)  # eval on last file in the epoch

    # print each epoch - with the file name
    printDebug("Epoch time[" + str(time.time() - epochBeginTime) + "]epochNum[" + str(epochNum) + "]")
    printToFile(
        "./models/model"
        + str(runStartTime)
        + "_lines" + str(totalLines)
        + "test" + str(testNum)
        + "Epoch" + str(epochNum)
        + ".log")


def handleASingleDFChunk(df, numOffiles, numOffilesInEpoch, testX, testY, trainX, trainY):
    global totalLines, SVDModel
    df = df.dropna()  # todo: do we need this?
    target = df.pop('is_click')

    if (test):
        printDebug("test mode - split train/validations")
        trainX, testX, trainY, testY = train_test_split(df, target, test_size=(1 - loadPercentageTest),
                                                        random_state=seed)
    else:
        trainX = df
        trainY = target

    ########################
    # SVD
    bFitModel = True
    if (bEnableSVD):
        if (epochNum < 1):
            bFitModel = False
            trainSVDWithCurrentDataChunk(trainX, trainY)
            printDebug(
                "train SVD "
                + "numOffilesInEpoch[" + str(numOffilesInEpoch) + "]"
                + "numOffiles[" + str(numOffiles) + "]"
                + "epochNum[" + str(epochNum) + "]"
            )
        else:
            addSvdDataToTheDFChunk(SVDModel, trainX, trainY)
    ########################

    if (bFitModel):
        # if we use SVD - we'll fit the maion model using the SVD Data - first epoch is dedicated for this
        handleDataChunk(trainX, trainY)
        printDebug(
            "handled lines[" + str(df.__len__()) + "]"
            + "total[" + str(totalLines) + "]"
            + "numOffilesInEpoch[" + str(numOffilesInEpoch) + "]"
            + "numOffiles[" + str(numOffiles) + "]"
            + "epochNum[" + str(epochNum) + "]"
        )
        totalLines = totalLines + df.__len__()
    return testX, testY, trainX, trainY


def addSvdDataToTheDFChunk(SVDModel, X, Y):
    # calculate SVD data from prev batches into this batch - if there is a model
    if ((not bEnableSVD) or (not bAddSvdToAnn)):
        return

    if (bSvdTrained):
        testing_predictions = predictWithSVD(SVDModel, X)
        X['svdUserTarget'] = testing_predictions
        # We want the ANN to use the SVD predicted data and not the calulated one that is provided to the SVD
        # since this is what it will get oin the test as well


def predictWithSVD(SVDModel, X):
    printDebug("start predictWithSVD")
    beginTime = time.time()
    test_set_svd_Predict = createDataFrameForSvdPredict(X)
    testing_predictions = SVDModel.test(test_set_svd_Predict)
    testing_predictions = [x[3] for x in testing_predictions]
    printDebug("predictWithSVD took [" + str(time.time() - beginTime) + "]")
    return testing_predictions


def trainSVDWithCurrentDataChunk(X, Y):
    X = X.dropna()
    df_train_svd = createDataFrameForSvdTrain(X, Y)
    reader = Reader(rating_scale=(0, 1))
    train_data = Dataset.load_from_df(df_train_svd, reader)
    train_set = train_data.build_full_trainset()
    train_svd(train_set)


def createDataFrameForSvdTrain(X, Y):
    train_svd_data = {
        'user_id_hash': X['user_id_hash'],
        'target_id_hash': X['target_id_hash'],
        'user_target_recs': X['user_target_recs'],
        'is_click': Y
    }
    df_train_svd = pd.DataFrame(data=train_svd_data)
    # df_train_svd['rate'] = df_train_svd['is_click'] * ((120 - (df_train_svd['user_target_recs'])) / 120)
    df_train_svd['rate'] = df_train_svd['is_click']
    df_train_svd.pop('user_target_recs')
    df_train_svd.pop('is_click')
    return df_train_svd


def createDataFrameForSvdPredict(df):
    test_svd_data = {
        'user_id_hash': df['user_id_hash'],
        'target_id_hash': df['target_id_hash'],
    }
    df_svd_Predict = pd.DataFrame(data=test_svd_data)
    df_svd_Predict['rate'] = 0.5  # test method of SVD must get the 'rate'
    test_set_svd_Predict = [tuple(x) for x in df_svd_Predict.to_numpy()]
    return test_set_svd_Predict


def normalizeResults(x):
    if (x < 0.38):
        return 0
    else:
        if (x > 0.62):
            return 1
        else:
            return x


def evaluateModel(model, testX, testY):
    if (not modelFitted):
        printDebug("evaluateModel skipped - model was not fit yet")
        return
    AUCSVD = 0
    AUCEnsamble = 0
    if (bSvdTrained):
        xCopy = testX.copy()
        yCopy = testY.copy()
        testResSvd = predictWithSVD(SVDModel, xCopy)
        m = tf.keras.metrics.AUC()
        m.update_state(yCopy.values, testResSvd)
        AUCSVD = m.result().numpy()
        if (bAddSvdToAnn):
            testX['svdUserTarget'] = testResSvd

    # main model (with or without SVD in to teh ANN)
    testX, testY = transformDataFramesToTFArr(testX, testY)
    testRes = model.predict(testX)
    m = tf.keras.metrics.AUC()
    m.update_state(testY, testRes.flatten())
    AUC = m.result().numpy()

    # ensamble
    if (bSvdTrained):
        ensambleRes = (testRes.flatten() + testResSvd) / 2
        m = tf.keras.metrics.AUC()
        m.update_state(testY, ensambleRes)
        AUCEnsamble = m.result().numpy()

    # Normalized version
    normRes = np.vectorize(normalizeResults)(testRes.flatten())
    m.reset_states()
    m.update_state(testY, normRes)
    AUCNorm = m.result().numpy()

    # todo: Check that the res data is not <0 or >1 and fix if it does
    printDebug(''
               + 'test: AUC[' + str(AUC) + ']'
               + 'AUCNorm[' + str(AUCNorm) + ']'
               + 'AUCSVD[' + str(AUCSVD) + ']'
               + 'AUCEnsamble[' + str(AUCEnsamble) + ']'
               + 'layers[' + str(layers) + ']'
               + 'Epoch[' + str(epochNum) + ']'
               + str(stats.describe(testRes))
               )


def readCSVFromZip(archive, file):
    global numOffiles
    global numOffilesInEpoch
    readBeginTime = time.time()
    fileData = archive.read(file.filename)
    numOffiles = numOffiles + 1
    numOffilesInEpoch = numOffilesInEpoch + 1
    s = str(fileData, 'utf-8')
    data = StringIO(s)
    df = pd.read_csv(data)
    printDebug("read Zip file took [" + str(time.time() - readBeginTime) + "][" + str(numOffiles) + "]")
    return df


def handleDataChunk(df, target):
    keepStatisticalData()
    fitAnn(df, target)


def fitAnn(df, target):
    global model
    global modelFitted
    X, y = transformDataFramesToTFArr(df, target)
    # fit Model with chunk Data
    fitBeginTime = time.time()
    printDebug("start fit dataChunk epochNum[" + str(epochNum) + "]epochs[" + str(epochs) + "]")
    # checkpoint_path = generateModelFileName()
    # Create a callback that saves the model's weights
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                                  save_weights_only=True,
    #                                                  verbose=1)
    model.fit(x=X,
              y=y,
              batch_size=64,
              epochs=epochsOfBatch,  # we do the epochs on the overall Data ; this is an ephoch of the minibatch
              use_multiprocessing=True,
              verbose=2,
              workers=3
              # ,callbacks=[cp_callback]
              )
    modelFitted = True
    printDebug("fit dataChunk took[" + str(time.time() - fitBeginTime) + "]")
    # loss, accuracy = model.evaluate(X, y)
    # printDebug('Accuracy: %.2f' % (accuracy * 100))


def generateModelFileName(basePath):
    if (basePath is None):
        checkpoint_path = "./models/model_" + str(runStartTime) + "_part" + str(numOffiles) + ".dump"
    else:
        checkpoint_path = basePath + str(runStartTime) + "_part" + str(numOffiles) + ".dump"
    return checkpoint_path


def keepStatisticalData():
    pass
    # statistical data
    # currentUsers = df['user_id_hash'].unique()
    # currentTargets = df['target_id_hash'].unique()


def transformDataFramesToTFArr(df, target):
    # https://www.tensorflow.org/tutorials/load_data/pandas_dataframe
    # Convert column which is an object in the dataframe to a discrete numerical value.
    # todo: Fix warning  A value is trying to be set on a copy of a slice from a DataFrame.  # Try using .loc[row_indexer,col_indexer] = value instead
    # todo: Check that categories are similare between file batches
    fitBeginTime = time.time()
    df['user_id_hash'] = pd.Categorical(df['user_id_hash'])
    df['user_id_hash'] = df.user_id_hash.cat.codes
    df['target_id_hash'] = pd.Categorical(df['target_id_hash'])
    df['target_id_hash'] = df.target_id_hash.cat.codes
    df['syndicator_id_hash'] = pd.Categorical(df['syndicator_id_hash'])
    df['syndicator_id_hash'] = df.syndicator_id_hash.cat.codes
    df['campaign_id_hash'] = pd.Categorical(df['campaign_id_hash'])
    df['campaign_id_hash'] = df.campaign_id_hash.cat.codes
    df['placement_id_hash'] = pd.Categorical(df['placement_id_hash'])
    df['placement_id_hash'] = df.placement_id_hash.cat.codes
    df['publisher_id_hash'] = pd.Categorical(df['publisher_id_hash'])
    df['publisher_id_hash'] = df.publisher_id_hash.cat.codes
    df['source_id_hash'] = pd.Categorical(df['source_id_hash'])
    df['source_id_hash'] = df.source_id_hash.cat.codes

    df['target_item_taxonomy'] = pd.Categorical(df['target_item_taxonomy'])
    df['target_item_taxonomy'] = df.target_item_taxonomy.cat.codes
    df['source_item_type'] = pd.Categorical(df['source_item_type'])
    df['source_item_type'] = df.source_item_type.cat.codes
    df['browser_platform'] = pd.Categorical(df['browser_platform'])
    df['browser_platform'] = df.browser_platform.cat.codes
    df['country_code'] = pd.Categorical(df['country_code'])
    df['country_code'] = df.country_code.cat.codes
    df['region'] = pd.Categorical(df['region'])
    df['region'] = df.region.cat.codes
    df['os_family'] = pd.Categorical(df['os_family'])
    df['os_family'] = df.os_family.cat.codes

    df['day_of_week'] = pd.Categorical(df['day_of_week'])
    df['day_of_week'] = df.day_of_week.cat.codes
    # printDebug(str(df.info()))
    # dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))# option to include X and target together
    # train_dataset = dataset.shuffle(len(df)).batch(1)

    df.pop('page_view_start_time')

    # time + gmt_offset --> new time ; remove GMT // (save time.. ?)
    df['time_of_day'] = df['time_of_day'] + (df['gmt_offset'] / 100.0)
    df.pop('gmt_offset')

    # user click rate, with an option that the user is a cold start
    df['user_click_rate'] = (df['user_clicks'] + 1) / (df['user_recs'] + 1)
    # give more meaning to click rate differences
    df['user_click_rate_pow'] = (df['user_click_rate'] * 10).__pow__(2)

    # todo: We should  find how to use categorial data
    df.pop('user_id_hash')
    df.pop('target_id_hash')
    df.pop('syndicator_id_hash')
    df.pop('campaign_id_hash')
    df.pop('placement_id_hash')
    df.pop('publisher_id_hash')
    df.pop('source_id_hash')

    # df.pop('target_item_taxonomy')
    # df.pop('source_item_type')
    # df.pop('browser_platform')
    # df.pop('country_code')
    # df.pop('region')
    # df.pop('os_family')

    printDebug("transformDataToX_Y took[" + str(time.time() - fitBeginTime) + "]")
    if (target is None):
        return df.values, None
    else:
        return df.values, target.values


def builedSvdModel(K, lam, delta):
    global SVDModel
    if (not bEnableSVD):
        return
    # SVDModel = SVD(n_factors=K, lr_all=lam, reg_all=delta)
    SVDModel = SVD(n_factors=300)


def builedModel():
    global model
    model = Sequential()
    METRICS = [
        # tf.keras.metrics.TruePositives(name='tp'),
        # tf.keras.metrics.FalsePositives(name='fp'),
        # tf.keras.metrics.TrueNegatives(name='tn'),
        # tf.keras.metrics.FalseNegatives(name='fn'),
        # tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        # tf.keras.metrics.Precision(name='precision'),
        # tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]
    loss = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    model.add(Dense(layers[0], input_dim=input_dim, activation='sigmoid'))
    model.add(Dense(layers[1], activation='sigmoid'))
    # model.add(Dense(layers[1], activation='relu'))
    # model.add(Dense(layers[1], activation='relu'))
    model.add(Dense(layers[2], activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))  # softmax/sigmoid
    # compile the keras model
    model.compile(loss=loss, optimizer=optimizer, metrics=METRICS)


def predictOnTest():
    global model
    testFilewName = "./testData/test_file.csv"
    printDebug("predictOnTest [" + testFilewName + "]")
    dfTest = loadUncompressed(testFilewName)
    IDs = dfTest.pop('Id')
    testResSvd = None

    if (bEnableSVD):
        if (bSvdTrained):
            # predict on the test data using SVD
            testResSvd = predictWithSVD(SVDModel, dfTest)
            if (bAddSvdToAnn):  # use SVD data as input in the test data for the ANN
                dfTest['svdUserTarget'] = testResSvd
            # write SVD res to file
            res = pd.DataFrame({'Id': IDs, 'Predicted': testResSvd}, columns=['Id', 'Predicted'])
            res.Id = res.Id.astype(int)
            resFileName = "./models/modelSvd_" + str(runStartTime) + "_res.csv"
            res.to_csv(resFileName, header=True, index=False)
        else:
            printDebug("Error - Check SVD status")

    test, _ = transformDataFramesToTFArr(dfTest, None)
    Predicted = model.predict(test)
    PredictedArr = np.array(Predicted)
    res = pd.DataFrame({'Id': IDs, 'Predicted': list(PredictedArr.flatten())}, columns=['Id', 'Predicted'])
    res.Id = res.Id.astype(int)
    resFileName = "./models/model_" + str(runStartTime) + "_res.csv"
    res.to_csv(resFileName, header=True, index=False)

    # ensamble
    if (bSvdTrained):
        ensambleRes = (Predicted.flatten() + testResSvd) / 2
        # write ensamble res to file
        res = pd.DataFrame({'Id': IDs, 'Predicted': ensambleRes}, columns=['Id', 'Predicted'])
        res.Id = res.Id.astype(int)
        resFileName = "./models/modelEnsamble_" + str(runStartTime) + "_res.csv"
        res.to_csv(resFileName, header=True, index=False)


def saveModel():
    printDebug('saveModel')
    checkpoint_path = generateModelFileName(basePath)
    # saving the model in tensorflow format
    model.save(checkpoint_path, save_format='tf')
    # loading the saved model


def run():
    global runStartTime, epochNum, testNum
    for testNum in range(0, numOfTests):
        printDebug('*********************************************')
        layers[0] = layers[0] + 1
        layers[1] = layers[1] + 1
        layers[2] = layers[2] + 1
        printDebug(''
                   + 'testNum[' + str(testNum) + ']'
                   + 'layers[' + str(layers) + ']'
                   + 'epochs[' + str(epochs) + ']'
                   )
        runStartTime = time.time()
        builedModel()
        builedSvdModel(K, lam, delta)
        # read the data and fit
        epochNum = 0
        for epochNum in range(0, epochs):
            printDebug("-------------------------------")
            printDebug("start epoch[" + str(epochNum) + "]")
            readAndRunZipFiles()

        saveModel()
        # loaded_model = tf.keras.models.load_model('./MyModel_tf')
        predictOnTest()


run()
printDebug(" ---- Done ---- ")
printToFile(
    "./models/model"
    + str(runStartTime)
    + "_lines" + str(totalLines)
    + "test" + str(testNum)
    + "Epoch" + str(epochNum)
    + "Done"
    + ".log")

exit(0)
