# Alex Danieli 317618718
# Gil Shamay 033076324
seed = 415

import time
import pandas as pd
import zipfile
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import pickle
import numpy as np

# todo: 2020-06-10 18:45:36.536587: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
###########################################################################
# TODO (Alex):
#  Check why do we have each time two rows of AUC one is always 0.73, while the other is always 0.99889
#  Check why we run our tree few times - it should be trained only once
#  Make tuning to parameters
#  Try LightGBM or XGBoost instead of RandomForest
#
# todo: Plan / feature engeneering\
#  DONE - consider remove first time value
#  DONE - time + gmt_offset --> new time ; remove GMT // (save time.. ?)
#  Done - calculate user click rate #  user -> targets + num seen #user that cliced more then once --> may click it again
#  DONE - find how to use this data w/o harming the accuracy - try to remove user / target / anything that is not common
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
numOfTests = 3
epochs = 5
test = True
# test = False
limitNumOfFilesInTest = 1
basePath = "C:\\Users\\alexd\\OneDrive\\Ben Gurion\\Recommender Systems\\ex2\\Ex2\\models\\"

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
# globals
stringToPrintToFile = ""
numOffiles = 0
numOffilesInEpoch = 0
model = None
epochNum = 0
testNum = 0
totalLines = 0


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
            # break  # todo: DEBUG DEBUG DEBUG - FOR FAST TESTS ONLY

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
            testX, testY, trainX, trainY = handleASingleDFChunk(
                df, numOffiles, numOffilesInEpoch, testX, testY, trainX, trainY)
            printDebug("file handle time[" + str(time.time() - fileBeginTime)
                       + "]epochNum[" + str(epochNum)
                       + "]numOffilesInEpoch[" + str(numOffilesInEpoch)
                       + "]")
            if (test):
                # calcullate error on the validation data
                # Evaluate the model with a partial part of the incoming data
                # can have wrong values between teh epochs if different entries are selected for the test (enries taht the model was trained on)
                evaluateModel(model, testX, testY, True)
                # test using a few files only
                if ((limitNumOfFilesInTest > 0) and (limitNumOfFilesInTest <= numOffiles)):
                    break

    # test each epoch - using the last read file any way
    if (totalLines > 0):
        # after every epoch - evaluate teh last trained file
        evaluateModel(model, trainX, trainY, False)
    # print each epoch - with the file name
    printDebug("Epoch time[" + str(time.time() - epochBeginTime) + "]epochNum[" + str(epochNum) + "]")
    printToFile(
        "./models/model"
        + str(runStartTime)
        + "_lines" + str(totalLines)
        + "test" + str(testNum)
        + "Epoch" + str(epochNum)
        + ".log")

# TODO: Check what is the purpose of getting testX, testY, trainX, trainY as part of an input
def handleASingleDFChunk(df, numOffiles, numOffilesInEpoch, testX, testY, trainX, trainY):
    global totalLines
    df = df.dropna()  # todo: do we need this?
    target = df.pop('is_click')
    if (test):
        printDebug("test mode - split train/validations")
        trainX, testX, trainY, testY = train_test_split(df, target, test_size=0.25, random_state=seed)
    else:
        trainX = df
        trainY = target
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


def normalizeResults(x):
    if (x < 0.3):
        return 0
    if (x > 0.7):
        return 1
    else:
        return x


def evaluateModel(model, testX, testY, bTransform):
    if (bTransform):
        testX, testY = transformDataFramesToTFArr(testX, testY)
    else:
        testX = testX.values
        testY = testY.values

    testRes = model.predict(testX)
    AUC = roc_auc_score(testY, testRes)

    normRes = np.vectorize(normalizeResults)(testRes)
    AUCNorm = roc_auc_score(testY, normRes)

    # todo: Check that the res data is not <0 or >1 and fix if it does
    printDebug(''
               + 'test: AUC[' + str(AUC) + ']'
               + 'test: AUCNorm[' + str(AUCNorm) + ']'
               + 'Epoch[' + str(epochNum) + ']'
               + str(stats.describe(testRes))
               )


def readCSVFromZip(archive, file):
    global numOffiles
    global numOffilesInEpoch
    readBeginTime = time.time()
    fileData = archive.read(file.filename)
    printDebug("read Zip file took [" + str(time.time() - readBeginTime) + "][" + str(numOffiles) + "]")
    numOffiles = numOffiles + 1
    numOffilesInEpoch = numOffilesInEpoch + 1
    s = str(fileData, 'utf-8')
    data = StringIO(s)
    df = pd.read_csv(data)
    return df


def handleDataChunk(df, target):
    keepStatisticalData()
    fitAnn(df, target)


def fitAnn(df, target):
    global model
    X, y = transformDataFramesToTFArr(df, target)
    # fit Model with chunk Data
    fitBeginTime = time.time()
    printDebug("start fit dataChunk epochNum[" + str(epochNum) + "]epochs[" + str(epochs) + "]")
    # checkpoint_path = generateModelFileName()
    # Create a callback that saves the model's weights
    model.fit(X, y)
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

    df['os_family'] = pd.Categorical(df['os_family'])
    df['os_family'] = df.os_family.cat.codes
    df['day_of_week'] = pd.Categorical(df['day_of_week'])
    df['day_of_week'] = df.day_of_week.cat.codes
    printDebug("transformDataToX_Y took[" + str(time.time() - fitBeginTime) + "]")

    df.pop('page_view_start_time')

    # time + gmt_offset --> new time ; remove GMT // (save time.. ?)
    df['time_of_day'] = df['time_of_day'] + (df['gmt_offset'] / 100.0)
    df.pop('gmt_offset')

    # user click rate, with an option that the user is a cold start
    df['user_click_rate'] = (df['user_clicks'] + 1) / (df['user_recs'] + 1)
    # give more meaning to click rate differences # TODO - why multiply 10 times - it is meaningless, no?
    df['user_click_rate_pow'] = (df['user_click_rate'] * 10).__pow__(2)

    # TODO: We should  find how to use this data
    df.pop('user_id_hash')
    df.pop('target_id_hash')
    df.pop('syndicator_id_hash')
    df.pop('campaign_id_hash')
    df.pop('placement_id_hash')
    df.pop('publisher_id_hash')
    df.pop('source_id_hash')

    if (target is None):
        return df.values, None
    else:
        return df.values, target.values


def builedModel():
    global model
    model = RandomForestClassifier(verbose=2, n_jobs=3)


def predictOnTest():
    global model
    testFilewName = "./testData/test_file.csv"
    printDebug("predictOnTest [" + testFilewName + "]")
    dfTest = loadUncompressed(testFilewName)
    IDs = dfTest.pop('Id')
    test,_ = transformDataFramesToTFArr(dfTest, None)
    Predicted = model.predict(test)
    PredictedArr = np.array(Predicted)
    res = pd.DataFrame({'Id': IDs, 'Predicted': list(PredictedArr.flatten())}, columns=['Id', 'Predicted'])
    res.Id = res.Id.astype(int)
    resFileName = "./models/model_" + str(runStartTime) + "_res.csv"
    res.to_csv(resFileName, header=True, index=False)


def run():
    global runStartTime, testNum
    for testNum in range(0, numOfTests):
        printDebug('*********************************************')
        printDebug(''
                   + 'testNum[' + str(testNum) + ']'
                   )
        runStartTime = time.time()
        builedModel()
        # read the data and fit
        printDebug("-------------------------------")
        readAndRunZipFiles()

        predictOnTest()


run()
printDebug(" ---- Done ---- ")
printToFile(
    "./models/model"
    + str(runStartTime)
    + "_lines" + str(totalLines)
    + "test" + str(testNum)
    + "Done"
    + ".log")
exit(0)
