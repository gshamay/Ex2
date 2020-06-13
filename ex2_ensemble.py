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
basePath = "C:\\Users\\alexd\\OneDrive\\Ben Gurion\\Recommender Systems\\ex2\\Ex2\\models\\"

####################################################################################
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
###############################################################################################


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
        chunksNum += 1
    took = time.time() - beginTime
    print("LOAD: chunksNum[" + str(chunksNum) + "]took[" + str(took) + "]data[" + str(len(data)) + "]")
    return data


def readAndRunZipFiles(model):
    beginTime = time.time()
    archive = zipfile.ZipFile('./data/bgu-rs.zip', 'r')
    for file in archive.filelist:
        if "part-" in file.filename and ".csv" in file.filename:
            fileBeginTime = time.time()
            df = readCSVFromZip(archive, file)
            df = df.dropna()  # todo: do we need this?
            trainY = df.pop('is_click')
            trainX = df
            print("file handle time[" + str(time.time() - fileBeginTime)+ "]")

    evaluateModel(model, trainX, trainY)
    print("Epoch time[" + str(time.time() - beginTime) + "]")


def normalizeResults(x):
    if (x < 0.3):
        return 0
    if (x > 0.7):
        return 1
    else:
        return x


def evaluateModel(model, testX, testY):
    testRes = model.predict(testX)
    AUC = roc_auc_score(testY, testRes)

    normRes = np.vectorize(normalizeResults)(testRes)
    AUCNorm = roc_auc_score(testY, normRes)

    # todo: Check that the res data is not <0 or >1 and fix if it does
    print(''
               + 'test: AUC[' + str(AUC) + ']'
               + 'test: AUCNorm[' + str(AUCNorm) + ']'
               + str(stats.describe(testRes))
               )


def readCSVFromZip(archive, file):
    readBeginTime = time.time()
    fileData = archive.read(file.filename)
    print("read Zip file took [" + str(time.time() - readBeginTime) + "]")
    s = str(fileData, 'utf-8')
    data = StringIO(s)
    df = pd.read_csv(data)
    return df


def trainModel(X, y, model):
    fitBeginTime = time.time()
    print("start fit dataChunk")
    model.fit(X, y)
    print("fit dataChunk took[" + str(time.time() - fitBeginTime) + "]")


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
    print("transformDataToX_Y took[" + str(time.time() - fitBeginTime) + "]")

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
    return RandomForestClassifier(verbose=2, n_jobs=3)


def predictOnTest(model):
    testFilewName = "./testData/test_file.csv"
    print("predictOnTest [" + testFilewName + "]")
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
        print('*********************************************')
        print(''
                   + 'testNum[' + str(testNum) + ']'
                   )
        runStartTime = time.time()
        model = builedModel()
        # read the data and fit
        print("-------------------------------")
        readAndRunZipFiles(model)

        predictOnTest(model)


run()
print(" ---- Done ---- ")

exit(0)
