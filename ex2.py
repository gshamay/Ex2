# Alex Danieli 317618718
# Gil Shamay 033076324

import time
import pandas as pd
import glob
import zipfile
from io import StringIO

import tensorflow as tf
from tensorflow import keras

#the DATA
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


def processDataChunk(dataChunk):
    # todo: check if needed
    # dataChunk.drop(dataChunk.columns[[0]], axis=1, inplace=True)
    return dataChunk


def load(path):
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
            # print(str(chunksNum) + " " + str(took))
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
        df = load(csvfile)
        print(df)
        handleDataChunk(df)


def readAndRunZipFiles():
    archive = zipfile.ZipFile('./data/bgu-rs.zip', 'r')
    totalLines = 0
    numOffiles = 0
    limitNumOfFiles = 5
    for file in archive.filelist:
        if ("part-" in file.filename and ".csv" in file.filename):
            fileData = archive.read(file.filename)
            numOffiles = numOffiles + 1
            s = str(fileData, 'utf-8')
            data = StringIO(s)
            df = pd.read_csv(data)
            handleDataChunk(df)
            totalLines = totalLines + df.__len__()
            printDebug(
                "lines[" + str(df.__len__()) + "]total[" + str(totalLines) + "]numOffiles[" + str(numOffiles) + "]")


def handleDataChunk(df):
    users = df['user_id_hash'].unique()
    targets = df['target_id_hash'].unique()

    print()

readAndRunZipFiles()
