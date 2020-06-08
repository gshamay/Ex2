# Alex Danieli 317618718
# Gil Shamay 033076324

import time
import pandas as pd
import glob
import zipfile
from io import StringIO

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


#read uncompressed  files
# csvFiles = glob.glob("./data/*.csv");
# for csvfile in  csvFiles:
#     data = load(csvfile)
#     print(data)

#read ziped files
archive = zipfile.ZipFile('./data/bgu-rs.zip', 'r')
totalLines = 0
numOffiles = 0
for file in archive.filelist:
    if ("part-" in file.filename and ".csv" in file.filename):
        fileData = archive.read(file.filename)
        numOffiles = numOffiles + 1
        s = str(fileData, 'utf-8')
        data = StringIO(s)
        df = pd.read_csv(data)

        #handle lines from the given file and continue to the next file
        totalLines = totalLines + df.__len__()
        printDebug("lines[" + str(df.__len__()) + "]total[" + str(totalLines) + "]numOffiles[" + str(numOffiles) + "]")
