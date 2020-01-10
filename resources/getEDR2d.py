
from random import randint
import csv
import codecs

def getEDR(altitude, training_time = None):
    altitude = str(altitude)[0:-2]

    if training_time is None:
        filename = "data/" + altitude + ".dat"
    else:
        filename = "training_data/" + training_time[0] + "/" + altitude + ".dat"
    file = open(filename, 'r')
    file_lines = file.readlines()
    file.close()

    ticker = 0

    for line in file_lines:
        split_line = line.split()

        file_lines[ticker] = split_line
        ticker += 1

    return file_lines
