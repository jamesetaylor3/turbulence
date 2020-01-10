
from random import randint
import csv
import codecs

def getEDR(location, backup_alitude = None):
    altitude = ""
    if backup_alitude == None:
        altitude = str(location.altitude)[0:-2] if len(str(location.altitude)) == 5 else "0" + str(location.altitude)[0:-2]
    else:
        altitude = str(backup_alitude)[0:-2]
    latitude = str(location.latitude)
    longitude = str(location.longitude)

    filename = "data/" + altitude + ".dat"

    file = open(filename, 'r')
    file_lines = file.readlines()
    file.close()

    costs = []
    ticker = 0

    for line in file_lines:
        split_line = line.split()
        cost = (float(split_line[0]) - float(latitude)) ** 2 + (float(split_line[1]) - float(longitude)) ** 2
        costs.append(cost)
        file_lines[ticker] = split_line
        ticker += 1
    min_cost = min(costs)
    edr = float(file_lines[costs.index(min_cost)][2])

    return edr

if __name__ == "__main__":
    from collections import namedtuple
    Location = namedtuple('Location', ('altitude', 'latitude', 'longitude'))
    location = Location(altitude = 21000, latitude=33.645088, longitude=-90.482031)
    print getEDR(location)
