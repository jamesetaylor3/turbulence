
#NOTICE TO AIRMEN
#Try to make the update sooner. I have it set at one hour beforehand to ensure
#    that we get a response that is accurate. I'm not positive this exists.

#This can either be run on demand (checkForUpdate) or with a timed buffer (timerCheck)

#Figure out why certain downloaded .gif's become messed up

from PIL import Image
from pytz import utc
from datetime import timedelta, datetime as dt
from urllib import urlretrieve as download
from os import system, listdir, popen
from tqdm import tqdm
import sys
import codecs
import csv
import random

altitudes = ["010", "030", "050", "070", "090", "110", "130", "150", "170",
             "190", "210", "230", "250", "270", "290", "310", "330", "350",
             "370", "390", "410", "430", "450"]

url_base = "https://www.aviationweather.gov/data/products/turbulence/"

def convertToEDR(r, g, b):
    costs = []
    rgb2edr = []
    with open('resources/rgb2edr.csv') as File:
        reader = csv.reader(File)
        for row in reader:
            rgb2edr.append(row)
        for row in rgb2edr:
            cost = (float(row[0]) - r) ** 2 + (float(row[1]) - g) ** 2 + (float(row[2]) - b) ** 2
            costs.append(cost)
        min_cost = min(costs)
        edr = float(rgb2edr[costs.index(min_cost)][3])
    #prevent zero EDR and round to two decimals. Make this better
    return "%.2f" % edr if "%2f" % edr != 0.00 else 1.00

def isGoodRGB(r, g, b):
    return not ((r, g, b) in [(0,0,0), (152, 152, 102)])

def isPixValid(pix):
    return (22 < pix[0] < 657) and (106 < pix[1] < 573)

def getRGB(rgb_im, pix):

    num_of_try = 0
    r = g = b = 0
    #Essentially two do-whiles
    #This really needs to be optmized
    while True:
        r, g, b = rgb_im.getpixel(pix)
        if isGoodRGB(r, g, b):
            break
        num_of_try += 1
        extender = num_of_try * 2
        while True:
            pix_x_change = random.randint(-extender, extender)
            pix_y_change = random.randint(-extender, extender)
            pix = (pix[0] + pix_x_change, pix[1] + pix_y_change)
            pix_is_valid = isPixValid(pix)
            if pix_is_valid:
                break
    return r,g,b

def downloadImagesAtHour(hours_back):

    hours_back += 2

    global altitudes
    global url_base

    now = dt.now(utc) - timedelta(hours = hours_back) #Must be universal time

    year = str(now.year)
    month = str(now.month) if len(str(now.month)) == 2 else "0" + str(now.month)
    day = str(now.day) if len(str(now.day)) == 2 else "0" + str(now.day)
    hour = str(now.hour) if len(str(now.hour)) == 2 else "0" + str(now.hour)
    date = year + month + day
    time = date + hour

    system("mkdir training_data/images/" + time)

    for altitude in tqdm(altitudes):
        if altitude == ".DS_Store":
            continue

        url_ext = date + "/" + hour + "/" + date + "_" + hour + "_F00_gtg_" + \
                  altitude + "_total.gif"

        url_total = url_base + url_ext
        filename = "training_data/images/" + time + "/" + altitude + ".gif"
        pngname = "training_data/images/" + time + "/" + altitude + ".png"
        try:
            download(url_total, filename)
            im = Image.open(filename)
            im.save(pngname)
            system("rm " + filename)

            content = popen("cat " + pngname).read()

            if "html" in str(content):
                system("rm -r training_data/images/" + time)
                print "Data at time " + time + " did not work"


        except:
            system("rm -r training_data/images/" + time)

def convertTrainingImagesToDat():
    print ""
    for timeblock in tqdm(listdir('training_data/images')):
        if timeblock == ".DS_Store":
            continue
        content = popen("mkdir training_data/" + timeblock).read()
        if "File exists" in content:
            continue

        for altitude in listdir('training_data/images/' + timeblock):

            if altitude == ".DS_Store":
                continue

            data_matrix = []
            with codecs.open('resources/coord2pix.csv', encoding="utf-8-sig") as File:
                reader = csv.reader(File)
                for row in reader:
                    data_matrix.append(row[0:4])
            imagename = "training_data/images/" + timeblock + "/" + altitude
            im = Image.open(imagename)
            rgb_im = im.convert('RGB')
            datname = "training_data/" + timeblock + "/" + altitude[:-4] + ".dat"
            datfile = open(datname, "w")

            for location in data_matrix:
                pix = (int(location[2]), int(location[3]))
                r, g, b = getRGB(rgb_im, pix)
                edr = convertToEDR(r, g, b)
                line = location[0] + " " + location[1] + " " + str(edr) + "\n"
                datfile.write(line)
            datfile.close()

def downloadCurrentImages():

        global altitudes
        global url_base

        now = dt.now(utc) - timedelta(hours = 2) #Must be universal time

        year = str(now.year)
        month = str(now.month) if len(str(now.month)) == 2 else "0" + str(now.month)
        day = str(now.day) if len(str(now.day)) == 2 else "0" + str(now.day)
        hour = str(now.hour) if len(str(now.hour)) == 2 else "0" + str(now.hour)
        date = year + month + day
        time = date + hour

        for altitude in tqdm(altitudes):
            if altitude == ".DS_Store":
                continue

            url_ext = date + "/" + hour + "/" + date + "_" + hour + "_F00_gtg_" + \
                      altitude + "_total.gif"

            url_total = url_base + url_ext
            filename = "data/images/" + altitude + ".gif"
            pngname = "data/images/" + altitude + ".png"

            print url_total

            download(url_total, filename)

            content = popen("cat " + filename).read()

            if "html" in str(content):
                print("\033[1;31;0m Image not avaible right now. Try again later")
                print("\033[0;37;0m \n")
                sys.exit()

            im = Image.open(filename)
            im.save(pngname)
            system("rm " + filename)



#Finish working on this next. Honestly, its not looking good
def convertCurrentImagesToPreciseDat():
    print ""

    for altitude in listdir('data/images'):

        if altitude in ".DS_Store":
            continue

        LEFT_LONGITUDE = -126.5
        RIGHT_LONGITUDE = -65.3
        TOP_LATITUDE = 50.4
        BOTTOM_LATITUDE = 23.3

        LEFT_PIXEL = 79
        RIGHT_PIXEL = 634
        TOP_PIXEL = 193
        BOTTOM_PIXEL = 533

        LONGITUDE_DELTA = RIGHT_LONGITUDE - LEFT_LONGITUDE
        LATITUDE_DELTA = BOTTOM_LATITUDE - TOP_LATITUDE
        L2R_DELTA = RIGHT_PIXEL - LEFT_PIXEL
        T2B_DELTA = BOTTOM_PIXEL - TOP_PIXEL

        data_matrix = []
        ticker = 0
        for long_depth in xrange(200):
            longitude_fraction = float(LONGITUDE_DELTA) / 200
            l2r_fraction = float(L2R_DELTA) / 200
            for lat_depth in xrange(200):
                latitude_fraction = float(LATITUDE_DELTA) / 200
                t2b_fraction = float(T2B_DELTA) / 200

                new_long = longitude_fraction * long_depth + LEFT_LONGITUDE
                new_lat = latitude_fraction * lat_depth + TOP_LATITUDE
                new_l2r = l2r_fraction * long_depth + LEFT_PIXEL
                new_t2B = t2b_fraction * lat_depth + RIGHT_PIXEL
                data_matrix.append([new_lat, new_long, new_l2r, new_t2B])

        imagename = "data/images/" + altitude
        im = Image.open(imagename)
        rgb_im = im.convert('RGB')
        datname = "data/pd/" + altitude[:-4] + ".dat"
        datfile = open(datname, "w")

        for location in data_matrix:
            pix = (int(location[2]), int(location[3]))
            #print pix
            r, g, b = getRGB(rgb_im, pix)
            edr = convertToEDR(r, g, b)
            line = str(location[0]) + " " + str(location[1]) + " " + str(edr) + "\n"
            datfile.write(line)
        datfile.close()

def convertCurrentImagesToDat():
    print ""

    for altitude in listdir('data/images'):

        if altitude == ".DS_Store":
            continue

        data_matrix = []
        with codecs.open('resources/coord2pix.csv', encoding="utf-8-sig") as File:
            reader = csv.reader(File)
            for row in reader:
                data_matrix.append(row[0:4])
        imagename = "data/images/" + altitude
        im = Image.open(imagename)
        rgb_im = im.convert('RGB')
        datname = "data/" + altitude[:-4] + ".dat"
        datfile = open(datname, "w")

        for location in data_matrix:
            pix = (int(location[2]), int(location[3]))
            r, g, b = getRGB(rgb_im, pix)
            edr = convertToEDR(r, g, b)
            line = location[0] + " " + location[1] + " " + str(edr) + "\n"
            datfile.write(line)
        datfile.close()

def hackyConvert():
    for timeblock in tqdm(listdir('training_data/images')):
        if timeblock == ".DS_Store":
            continue
        for altitude in tqdm(listdir('training_data/images/' + timeblock)):
            if altitude == ".DS_Store":
                continue

            data_matrix = []
            x_start = 180
            y_start = 60
            x_end = 540
            y_end = 640

            x_step = 0

            latitude_start = 49.742232
            longitude_start = -130.011254
            latitude_end = 23.104997
            longitude_end = -67.636193

            while (x_start + x_step < x_end):
                y_step = 0
                while (y_start + y_step < y_end):
                    current_x = x_start + x_step
                    print current_x
                    current_y = y_start + y_step
                    current_lat = (latitude_end - latitude_start) * (current_x / x_end) + latitude_start
                    current_long = (longitude_end - longitude_start) * (current_y / y_end) + longitude_start
                    data_matrix.append([current_lat, current_long, current_y, current_x])
                    y_step += 1
                x_step += 1

            imagename = "training_data/images/" + timeblock + "/" + altitude
            im = Image.open(imagename)
            rgb_im = im.convert('RGB')
            system("mkdir training_data/" + timeblock)
            datname = "training_data/" + timeblock + "/" + altitude[:-4] + ".dat"
            datfile = open(datname, "w")

            for location in data_matrix:
                pix = (int(location[2]), int(location[3]))
                r, g, b = getRGB(rgb_im, pix)
                edr = convertToEDR(r, g, b)
                line = str(location[0]) + " " + str(location[1]) + " " + str(edr) + "\n"
                datfile.write(line)
            datfile.close()


if __name__ == "__main__":
    if sys.argv[1] == "training":
        hours_to_go_back = int(sys.argv[2])
        print "Downloading Images"
        for hours_back in tqdm(xrange(hours_to_go_back)):
            downloadImagesAtHour(hours_back)
        print "Converting to .dat"
        convertTrainingImagesToDat()

    if sys.argv[1] == "current":
        downloadCurrentImages()
        convertCurrentImagesToDat()

    if sys.argv[1] == "hacky":
        hackyConvert()
