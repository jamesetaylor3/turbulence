# Derived from: https://www.johndcook.com/blog/2009/04/27/converting-miles-to-degrees-longitude-or-latitude/
# This assumes flat earth--unfortunately, we dont have a flat earth

import math

earth_radius = 3960.0

class accepts(object):
    def __init__(self, *args):
        self.acceptable_types = args

    def __call__(self, func):
        def wrapper(*args):
            for acceptable, actual in zip(self.acceptable_types, args):
                if type(acceptable) == list:
                    assert type(actual) in acceptable, "Wrong types passed"
                else:
                    assert type(actual) == acceptable, "Wrong types passed"
            return func(*args)
        return wrapper

@accepts(float)
def _changeInLatitude(miles):
    return math.degrees(miles/earth_radius)

@accepts(float, float)
def _changeInLongitude(miles, latitude):
    r = earth_radius * math.cos(math.radians(latitude))
    return math.degrees(miles / r)

@accepts(float, float, int, int, int, float)
def findNewLocation(latitude, longitude, altitude, heading, altitude_change, miles):
    heading_rad = math.radians(heading)
    long_miles = miles * math.cos(heading_rad)
    lat_miles = miles * math.sin(heading_rad)

    delta_long = _changeInLongitude(long_miles, latitude)
    delta_lat = _changeInLatitude(lat_miles)

    new_longitude = longitude + delta_long
    new_latitude = latitude + delta_lat

    new_altitude = altitude + altitude_change

    return new_latitude, new_longitude, new_altitude

def distanceTwoPoints(PointA, PointB):

    dlon = math.radians(PointB.longitude - PointA.longitude)
    dlat = math.radians(PointB.latitude - PointA.latitude)

    a = math.sin(dlat / 2)**2 + math.cos(math.radians(PointA.latitude)) * math.cos(math.radians(PointB.latitude)) * \
        math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = earth_radius * c

    return distance
