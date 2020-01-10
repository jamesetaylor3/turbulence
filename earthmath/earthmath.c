#include <Python.h>
#include <math.h>

#define PI 3.14159265
#define EARTH_RADIUS 3960.0

float radians(float angle) {
    return M_PI / 180 * angle;
}

float degrees(float angle) {
    return 180 / M_PI * angle;
}

float changeInLatitude(float miles) {
    return degrees(miles / EARTH_RADIUS);
}

float changeInLongitude(float miles, float latitude) {
    float r = EARTH_RADIUS * cos(radians(latitude));
    return degrees(miles / r);
}

static PyObject * earthmath_distanceTwoPoints(PyObject *self, PyObject *args) {

    float PointALatitude, PointALongitude, PointBLatitude, PointBLongitude;
    float dlon, dlat;
    float a, c;
    float distance;

    if (!PyArg_ParseTuple(args, "ffff", &PointALatitude, &PointALongitude,
                                        &PointBLatitude, &PointBLongitude))
        return NULL;

    dlon = radians(PointBLongitude - PointALongitude);
    dlat = radians(PointBLatitude - PointALatitude);

    a = pow(sin(dlat / 2), 2) + cos(radians(PointALatitude)) * cos(radians(PointBLatitude)) * pow(sin(dlon / 2), 2);
    c = 2 * atan2(sqrt(a), sqrt(1 - a));

    distance = EARTH_RADIUS * c;

    return Py_BuildValue("f", distance);
}

static PyObject * earthmath_findNewLocation(PyObject *self, PyObject *args) {

    float latitude, longitude, heading, miles;
    int altitude, altitude_change;
    float heading_rad, long_miles, lat_miles, delta_long, delta_lat;
    float new_longitude, new_latitude;
    int new_altitude;

    if (!PyArg_ParseTuple(args, "ffifif", &latitude, &longitude, &altitude,
                                          &heading, &altitude_change, &miles))
        return NULL;

    heading_rad = radians(heading);
    long_miles = miles * cos(heading_rad);
    lat_miles = miles * sin(heading_rad);

    delta_long = changeInLongitude(long_miles, latitude);
    delta_lat = changeInLatitude(lat_miles);

    new_longitude = longitude + delta_long;
    new_latitude = latitude + delta_lat;
    new_altitude = altitude + altitude_change;

    return Py_BuildValue("ffi", new_latitude, new_longitude, new_altitude);

}

static PyMethodDef EarthmathMethods[] = {

    {"distanceTwoPoints", earthmath_distanceTwoPoints, METH_VARARGS,
     "find distance between two points"},
    {"findNewLocation", earthmath_findNewLocation, METH_VARARGS,
     "find new location"},

    {NULL, NULL, 0, NULL}        /* Sentinel */
};

DL_EXPORT(void) initearthmath(void) {
  Py_InitModule("earthmath", EarthmathMethods);
}
