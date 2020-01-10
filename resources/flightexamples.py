from collections import namedtuple
import random

Location = namedtuple('Location', ('latitude', 'longitude'))

training_examples = [
    {
        "departure": Location(latitude=40.7127753, longitude=-74.0059728),
        "destination": Location(latitude=35.7795897, longitude=-78.6381787)
    },
    {
        "departure": Location(latitude=39.011902, longitude=-98.484246),
        "destination": Location(latitude=25.761680, longitude=-80.191790)
    },
    {
        "departure": Location(latitude=37.774929, longitude=-122.419416),
        "destination": Location(latitude=29.760427, longitude=-95.369803)
    },
    {
        "departure": Location(latitude=47.551493, longitude=-101.002012),
        "destination": Location(latitude=47.606209, longitude=-122.332071)
    },
    {
        "departure": Location(latitude=40.760779, longitude=-111.891047),
        "destination": Location(latitude=32.354668, longitude=-89.398528)
    },

]

testing_examples = [
    {
        "departure": Location(latitude=40.7127753, longitude=-74.0059728),
        "destination": Location(latitude=35.7795897, longitude=-78.6381787)
    },
]


def training_sample():
    #return random.sample(training_examples, 1)[0]
    return training_examples[0]

def testing_sample():
    #return random.sample(testing_examples, 1)[0]
    return testing_examples[0]
