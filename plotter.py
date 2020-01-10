import torch
from collections import namedtuple
import hyperparameters as hyp
from resources.getEDR2d import getEDR
import numpy as np
import time
try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib_exists = True
except:
    print "You dont have library matplotlib. Carrying on without plotting."
    matplotlib_exists = False

Location = namedtuple('Location', ('latitude', 'longitude'))

class MapGraph:

    def __init__(self, fig_id, turbulence_overlay=False):
        self.locations = []
        self.fig_id = fig_id
        self.created_EDR_table = False

    def update(self, location):
        pass

    def plot(self, final=False):
        if matplotlib_exists:
            if final:
                self._plotTurbulence()
                time.sleep(10)
            self._plotLocation()
        if final:
            print "Done is starting now"
            time.sleep(10)

    def _plotLocation(self):
        plt.figure(self.fig_id)
        plt.clf()
        latitudes_t = torch.tensor([location.latitude for location in self.locations],
                                   dtype=torch.float)
        longitudes_t = torch.tensor([location.longitude for location in self.locations],
                                    dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.plot(latitudes_t.numpy(), longitudes_t.numpy())
        plt.savefig('../map.png')  # Only call this at end bc its slow
        plt.pause(0.001)

    def _plotTurbulence(self):
        latitudes = [location.latitude for location in self.locations]
        longitudes = [location.longitude for location in self.locations]
        top = float(max(latitudes))
        bottom = float(min(latitudes))
        right = float(max(longitudes))
        left = float(min(longitudes))
        segments = 20
        lat_segment_length = (top - bottom) / segments
        long_segment_length = (right - left) / segments
        turbulence_values = []
        latitudes = []
        longitudes = []
        for lat_segment in xrange(segments):
            for long_segment in xrange(segments):
                latitudes.append(lat_segment * lat_segment_length + bottom)
                longitudes.append(long_segment * long_segment_length + left)
                EDR = self._EDR(Location(latitude=latitudes[-1],
                                         longitude=longitudes[-1]))
                turbulence_values.append(EDR / 0.3)
        plt.scatter(np.array(latitudes), np.array(longitudes),
                    c = turbulence_values)


    def _EDR(self, location):
        if not self.created_EDR_table:
            self.EDR_Map = getEDR(hyp.ALTITUDE, hyp.DATETIME)

        costs = []
        for spot in self.EDR_Map:
            cost = (float(spot[0]) - location.latitude) ** 2 + \
                   (float(spot[1]) - location.longitude) ** 2
            costs.append(cost)
        min_cost = min(costs)

        edr = float(self.EDR_Map[costs.index(min_cost)][2])

        return edr

class TimeGraph:

    def __init__(self, name, fig_id, sma = False, save = False):
        self.data = []
        self.name = name
        self.fig_id = fig_id
        self.sma = sma
        self.save = save

    def update(self, new_data):
        self.data.append(new_data)

    def plot(self):
        if matplotlib_exists:
            self._plot()

    def _plot(self):
        plt.figure(self.fig_id)
        plt.clf()

        items_t = torch.tensor(self.data, dtype=torch.float)

        plt.title('Training...')
        plt.xlabel('Timestep')
        plt.ylabel(self.name)

        plt.plot(items_t.numpy())
        if (len(items_t) >= 100) and self.sma:
            means = items_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        if self.save:
            plt.savefig('../{}.png'.format(self.name))  # Only call that at end bc its slow

        plt.pause(0.001)  # pause a bit so that plots are updated
