
from collections import namedtuple
from resources.getEDR2d import getEDR
import resources.earthmath as earthmath
import hyperparameters as hyp
import plotter
import math
import torch
import random

#time_diff_graph = plotter.TimeGraph("Time Diff Costs", 3)
#turb_graph = plotter.TimeGraph("Turbulence Costs", 4)
#fuel_graph = plotter.TimeGraph("Fuel Costs", 6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Action = namedtuple('Action', ('heading_change', 'speed_change'))
Velocity = namedtuple('Velocity', ('heading', 'speed'))
Location = namedtuple('Location', ('latitude', 'longitude'))

class Size:
    LIGHT = 1
    MEDIUM = 2
    HEAVY = 3

class World:
    def __init__(self, training = False):
        self.timestep = 0
        self.done = False
        self.EDR_Map = []

    def __iter__(self):
        return self

    def next(self):
        self.timestep += 1
        return self.timestep

    def start(self, plane):
        self.plane = plane
        initial_state = self._generateGrid(plane)
        return initial_state

    def update_plane(self, plane):
        self.plane = plane

    def move(self, state, action):
        plane = self.plane

        new_heading = plane.velocity.heading + action.heading_change
        new_heading = new_heading % 360

        # Must be between MAX_CRUISING_SPEED and 100
        new_speed = max(min(plane.velocity.speed + action.speed_change,
                        hyp.MAX_CRUISING_SPEED), 100)
        new_speed = plane.velocity.speed  # Temporary. jUst to sseeeeeeeeeeeeeeeeeeeee <--------------
        new_velocity = Velocity(heading=new_heading, speed=new_speed)
        miles = new_velocity.speed * hyp.TIMESTEP_LENGTH
        new_latitude, new_longitude, _ = earthmath.findNewLocation(
                                                   plane.location.latitude,
                                                   plane.location.longitude,
                                                   hyp.ALTITUDE,
                                                   new_velocity.heading,
                                                   0, miles)

        plane.old_location = Location(latitude=plane.location.latitude,
                                      longitude=plane.location.longitude)
        plane.old_velocity = plane.velocity
        plane.location = Location(latitude=new_latitude,
                                  longitude=new_longitude)
        plane.velocity = new_velocity

        reward = self._getReward(plane)  # Plane arg is temp

        self.done, final_reward = self._isDone()

        state_p = []
        if self.done:
            reward += final_reward
            state_p = None
        else:
            state_p = self._generateGrid(plane)

        return state_p, reward

    # "Private"
    def _EDR(self, location):

        if self.timestep % 40 == 0:
            self.EDR_Map = getEDR(hyp.ALTITUDE, hyp.DATETIME)

        costs = []
        for spot in self.EDR_Map:
            cost = (float(spot[0]) - location.latitude) ** 2 + \
                   (float(spot[1]) - location.longitude) ** 2
            costs.append(cost)
        min_cost = min(costs)

        edr = float(self.EDR_Map[costs.index(min_cost)][2])

        return edr

    def _isDone(self):
        plane = self.plane
        file = open("Done.txt", "a")
        if hyp.TIMESTEP_LENGTH * self.timestep >= hyp.FUEL_HOURS:
            print "\nPlane ran out of fuel"
            file.write("\nPlane ran out of fuel")
            file.close()
            return True, -300
        if earthmath.distanceTwoPoints(plane.location, plane.destination) < 50:
            print "\nPlane reached destination in %i timesteps" % self.timestep
            file.write("\nPlane reached destination in %i timesteps" % self.timestep)
            file.close()
            return True, 100
        if not ((24.102011 < plane.location.latitude < 49.553788) and
                (-125.0580 < plane.location.longitude < -66.4735)):
            print "\nPlane ran away"
            file.write("\nPlane ran away")
            file.close()
            return True, -300
        return False, None


    # I know there is an issue with this. Please check
    def _generateGrid(self, plane):
        heading_rad = math.radians(plane.velocity.heading)
        MIDDLE = hyp.GRID_DIM / 2 + 1

        box = []
        for x in xrange(hyp.GRID_DIM):  # Latitude
            row = []
            for y in xrange(hyp.GRID_DIM):  # Longitude
                x_factor = hyp.GRID_DIM - MIDDLE - x
                y_factor = y - 1

                angle_delta = 0
                if y_factor is not 0:
                    angle_delta = math.atan2(x_factor, y_factor)
                else:
                    angle_delta = math.pi / 2 if y_factor > 0 \
                                              else -1 * math.pi / 2
                coordinate_angle = heading_rad + angle_delta

                x_shift = x_factor * hyp.BD_COORD * math.sin(coordinate_angle)
                # Long gets smaller L -> R
                y_shift = y_factor * hyp.BD_COORD * \
                    math.cos(coordinate_angle) * -1  # This is ugly

                shifted_location = Location(latitude=plane.location.latitude
                                            + x_shift,
                                            longitude=plane.location.longitude
                                            + y_shift)

                edr = self._EDR(shifted_location)

                if x_factor == y_factor == 0:
                    plane.current_EDR = edr

                row.append(edr)
            box.append(row)

        box = [[box]]

        return torch.tensor(box, device=device)

    def _getReward(self, plane):
        plane = self.plane

        bounds = {
            Size.LIGHT: (0.64, 0.12),
            Size.MEDIUM: (0.79, 0.16),
            Size.HEAVY: (0.96, 0.18)
        } [plane.size]

        turbulence_cost = None
        if plane.current_EDR > bounds[0]:
            m = 1 / (1 - bounds[0])
            b = 100 - m
            turbulence_cost = m * x + b
        elif plane.current_EDR > bounds[1]:
            a = 98 / (math.exp(bounds[0]) - math.exp(bounds[1]))
            b = 1 - a * math.exp(bounds[1])
            turbulence_cost = a * math.exp(plane.current_EDR) + b
        elif plane.current_EDR > (bounds[1] * 0.75):
            m = 1 / (bounds[0] - (bounds[1] * 0.75))
            b = -1 * m * (bounds[1] * 0.75)
            turbulence_cost = m * plane.current_EDR + b
        else:
            turbulence_cost = 0

        # Cost of remaining fuel
        final_time = 13
        final_cost = 100
        initial_cost = 2
        a = (final_cost - initial_cost) / (math.exp(final_time) - 1)
        b = initial_cost - a
        relative_time = final_time * (self.timestep * hyp.TIMESTEP_LENGTH) \
            / hyp.FUEL_HOURS

        fuel_cost = a * math.exp(relative_time) + b

        # ETA Change costs. Hovering too much in the 2's
        # If had made the direct-to path last time
        # This needs to be completely redone. make sure angles are good
        distance_to_dest_from_prev_loc = earthmath.distanceTwoPoints(
            plane.old_location, plane.destination)

        time_to_cover_distance1 = distance_to_dest_from_prev_loc / \
            plane.old_velocity.speed

        # Direct to from where we are now
        distance_to_dest_from_curr_loc = earthmath.distanceTwoPoints(
            plane.location, plane.destination)

        time_to_cover_distance2 = distance_to_dest_from_curr_loc / \
            plane.velocity.speed

        time_difference = (time_to_cover_distance2 - time_to_cover_distance1) \
                          * 60
        c_max = 15

        a = (c_max * math.sqrt(math.e)) / (math.e - 1)
        b = -a / math.sqrt(math.e)
        distance_difference_cost = a * math.e ** time_difference + b

        time_difference_cost = time_difference * 10

        dest_t0 = abs(earthmath.distanceTwoPoints(plane.destination,
            plane.old_location))

        dest_t1 = abs(earthmath.distanceTwoPoints(plane.destination,
            plane.location))

        t1_t0 = abs(earthmath.distanceTwoPoints(plane.location,
            plane.old_location))

        angle = math.acos((math.pow(dest_t0, 2) + math.pow(t1_t0, 2) - \
            math.pow(dest_t1, 2)) / (2 * dest_t0 * t1_t0))

        #time_diff_graph.update(time_difference_cost)
        #time_diff_graph.plot()
        #turb_graph.update(turbulence_cost)
        #turb_graph.plot()
        #fuel_graph.update(fuel_cost)
        #fuel_graph.plot()

        if self.timestep % 100 == 0:
            angle_costs_file = open("../angle.txt", "a")
            turb_costs_file = open("../turb.txt", "a")
            fuel_costs_file = open("../fuel.txt", "a")
            angle_costs_file.write(str(angle) + "\n")
            turb_costs_file.write(str(turbulence_cost) + "\n")
            fuel_costs_file.write(str(fuel_cost) + "\n")
            angle_costs_file.close()
            turb_costs_file.close()
            fuel_costs_file.close()

        overall_cost = turbulence_cost + (fuel_cost) + time_difference_cost * 10
        return -overall_cost
