#Notes
#Changes in x are changes in latitude
#Changes in y are changes in longitude
#Changes in z are changes in altitude

#All heading angles are not bering, but rather in tune for unit circle.
#This program calculates in ground speed of miles per hour.
#   Both must be converted prior to input to this program

#Things to get done
#Need destination and departure coord to understand the math.cost of going out of the way
#Make it based on miles to standardized everything

#Dont neccisarily need to do reward each timestep. Perhaps only every 5 or so?
#How do you attribute 30 actions to one reward
#Combine them?

'''I am afraid that this is going to focus too much on getting the plane there rather than learning to avoid turbulence'''
# Quote of the day

from resources.getEDR3d import getEDR
from collections import namedtuple
from itertools import count
import random
import math
import numpy as np
import earthmath

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm

GRID_DIM = 9
#Degrees in latitude/longitude/altitude per box (BOXDELTA)
#BD COORD really needs to be tweaked (try to make it in a certain amount of miles (easier to understand))
BD_COORD = 0.5
BD_ALTITUDE = 2000

FUEL_HOURS = 6
MAX_CRUISING_SPEED = 575

TIMESTEP_LENGTH = 1 / 120.0 #Duration (in hours) of each timestep
REWARD_STEP = 1 / 4.0

NUM_EPISODES = 50

if (GRID_DIM % 2 == 0) or (GRID_DIM < 3):
    raise Exception("GRID_DIM must be an odd integer greater than one!")

Action = namedtuple('Action', ('heading_change', 'altitude_change', 'speed_change'))
Velocity = namedtuple('Velocity', ('heading', 'speed'))
Location = namedtuple('Location', ('altitude', 'latitude', 'longitude'))
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Size:
    LIGHT = 1
    MEDIUM = 2
    HEAVY = 3

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        #Saves a transition
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    #Stochastic gradient descent
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    #This is completely random
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)
    def forward(self, x):
        pass

class Airplane:
    def __init__(self, size):
        self.size = size
        self.old_location = ""
        self.old_velocity = ""

    def setVelocity(self, velocity):
        self.velocity = velocity

    def setTrip(self, origin, destination):
        self.location = origin
        self.current_EDR = getEDR(origin)
        self.destination = destination

    def chooseAction(self, state, env):
        #Obviously not going to be random actions in final. This is simply for demo purposes
        random_heading_change = float([-5, 0, 5][random.randint(-1,1)])
        random_altitude_change = 0
        random_speed_change = float([-10, 0, 10][random.randint(-1,1)])
        return Action(heading_change = random_heading_change,
                      altitude_change = random_altitude_change,
                      speed_change = random_speed_change)

class World:
    def __init__(self):
        self.timestep = 0
        self.EDR_Map = []

    def start(self, plane):
        initial_state = self._generateGrid(plane)
        return initial_state

    def move(self, state, action):
        #global TIMESTEP_LENGTH
        global plane

        new_heading = plane.velocity.heading + action.heading_change
        new_speed = max(min(plane.velocity.speed + action.speed_change,
                        MAX_CRUISING_SPEED), 100) #Must be between MAX_CRUISING_SPEED and 100
        new_velocity = Velocity(heading = new_heading, speed = new_speed)
        miles = new_velocity.speed * TIMESTEP_LENGTH
        new_latitude, new_longitude, new_altitude = earthmath.findNewLocation(plane.location.latitude,
                                                    plane.location.longitude,
                                                    plane.location.altitude,
                                                    new_velocity.heading,
                                                    action.altitude_change, miles)

        plane.old_location = plane.location
        plane.old_velocity = plane.velocity
        plane.location = Location(latitude=new_latitude, longitude=new_longitude, altitude=new_altitude)
        plane.velocity = new_velocity
        state_p = self._generateGrid(plane)
        self.timestep += 1

        if (TIMESTEP_LENGTH * self.timestep) % REWARD_STEP == 0: reward = self._getReward() #only get reward every REWARD_STEP hours
        else: reward = None

        done, final_reward = self._isDone()

        if done: reward = int(reward or 0) + final_reward

        return state_p, reward, done

    # "Private"
    def _isDone(self):
        global plane
        if TIMESTEP_LENGTH * self.timestep >= FUEL_HOURS:
            print "Plane ran out of fuel"
            return True, -200
        if earthmath.distanceTwoPoints(plane.location, plane.destination) < 50:
            print "Plane reached destination"
            return True, 100
        if not ((24.102011 < plane.location.latitude < 49.553788) and \
                (-125.0580 < plane.location.longitude < -66.4735)):
            print "Plane ran away"
            return True, -200
        return False, None

    def _EDR(self, location):

        return getEDR(location)

    #Maybe artificially inflate EDR values to discourage certain behavior
    #EDR at altitudes < 1000 and > 45000, get EDR = 1
    #Examine this thouroughly to find any problems. Its controlled with many HYPERPARAMETERS so we are probably aight
    def _generateGrid(self, plane):
        heading_rad = math.radians(plane.velocity.heading)
        MIDDLE = GRID_DIM / 2 + 1
        grid_total = []
        for z in xrange(GRID_DIM): #Altitude
            box = []
            for x in xrange(GRID_DIM): #Latitude
                row = []
                for y in xrange(GRID_DIM): #Longitude
                    x_factor = GRID_DIM - MIDDLE - x
                    y_factor = y - 1
                    z_factor = MIDDLE - GRID_DIM + z

                    angle_delta = 0
                    if not y_factor is 0:
                        angle_delta = math.atan2(x_factor, y_factor)
                    else:
                        angle_delta = math.pi / 2 if y_factor > 0 else -1 * math.pi / 2

                    coordinate_angle = heading_rad + angle_delta

                    x_shift = x_factor * BD_COORD * math.sin(coordinate_angle)
                    y_shift = y_factor * BD_COORD * math.cos(coordinate_angle) * -1 #Long gets smaller L -> R
                    z_shift = z_factor * BD_ALTITUDE

                    shifted_location = Location(altitude = plane.location.altitude + z_shift,
                                                latitude = plane.location.latitude + x_shift,
                                                longitude = plane.location.longitude + y_shift)

                    edr = self._EDR(shifted_location) if 1000 <= shifted_location.altitude \
                        <= 45000 else 1
                    if x_factor == y_factor == z_factor == 0: plane.current_EDR = edr
                    row.append(edr)
                box.append(row)
            grid_total.append(box)
        return grid_total

    def _getReward(self):
        global plane

        bound_options = {
            Size.LIGHT : (0.64, 0.36, 0.17, 0.12),
            Size.MEDIUM : (0.79, 0.44, 0.20, 0.16),
            Size.HEAVY : (0.96, 0.54, 0.24, 0.18)
        }
        bounds = bound_options[plane.size]

        #This may need some severe revamping. dont really know yet though
        def EXTREME_TURBULENCE():
            return 100
        def SEVERE_TURBULENCE(bounds):
            bounds_range = bounds[0] - bounds[1]
            percent_in = (float(plane.current_EDR) - bounds[1]) / bounds_range
            return 33.333 * (1 + percent_in)
        def MODERATE_TURBULENCE(bounds):
            bounds_range = bounds[1] - bounds[2]
            percent_in = (float(plane.current_EDR) - bounds[2]) / bounds_range
            return 16.666 * (1 + percent_in)
        def LIGHT_TURBULENCE(bounds):
            bounds_range = bounds[2] - bounds[3]
            percent_in = (float(plane.current_EDR) - bounds[3]) / bounds_range
            return 8.333 * (1 + percent_in)
        def NO_TURBULENCE():
            return 0

        if bounds[0] < float(plane.current_EDR): turbulence_cost = EXTREME_TURBULENCE()
        elif bounds[1] < float(plane.current_EDR) <= bounds[0]: turbulence_cost = SEVERE_TURBULENCE(bounds)
        elif bounds[2] < float(plane.current_EDR) <= bounds[1]: turbulence_cost = MODERATE_TURBULENCE(bounds)
        elif bounds[3] < float(plane.current_EDR) <= bounds[2]: turbulence_cost = LIGHT_TURBULENCE(bounds)
        else: turbulence_cost = NO_TURBULENCE()

        #Cost of remaining fuel
        final_time = 13
        final_cost = 100
        initial_cost = 2
        a = (final_cost - initial_cost) / (math.exp(final_time) - 1)
        b = initial_cost - a
        relative_time = final_time * (self.timestep * TIMESTEP_LENGTH) / FUEL_HOURS
        fuel_cost = a * math.exp(relative_time) + b

        #If had made the direct-to path last time
        distance_to_dest_from_prev_loc = earthmath.distanceTwoPoints(plane.old_location.latitude, plane.old_location.longitude,
                                                                     plane.destination.latitude, plane.destination.longitude)
        rewardstep_miles = REWARD_STEP * plane.old_velocity.speed
        distance_left_to_cover = distance_to_dest_from_prev_loc - rewardstep_miles
        time_to_cover_distance1 = distance_left_to_cover / plane.old_velocity.speed

        #Direct to from where we are now
        distance_to_dest_from_curr_loc = earthmath.distanceTwoPoints(plane.location.latitude, plane.location.longitude,
                                                                     plane.destination.latitude, plane.destination.longitude)
        time_to_cover_distance2 = distance_to_dest_from_prev_loc / plane.velocity.speed

        time_difference = time_to_cover_distance2 - time_to_cover_distance1
        distance_difference_cost = pow(10, time_difference / 4)  #This needs to change no doubt about it
        #Think about it: if we have super small rewardstep sizes then this will always be super small.
        #Thus the whole idea of penalizing behavior super that moves in wrong direction will be lost

        final_cost = turbulence_cost + fuel_cost + distance_difference_cost
        return -final_cost

def averageAction(actions):
    def Average(lst):
        return sum(lst) / len(lst)

    heading_changes = [(lambda action: action.heading_change)(action) for action in actions]
    altitude_changes = [(lambda action: action.altitude_change)(action) for action in actions]
    speed_changes = [(lambda action: action.speed_change)(action) for action in actions]

    average_heading_change = Average(heading_changes)
    average_altitude_change = Average(altitude_changes)
    average_speed_change = Average(speed_changes)

    average_action = Action(heading_change = average_heading_change,
                            altitude_change = average_altitude_change,
                            speed_change = average_speed_change)

    return average_action

if __name__ == "__main__":

    memory = ReplayMemory(10000)

    for i_episode in tqdm(xrange(NUM_EPISODES)):
        initial_location = Location(altitude=31000, latitude=33.572272, longitude=-89.603125)
        initial_velocity = Velocity(heading=0, speed=400)

        destination = Location(altitude=31000, latitude=46.734822, longitude=-121.946875)

        plane = Airplane(Size.LIGHT)
        plane.setVelocity(initial_velocity)
        plane.setTrip(initial_location, destination)

        env = World()

        state = env.start(plane)

        rewards = []
        actions = []
        for t in count():
            #print "Timestep:", env.timestep
            #print plane.location
            #print plane.velocity
            #print "EDR:", plane.current_EDR
            #print ""

            action = plane.chooseAction(state, env)
            next_state, reward, done = env.move(state, action)

            actions.append(action)

            if reward is not None:
                rewards.append(reward)
                #print "Reward:", reward
                #print "Total_reward:", sum(rewards)
                average_action = averageAction(actions)
                memory.push(state, average_action, next_state, reward)
                actions = []

            state = next_state
            #print ""

            if done:
                break

    print "Transitions"
    print [transition for transition in memory.memory]
