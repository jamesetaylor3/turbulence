from collections import namedtuple
import hyperparameters as hyp
import numpy as np
import math
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Action = namedtuple('Action', ('heading_change', 'speed_change'))
Velocity = namedtuple('Velocity', ('heading', 'speed'))
Transition = namedtuple('Transition', ('state', 'heading', 'action',
                                       'next_state', 'next_heading', 'reward'))

class Airplane:
    def __init__(self, size):
        self.size = size
        self.old_location = None
        self.old_velocity = None
        self.current_EDR = None

    def setVelocity(self, velocity):
        self.velocity = velocity

    def setTrip(self, trip):
        self.location = trip["departure"]
        self.destination = trip["destination"]

    def chooseAction(self, state, policy_net, timestep):
        grid_heading_to_destination = torch.tensor([[self.gridHeadingDestination()]], device=device)

        sample = random.random()
        eps_threshold = hyp.EPS_END + (hyp.EPS_START - hyp.EPS_END) * math.exp(
            -1. * timestep / hyp.EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
               q_vals = policy_net(state, grid_heading_to_destination)[0] \
                        .numpy()
               return np.argmax(q_vals)
        else:
            return random.randint(0,2)  # random.randint(0,8)

    def gridHeadingDestination(self):
        d_lat_to_dest = self.destination.latitude - self.location.latitude
        d_long_to_dest = self.destination.longitude - self.location.longitude
        heading_to_destination = math.atan2(d_lat_to_dest, d_long_to_dest)
        #grid_heading_to_destination = math.degrees(heading_to_destination) - \
        #                              self.velocity.heading
        grid_heading_to_destination = heading_to_destination - math.radians(
                                      self.velocity.heading)
        return (grid_heading_to_destination - math.pi) % (2 * math.pi)
        return grid_heading_to_destination

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # Saves a transition
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
