# Notes
# Changes in x are changes in latitude
# Changes in y are changes in longitude

# All heading angles are not bering, but rather in tune for unit circle.
# This program calculates in ground speed of miles per hour.
#   Both must be converted prior to input to this program

# Need to create a bunch of test flights. Just choose random locations in USA
#   that are randomized

from collections import namedtuple
import random
import math
import numpy as np
from sys import argv
import resources.flightexamples as flightexamples
from agent import Airplane, ReplayMemory
from environment import World
import hyperparameters as hyp
import plotter
exec("from nets.{}.nn import DQN").format(hyp.NET)

import torch
import torch.optim as optim
import torch.nn.functional as F

import atexit

assert len(argv) == 3, "Not proper argument length! Need both action and " \
    + "output/input DQN name!"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Action = namedtuple('Action', ('heading_change', 'speed_change'))
Velocity = namedtuple('Velocity', ('heading', 'speed'))
Transition = namedtuple('Transition', ('state', 'dest_heading', 'action',
                                       'next_state', 'next_dest_heading',
                                       'reward'))

action_possibilities = [Action(heading_change=hc, speed_change=sc)
                        for hc in (-5,0,5) for sc in (-10,0,10)]
action_possibilities_short = [Action(heading_change=hc, speed_change=0)
                              for hc in (-5,0,5)]


class Size:
    LIGHT = 1
    MEDIUM = 2
    HEAVY = 3

policy_net = DQN().to(device)
target_net = DQN().to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=hyp.LEARNING_RATE)

memory = ReplayMemory(10000)

#losses_graph = plotter.TimeGraph("Losses", 1, save=True)

def exit_handler():
    if argv[1] == "train":
        print "Exiting and saving..."
        saved_name = argv[2]
        torch.save(target_net.state_dict(), "{}_target.net".format(saved_name))
        torch.save(policy_net.state_dict(), "{}_policy.net".format(saved_name))
        file = open("Done.txt", "r")
        lines = file.readlines()
        wins = 0
        for line in lines:
            if "dest" in line:
                win += 1
        print "{}% success rate".format(100*float(wins)/len(lines))
    if argv[2] == "test":
        print "Exiting..."

atexit.register(exit_handler)

def print_weight_norms():
    print "Norm of policy embedding weights: " + str(float(torch.norm(policy_net.emb.weight)))
    print "Norm of target embedding weights: " + str(float(torch.norm(target_net.emb.weight)))
    print "Norm of policy head layer: " + str(float(torch.norm(policy_net.head.weight)))
    print "Norm of target head layer: " + str(float(torch.norm(target_net.head.weight)))


def optimize_model():
    global env
    if len(memory) < hyp.BATCH_SIZE:
        return
    transitions = memory.sample(hyp.BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                  batch.next_state)), device=device,
                                  dtype=torch.uint8)
    print batch.state[0]
    import sys
    sys.exit()
    state_batch = torch.cat(batch.state)
    print(state_batch )
    dest_heading_batch = torch.cat(batch.dest_heading)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    next_dest_heading_batch = torch.cat(batch.next_dest_heading)

    #Find state_action_values and expected_state_action_values
    state_action_values = policy_net(state_batch, dest_heading_batch) \
                          .gather(1, action_batch)

    next_state_values = torch.zeros(hyp.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(next_state_batch,
                                            next_dest_heading_batch) \
                                            .max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values.unsqueeze(1) * \
                                    hyp.GAMMA) + reward_batch

    loss = F.mse_loss(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    if env.timestep % 100 == 0:
        print "Loss:"  + str(loss)
        loss_file = open("../losses.txt", "a")
        loss_file.write(str(float(loss)) + "\n")
        loss_file.close()

    #losses_graph.update(float(loss))
    #losses_graph.plot()


if argv[1] == "train":

    for i_episode in xrange(hyp.NUM_EPISODES):
        print "Episode #{}".format(i_episode)

        trip = flightexamples.training_sample()

        initial_velocity = Velocity(heading=180, speed=400)

        plane = Airplane(Size.LIGHT)
        plane.setVelocity(initial_velocity)
        plane.setTrip(trip)

        env = World(training = True)
        state = env.start(plane)

        map_graph = plotter.MapGraph(fig_id=2)

        for t in env:

            action_index = plane.chooseAction(state, policy_net, env.timestep)
            action = action_possibilities_short[action_index]

            dest_heading = torch.tensor([[plane.gridHeadingDestination()]],
                                        device=device)

            next_state, reward = env.move(state, action)

            if env.done:
                next_state = state
                next_dest_heading = dest_heading
                print(env.timestep)
                break
            else:
                # next_state is already correct
                next_dest_heading = torch.tensor([[
                    plane.gridHeadingDestination()]], device=device)

            # Make sure the environment has the most recent plane
            env.update_plane(plane)

            reward = torch.tensor([[reward]], device=device)
            action_index = torch.tensor([[action_index]], device=device)
            next_dest_heading = torch.tensor([[plane.gridHeadingDestination()]],
                                             device=device)

            memory.push(state, dest_heading, action_index,
                        next_state, next_dest_heading, reward)

            state = next_state

            optimize_model()

            map_graph.update(plane.location)
            map_graph.plot()

            if t % hyp.TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
                #print_weight_norms()


if argv[1] == "test":
    saved_name = argv[2]

    target_net.load_state_dict(torch.load("../savednets/{}.net".format(saved_name)))

    # gets a departure and destination.
    trip = flightexamples.testing_sample()

    initial_velocity = Velocity(heading=180, speed=400)

    plane = Airplane(Size.LIGHT)
    plane.setVelocity(initial_velocity)
    plane.setTrip(trip)

    env = World()
    state = env.start(plane)

    map_graph = plotter.MapGraph(fig_id=2)

    for t in env:
        action_index = plane.chooseAction(state, target_net, 1e10)

        action = action_possibilities_short[action_index]

        dest_heading = torch.tensor([[plane.gridHeadingDestination()]],
                                    device=device)
        next_state, reward = env.move(state, action)

        if env.done:
            print(env.timestep)
            map_graph.update(plane.location)
            map_graph.plot(final=True)
            break

        # Make sure the environment has the most recent plane
        env.update_plane(plane)

        state = next_state

        map_graph.update(plane.location)
        map_graph.plot()
