# -*- coding: utf-8 -*-

""" Delta project
 Creates an agent which interacts with the microgridRLsimulator.
 The class MicrogridEnv represents the simulator

Usage:
    delta [options] <protocol>

where
    <protocol> is the name of the JSON file describing the parameters of the microgridRLsimulator and the agent.
    The protocol JSON file is available in delta/protocols/<protocol>.json
    The parameters set in the JSON file can be overwritten by the options below, specified in the command line.

Options:
    -h                          Display this help.
    -o PATH                     Output path.
    --log_level LEVEL           Set logging level (debug, info, warning, critical) [Default: info].
    --log PATH                  Dump the log into a file
"""

from microgridRLsimulator.gym_wrapper import MicrogridEnv
import os
from docopt import docopt
import logging
from dateutil.parser import isoparse
import json
import importlib.util
import numpy as np
from shutil import copyfile
from datetime import datetime


class Experiment(object):
    """
    This class makes an experiment and an agent from a protocol
    """

    def __init__(self, protocol_exp):
        # the agent and environment's parameters are set in the protocol
        self.protocol = protocol_exp

        self.env_train, self.env_simulate = self.get_environment()
        self.agent = self.get_agent()

    def get_environment(self):
        """
        :return: the environment with parameters specified in the protocol
        """

        # parse the dates into a suitable form
        env_param = self.protocol["env"]
        train_start_date = isoparse(env_param['train_from_date'])
        train_end_date = isoparse(env_param['train_to_date'])
        test_start_date = isoparse(env_param['test_from_date'])
        test_end_date = isoparse(env_param['test_to_date'])

        return MicrogridEnv(train_start_date, train_end_date, env_param["name"]), \
            MicrogridEnv(test_start_date, test_end_date, env_param["name"])

    def get_agent(self):
        """
        :return: the agent with parameters specified in the protocol
        """
        module = importlib.import_module(self.protocol["agent"]["module"].replace('/', '.'))
        class_name = self.protocol["agent"]["class_name"]
        class_agent = getattr(module, class_name)

        agent_options = self.protocol["agent"]["parameters"]

        # if "parameters" in self.protocol["agent"]:
        #     with open(self.protocol["agent"]["parameters"], 'rb') as jsonFileParam:
        #         agent_options = json.load(jsonFileParam)[self.protocol["agent"]["class_name"]]
        # else:
        #     agent_options = {}

        return class_agent(self.env_train, **agent_options)

    def run(self):
        # Create results/ folder (if needed)
        results_folder = "results/results_%s_%s" % (self.protocol["env"]["name"],
                                                    datetime.now().strftime('%Y-%m-%d_%H%M%S'))
        os.makedirs(results_folder)

        # loop on the seed to simulate the agent
        for seed in self.protocol["agent"]["seeds"]:  # Recall that the environment is deterministic
            #self.env_train.random.seed(seed) if the env becomes stochastic
            #self.env_simulate.random.seed(seed)
            self.env_train.action_space.np_random.seed(seed)
            self.env_simulate.action_space.np_random.seed(seed)

            # set the training environment
            self.agent.set_environment(self.env_train)

            # first, train the agent
            self.agent.train_agent()

            # set the simulate environment and test the agent
            self.agent.set_environment(self.env_simulate)
            self.agent.simulate_agent()

            # store and plot the results in the right directory
            os.makedirs(results_folder + "/seed_" + str(seed))
            self.env_simulate.simulator.store_and_plot(folder=results_folder + "/seed_" + str(seed), learning_results=None)

        # copy the protocol file in the results file
        copyfile(self.protocol["path"], results_folder + "/protocol.json")


if __name__ == '__main__':
    # Parse command line arguments
    args = docopt(__doc__)

    # Set logging level
    numeric_level = getattr(logging, args['--log_level'].upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args['--log_level'])
    logging.basicConfig(level=numeric_level, filename=args['--log'])

    # Create logger
    logger = logging.getLogger(__name__)  #: Logger.

    # Get the protocol info
    protocol_path = 'delta/protocols/' + args['<protocol>'] + '.json'
    with open(protocol_path, 'rb') as jsonFile:
        protocol = json.load(jsonFile)
        protocol.update({"path": protocol_path})

    # Create an experiment
    experiment = Experiment(protocol)

    # Run the experiment : train and simulate the agent and store the results
    experiment.run()

"""

todo : incorporate the following code and update the docopt 

# Configure the agent (--agent case)
if args['--agent_file'] is None:
    if isinstance(args['--agent'], str):
        try:
            agent_type = AGENT_TYPES[args['--agent']]
        except KeyError:
            logger.error('Controller "%s" switch to the default controller (%s).' % (
                args['--agent'], DEFAULT_CONTROLLER.__qualname__))
            agent_type = DEFAULT_CONTROLLER

# Configure the agent (--agent_file case)
else:
    try:
        agent_mod = importlib.import_module(args['--agent_file'].rsplit('.', 1)[0].replace('/', '.'))
        agent_type = agent_mod.agent_type
    except:
        logger.error('Controller "%s" switch to the default controller (%s).' % (
            args['--agent_file'], DEFAULT_CONTROLLER.__qualname__))
        agent_type = DEFAULT_CONTROLLER

# Instantiate the agent
if args['--agent_options'] is not None:
    with open(args['--agent_options'], 'rb') as jsonFile:
        agent_options = json.load(jsonFile)[args['--agent']]
else:
    agent_options = {}

"""
