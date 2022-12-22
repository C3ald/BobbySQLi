import pickle
import gym
from sqli_tester.ai import ModelClass
from ai.envs.Env import BobbySQLEnv
import argparse
import os





def load_model(model_file):
    # Load the model from the specified file
    with open(model_file, 'rb') as f:
        if len(f.readlines()) > 0:
                env, model = pickle.load(f)
                env.model = model
                return env
        else:
                raise Exception('no model')


def save_all(model_file, env):
        with open('data_file.pkl') as f:
                pickle.dump((env, env.model), f)

def initialize():
        """ initializes the backend stuff and creates a model and environment if there is none """
        try:
                load_model("./data_file.pkl")
                print("loaded model")
        except:
                os.system("touch data_file.pkl")
                modelc = ModelClass()
                model = modelc.train()
                env = BobbySQLEnv()
                env.model = model
                save_all(model_file, env)
                print("saved model")
initialize()