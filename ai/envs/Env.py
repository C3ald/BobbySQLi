import gym
import numpy as np
from gym.envs.registration import register

class BobbySQLEnv(gym.Env):
     """ generates the AI model """
     def __init__(self, website, model):
        self.website = website
        self.model = model
        self.current_payload = None
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
    
     def reset(self):
        self.current_payload = self._generate_payload()
        return self._get_observation()
    
     def step(self, action):
        if action == 0:
            success = self.website.inject(self.current_payload)
            reward = 1 if success else -1
        else:
            reward = 0
        done = True
        self.model.update_weights(reward)
        return self._get_observation(), reward, done, {}
    
     def _get_observation(self):
        return [int(self.website.is_vulnerable())]
    
     def _generate_payload(self):
        payload = self.model.gen_payload()
        return payload



class Example_model:
        def __init__(self):
                pass
        def example(self):
                env = gym.make('MountainCar-v0')
                # Observation and Action space
                os_space = env.observation_space
                action_space = env.action_space
                print(os_space)
                print(action_space)

if __name__ == "__main__":
        example = Example_model()
        example.example()