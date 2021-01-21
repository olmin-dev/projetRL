from keras.models import Sequential
from keras.layers import Embedding, Reshape
from keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from keras.layers import BatchNormalization
import gym
import numpy as np
import itertools as it

action_space = [[ -1.0, 0.0, 0.0 ],  [ +1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.8 ], [ 0.0, 1.0, 0.8 ], [ 0.0, 0.0, 0.0 ]]
class CarRacing:
    def __init__(self):
        print("YOUPIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
        self.env = gym.make('CarRacing-v0')
        self.env.action_space = action_space
        self.action_space = action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info = self.env.step(self.action_space[action])
        return obs, rew, done, info

    def set_state(self):
        self.env = deepcopy(state)
        obs = np.array(list(self.env.unwrapped.state))
        return obs

    def get_state(self):
        return deepcopy(self.env)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

env = CarRacing()
env.reset()



# print("State Space {}".format(env.P[331]))

# model = Sequential(name='rvoum')
# model.add(Reshape((96,96,3),input_shape = (1,96,96,3)))
# model.add(Conv2D(filters=32, kernel_size=(3, 3), strides = 3,activation="relu", input_shape=(1,96,96,3)))
# model.add(Flatten())
# model.add(Dense(3,activation="selu"))
# #model.add(Embedding(500,6,input_length = 1, name='Embedding'))
# model.add(Reshape((3,),name='Reshape'))
model = Sequential()
model.add(Reshape((96,96,3),input_shape = (1,96,96,3)))
model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(216, activation='relu'))
model.add(Dense(5, activation=None))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.01, epsilon=1e-7))
model.summary()


policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=5000, window_length = 1)
nb_actions = 5

dqn = DQNAgent(model = model, memory = memory, nb_actions = nb_actions, nb_steps_warmup=500, target_model_update=1e-2,policy=policy)
dqn.compile(Adam(lr=1e-3),metrics=['mae'])

nb_steps = 1e6
log_interval = 1e5
for i in range(int(nb_steps/30000)) :
    dqn.fit(env,nb_steps=30000, log_interval=log_interval, verbose=1,nb_max_episode_steps=99)
    env.close()

dqn.test(env,nb_episode=5, visualize=True, nb_max_episode_steps=99)
for i in range(100000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)



env.close()
