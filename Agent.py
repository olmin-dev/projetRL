from keras.models import Sequential
from keras.layers import Embedding, Reshape
from keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from keras.layers import BatchNormalization
import gym
from gym.envs.registration import registry, register, make, spec
import numpy as np
import itertools as it
import tensorflow as tf
import keras
from keras.models import model_from_json
import matplotlib.pyplot as plt

action_space = [[ -1.0, 0.0, 0.0 ],  [ +1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.7 ], [ 0.0, 0.8, 0.1 ], [ -0.8, 7.0, 0.0 ], [0.8, 0.7, 0.0], [0.8, 0.3, 0.0], [-0.8, 0.3, 0.0], [-0.3, 0.8, 0.0], [0.3, 0.8, 0.0], [0.5, 0.0, 0.5], [-0.5, 0.0, 0.5], [-0.5, 0.3, 0.0], [0.5, 0.3, 0.0], [0.0, 0.0, 0.3], [0.0, 0.3, 0.0]]

register(
    id='CarRacing-v1',
    entry_point='gym.envs.box2d:CarRacing',
    max_episode_steps=1000,
    reward_threshold=900,
)

class CarRacing:
    def __init__(self):
        self.env = gym.make('CarRacing-v1')
        self.env.action_space = action_space
        self.action_space = action_space
        self.observation_space = self.env.observation_space
        self.nb_neg = 0
        self.critical_neg = 30
        self.critical_punition = -50

    def reset(self):
        self.nb_neg = 0
        self.nb_step = 0
        return self.env.reset()

    def step(self, action):
        self.nb_step += 1
        obs, rew, done, info = self.env.step(self.action_space[action])
        if(rew < 0) :
            self.nb_neg += 1
        if(rew> 0 ) :
            self.nb_neg = 0
        if(self.nb_neg >= self.critical_neg) :
            done = True
            rew = self.critical_punition
            self.nb_neg = 0
        return obs, rew, done, info

    def init_test(self):
        self.critical_neg = 1000
        self.critical_punition = 0

    def set_state(self):
        self.env = deepcopy(state)
        obs = np.array(list(self.env.unwrapped.state))
        return obs

    def render(self, mode='human'):
        return self.env.render(mode)

    def get_state(self):
        return deepcopy(self.env)

    def close(self):
        self.env.close()

def stat(name):
    scores = []
    steps = []
    with open(name) as f:
        for i in range(100):
            line = f.readline()
            scores.append(float(line[19:26]))
            steps.append(int(line[35:38]))
    mean_scores = sum(scores)/len(scores)
    mean_steps = sum(steps)/len(steps)

    a = [i for i in range(len(scores))]

    plt.plot(a, scores)
    plt.show()

    print("Mean score:{}, max: {}, min:{}".format(mean_scores, max(scores), min(scores)))
    print("Mean played steps:{}, max: {}, min:{}".format(mean_steps, max(steps), min(steps)))

# stat("stats.txt")

def main():
    env = CarRacing()
    env.reset()

    # load json and create model
    json_file = open('./saved/model2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./saved/model2.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='mse', optimizer='adam')
    model = loaded_model

    # print("State Space {}".format(env.P[331]))

    # model = Sequential()
    # model.add(Reshape((96,96,3),input_shape = (1,96,96,3)))
    # model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, 3)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(216, activation='relu'))
    # model.add(Dense(16, activation=None))
    # model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001, epsilon=1e-8))
    # model.summary()


    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=5000, window_length = 1)
    nb_actions = 16

    dqn = DQNAgent(model = model, memory = memory, nb_actions = nb_actions, nb_steps_warmup=10, target_model_update=1e-5,policy=policy)
    dqn.compile(Adam(lr=0.0001),metrics=['mse'])

    log_interval = 1e4
    for i in range(0):
        print(i, "\n")
        dqn.fit(env,nb_steps=3000, log_interval=log_interval, verbose=1, nb_max_episode_steps=1000, action_repetition=3)
        env.close()

        model_json = model.to_json()
        with open("model2.json", "w") as json_file:
            json_file.write(model_json)
            model.save_weights("model2.h5")
            print("Saved model to disk")

    env.init_test()

    dqn.test(env, nb_episodes=5, visualize=True,nb_max_episode_steps=10000, action_repetition=2)
    env.close()

main()
