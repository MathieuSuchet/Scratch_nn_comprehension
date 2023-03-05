import json
import os
import random
from multiprocessing import Pipe

import numpy as np
from keras import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import save_model, load_model

import Listeners
from Environment import Environment
from Listeners import ReplayListener


class Model:
    BATCH_SIZE = 500
    GAMMA = 0.99

    def __init__(self, name, observation_space, action_space, steps: int = 0):

        self.EPISODES = 5000
        self.episodes = 0

        self.replay_listeners = []
        self.steps = steps

        self.name = name
        self.action_space = action_space
        self.state_space = observation_space

        self.nn = Sequential()
        self.memory = []
        self.nn.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.nn.add(Dense(24, activation="relu"))
        self.nn.add(Dense(action_space, activation="softmax"))
        self.nn.compile(loss="mse", optimizer=Adam(learning_rate=0.001))

    def add_listener(self, listener: Listeners.Listener):
        if isinstance(listener, ReplayListener):
            self.replay_listeners.append(listener)

    def predict(self, x):
        return self.nn.predict(x)

    def remember(self, state, action, reward, state_next, terminal):
        self.memory.append([state, action, reward, state_next, terminal])

    def act(self, result, deterministic: bool):
        if deterministic:
            return np.argmax(np.squeeze(result))
        else:
            return random.choices(list(range(self.action_space)), weights=np.squeeze(result), k=1)[0]

    def experience_replay(self):
        if len(self.memory) < self.BATCH_SIZE:
            for listener in self.replay_listeners:
                listener.on_replay_states_fail()
            return

        for listener in self.replay_listeners:
            listener.on_replay_start()

        batch = random.sample(self.memory, self.BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward + self.GAMMA * np.amax(self.nn.predict(np.reshape(state_next, [1, self.state_space]))) * terminal
            # print(f"Reward      : {reward}\n"
            #       f"State       : {state}\n"
            #       f"Action      : {action}\n"
            #       f"Next state  : {state_next}\n"
            #       f"Is terminal : {bool(terminal)}")
            q_values = self.nn.predict(np.reshape(state, [1, self.state_space]))
            q_values[0][action] = q_update
            self.nn.fit(np.reshape(state, [1, self.state_space]), q_values, verbose=0)
            self.steps += 1

        for listener in self.replay_listeners:
            listener.on_replay_end()

    def save(self, path):

        if not os.path.exists(path):
            os.mkdir(path)
            open(f"{path}/assets.json", "x").close()

        with open(f"{path}/assets.json", "w") as f:
            json.dump({
                "steps": self.steps,
                "name": self.name
            }, f)

        save_model(self.nn, f"{path}/{self.name}", save_format="h5")

        print(f"Model saved at {path}")

    @staticmethod
    def load(path):

        with open(f"{path}/assets.json", "r") as f:
            assets = json.load(f)

        steps = assets["steps"]
        name = assets["name"]

        loaded_model = load_model(f"{path}/{name}")
        model = Model(name, loaded_model.input_shape[1], loaded_model.output_shape[1], steps)
        return model

    def run_multiprocesses(self, num_worker=1):
        works, parent_conns, child_conns = [], [], []

        for idx in range(num_worker):
            parent_conn, child_conn = Pipe()
            work = Environment(idx, child_conn, "LunarLander-v2", self.state_space, False)
            print(f"Starting environment {idx}")
            work.start()
            works.append(work)
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)

        states = [[] for _ in range(num_worker)]
        next_states = [[] for _ in range(num_worker)]
        actions = [[] for _ in range(num_worker)]
        rewards = [[] for _ in range(num_worker)]
        dones = [[] for _ in range(num_worker)]
        predictions = [[] for _ in range(num_worker)]
        score = [0 for _ in range(num_worker)]

        state = [[] for _ in range(num_worker)]
        for worker_id, parent_conn in enumerate(parent_conns):
            state[worker_id] = parent_conn.recv()

        while self.episodes < self.EPISODES:
            predictions_list = [self.predict(s.reshape([1, s.shape[0]])) for s in state]
            actions_list = [self.act(s, deterministic=False) for s in predictions_list]

            for worker_id, parent_conn in enumerate(parent_conns):
                parent_conn.send(actions_list[worker_id])
                action_onehot = np.zeros([self.action_space])
                action_onehot[actions_list[worker_id]] = 1
                actions[worker_id].append(actions_list[worker_id])
                predictions[worker_id].append(predictions_list[worker_id])

            for worker_id, parent_conn in enumerate(parent_conns):
                st, action, reward, next_state, done = parent_conn.recv()

                states[worker_id].append(state[worker_id])
                next_states[worker_id].append(next_state)
                rewards[worker_id].append(reward)
                dones[worker_id].append(done)
                state[worker_id] = next_state
                score[worker_id] += reward

                if done:
                    print(f"Episode {self.episodes} ended (Avg {score[worker_id]})")
                    score[worker_id] = 0
                    if self.episodes < self.EPISODES:

                        self.episodes += 1

            for worker_id in range(num_worker):
                if len(states[worker_id]) >= self.BATCH_SIZE:
                    self.memory = list(zip(states[worker_id],
                                      actions[worker_id],
                                      rewards[worker_id],
                                      next_states[worker_id],
                                      dones[worker_id]))
                    print(f"Worker {worker_id}")
                    self.experience_replay()

                    states[worker_id] = []
                    next_states[worker_id] = []
                    actions[worker_id] = []
                    rewards[worker_id] = []
                    dones[worker_id] = []
                    predictions[worker_id] = []

        for work in works:
            work.terminate()
            print('TERMINATED:', work.name)
            work.join()
