from multiprocessing import Process
import gymnasium as gym
import numpy as np


class Environment(Process):
    def __init__(self, env_id, child_conn, name, state_space, render=False):
        super(Environment, self).__init__()
        self.env_id = env_id
        self.child_conn = child_conn
        self.state_space = state_space
        self.render = render
        self.env = gym.make(name, render_mode="human" if render else None)

    def run(self) -> None:
        super(Environment, self).run()
        print("Environment started")
        state, _ = self.env.reset()
        state = np.reshape(state, [1, self.state_space]).squeeze()
        if self.child_conn:
            self.child_conn.send(state)
        while True:
            if self.child_conn:
                action = self.child_conn.recv()
            else:
                action = self.model.act(state, True)
            if self.render:
                self.env.render()

            state_next, reward, terminal, info, _ = self.env.step(action)
            state_next = np.reshape(state_next, [1, self.state_space]).squeeze()
            reward = reward if not terminal else -reward

            if terminal:
                state, _ = self.env.reset()
                state = np.reshape(state, [1, self.state_space]).squeeze()

            if self.child_conn:
                self.child_conn.send([state, action, reward, state_next, terminal])

