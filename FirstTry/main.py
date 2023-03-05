import time
from threading import Thread

import numpy as np

import Graphs

if __name__ == "__main__":
    def cartpole():
        import gymnasium as gym
        from Listeners import ReplayListener
        from Model import Model
        env = gym.make("LunarLander-v2", render_mode="human")
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        # try:
        #     model = Model.load("models/Astra")
        #     print(f"Model found : {model.steps} steps")
        # except IOError:
        model = Model("Astra_Lander", observation_space, action_space)
        print("Creating a new model")

        listener = ReplayListener()
        model.add_listener(listener)

        try:
            t = Thread(target=show_instance, args=(env, model))
            t.start()
            model.run_multiprocesses(5)
            model.save(f"models/{model.name}")
            t.join()
            print("Saving model...")

        except KeyboardInterrupt:
            print("Saving model...")
            model.save(f"models/{model.name}")


    def progressive_graph():
        graph = Graphs.GraphMaker()

        data = Graphs.Data("Data #1")
        graph.add_data(data)

        for i in range(0, 10):
            data.add(i, i ** 2 + i - 2)

        graph.start()



    #     import matplotlib.pyplot as plt
    #     min = 0
    #     max = 10
    #
    #     x = range(-50, -40)
    #     y = [2 * j ** 3 - j ** 2 + 4 * j - 2 for j in x]
    #     y2 = [3 * j ** 3 - j ** 2 + 4 * j - 2 for j in x]
    #
    #     plt.ion()
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.grid()
    #     # plt.plot(x, y, label="Binomial curve")
    #     line1 = ax.scatter(x, y, color='green', label='Observed')
    #     line2 = ax.scatter(x, y2, color='red', label='Predicted')
    #
    #     ax.legend()
    #
    #     while True:
    #         min += 1
    #         max += 1
    #
    #         x = range(x.start, x.stop + 1, x.step)
    #         y = [2 * j ** 3 - j ** 2 + 4 * j - 2 for j in x]
    #         y2 = [3 * j ** 3 - j ** 2 + 4 * j - 2 for j in x]
    #         ax.grid()
    #
    #         line1 = ax.scatter(x, y, color='green', label='Observed')
    #         line2 = ax.scatter(x, y2, color='red', label='Predicted')
    #
    #         line1.set_antialiased(True)
    #         line2.set_antialiased(True)
    #
    #         fig.canvas.draw()
    #         fig.canvas.flush_events()
    #         fig.show()
    #         plt.cla()


    def show_instance(env, model):

        observation_space = env.observation_space.shape[0]

        while True:
            state = env.reset()[0]
            state = np.reshape(state, [1, observation_space])
            while True:
                action = model.act(model.predict(state), deterministic=False)
                state_next, reward, terminal, info, _ = env.step(action)

                if terminal:
                    break

    cartpole()