import Graphs


class Listener:
    pass


class ReplayListener(Listener):
    def on_replay_end(self):
        print("Replay ended")

    def on_replay_start(self):
        print("Replay started")

    def on_replay_states_fail(self):
        print("Replay doesn't have enough states")


class StatsListener(Listener):
    def __init__(self, graph_maker, data):
        self.graph_maker = graph_maker
        self.data = data
