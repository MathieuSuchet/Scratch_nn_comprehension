import matplotlib.animation
import matplotlib.pyplot as plt

from Listeners import StatsListener


class GraphMaker:

    def __init__(self):
        self.datas = []
        self.listeners = []

    def start(self):
        plt.ion()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid()

        for data in self.datas:
            ax.scatter(data.x, data.y, label=data.label)

        self.graph = ax
        self.legend = self.graph.legend([data.label for data in self.datas])
        self.fig = fig

        anim = matplotlib.animation.FuncAnimation(
            self.fig,
            self.update,
            fargs=([data.x for data in self.datas], [data.y for data in self.datas]),
            interval=1000)
        plt.show()

    def add_data(self, data):
        self.datas.append(data)
        data.listener.graph_maker = self
        self.listeners.append(data.listener)

    def update(self, i, x,y):

        self.graph.clear()
        for data, ydata in zip(x,y):

            new_x = data[-1] + 1
            new_y = new_x ** 2 + new_x - 2

            data.append(new_x)
            ydata.append(new_y)
            self.graph.plot(x, y)

        # Format plot
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.30)
        plt.title('Data')


class Data:
    def __init__(self, label):
        self.x = []
        self.y = []
        self.label = label
        self.listener = StatsListener(None, self)

    def add(self, x, y):
        self.x.append(x)
        self.y.append(y)

        # if self.listener:
        # self.listener.graph_maker.update()
