import keras
import matplotlib.pyplot as plt
from drawnow import drawnow

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.counter=0

        self.fig = plt.figure()
        self.logs = []

    def __init__(self, slowlyCutBeginning=True):
        self.slowlyCutBeginning=slowlyCutBeginning

    def paintPlot(self):
        plt.plot(self.x, self.acc, label="acc")
        plt.legend()
        self.fig.savefig("acc_history.png")
       # plt.show();

    def on_epoch_end(self, epoch, logs={}):
        self.counter+=1
        if self.slowlyCutBeginning and self.counter%10==0:
            self.logs=self.logs[1:]
            self.x=self.x[1:]
            self.acc=self.acc[1:]
        self.logs.append(logs)
        self.x.append(self.i)
        self.acc.append(logs.get('acc'))
        self.i += 1
        drawnow(self.paintPlot)

