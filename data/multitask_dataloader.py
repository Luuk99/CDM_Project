# imports
import numpy as np


def cycle(iterable):
    """
    Function that cycles an iterable.
    """

    while True:
        for x in iterable:
            yield x


class MultiTaskDataloader(object):
    """
    Class that combines multiple dataloaders for multitask training.
    """

    def __init__(self, dataloaders, tau=1.0):
        """
        Initialization function that creates the MultiTaskDataloader.
        Inputs:
            dataloaders - Dictionary of dataloaders to combine
            tau - Value to tweak the sampling weights
        """

        self.dataloaders = dataloaders

        Z = sum(pow(v, tau) for v in self.dataloader_sizes.values())
        self.tasknames, self.sampling_weights = zip(*((k, pow(v, tau) / Z) for k, v in self.dataloader_sizes.items()))
        self.dataiters = {k: cycle(v) for k, v in dataloaders.items()}

        self.train_only_one_task = False
        self.current_taskname = "Circa"

    def set_one_task(self, current_taskname):
        """
        Change the model such that it trains on one specific task
        Inputs:
            current_taskname: taskname on which to train on.
        """
        self.train_only_one_task = True
        self.current_taskname = current_taskname

    @property
    def dataloader_sizes(self):
        """
        Function to get the sizes of the dataloaders
        Outputs:
            self._dataloader_sizes - List of dataloader lengths
        """

        if not hasattr(self, '_dataloader_sizes'):
            self._dataloader_sizes = {k: len(v) for k, v in self.dataloaders.items()}
        return self._dataloader_sizes

    def __len__(self):
        """
        Function that calculates the length of the entire combined dataloader.
        Outputs:
            length - Length of all dataloaders combined
        """

        return sum(v for k, v in self.dataloader_sizes.items())

    def __iter__(self):
        """
        Function that allows for iteration over the MultiTaskDataloader.
        Outputs:
            batch - Batch from a randomly picked dataloader
        """
        for i in range(len(self)):
            if not self.train_only_one_task:
                taskname = np.random.choice(self.tasknames, p=self.sampling_weights)
            else:
                taskname = self.current_taskname
            dataiter = self.dataiters[taskname]
            batch = next(dataiter)

            yield (taskname, self.tasknames.index(taskname), batch)
