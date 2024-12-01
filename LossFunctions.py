import numpy as np

class MSELoss(object):
    def __init__(self, name=""):
        self.name = name

    def forward(self, pred, label):
        """
        Forward message of minimum squared error loss-function.
        params:
        @pred: predicted value
        @label: true value
        return: square of difference between 'pred' and 'label' values
        """        
        return (pred - label)**2


    def derivative(self, pred, label):
        """
        Derivative of MSE loss-function.
        params:
        @pred: predicted value
        @label: true value
        return: derivative of MSE in the point (pred-label)
        """

        return 2*(pred - label)
