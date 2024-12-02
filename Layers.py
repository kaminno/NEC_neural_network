import numpy as np

class LinearLayer():
    def __init__(self, name="", output=False):
        self.name = name
        self.output = output

    def forward(self, X):
        """
        feed-forward method of linear layer. Given input vector X, apply linear function to it and return f(X)
        params:
        @X, float[]: vector of (n_neurons)
        return: f(X), in this case f(X)=X
        """
        return X
    
    def derivative(self, h):
        """
        computes derivative of linear activation function f'(x), where f(x)=x
        params:
        @h, float[]: 
        return f'(h)
        """
        return np.ones(h.shape)

class ReLULayer():
    def __init__(self, name=""):
        self.name = name

    def forward(self, X):
        """
        feed-forward method of ReLU layer. Given input vector X, apply Relu function
        params:
        @X, float[]: vector of (n_neurons)
        return: ReLU(X)
        """
        return np.maximum(0, X)
    
    def derivative(self, h):
        """
        computes derivative of ReLU activation function ReLU'(x)
        params:
        @h, float[]: 
        return ReLU'(h)
        """

        return np.where(h > 0, 1, 0)

class SigmoidLayer():
    def __init__(self, name=""):
        self.name = name

    def forward(self, X):
        """
        feed-forward method of sigmoid layer. Given input vector X, apply sigmoid function
        params:
        @X, float[]: vector of (n_neurons)
        return: sigmoid(X)
        """
        return 1 / (1 + np.exp(-X))
    
    def derivative(self, h):
        """
        computes derivative of sigmoid activation function sigma'(x)
        params:
        @h, float[]: 
        return sigma'(h)
        """

        sigma = 1 / (1 + np.exp(-h))
        return sigma * (1 - sigma)

class TanhLayer():
    def __init__(self, name=""):
        self.name = name

    def forward(self, X):
        """
        feed-forward method of tanh layer. Given input vector X, apply tanh() function
        params:
        @X, float[]: vector of (n_neurons)
        return: tanh(X)
        """
        return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
    
    def derivative(self, h):
        """
        computes derivative of tanh activation function tanh'(x)
        params:
        @h, float[]: 
        return tanh'(h)
        """

        tanh = (np.exp(h) - np.exp(-h)) / (np.exp(h) + np.exp(-h))
        return 1 - tanh**2
