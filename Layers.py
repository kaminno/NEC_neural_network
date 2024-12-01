import numpy as np

class LinearLayer():
    def __init__(self, name="", output=False):
        self.name = name
        self.output = output

    # def has_params(self):
    #     return False

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
