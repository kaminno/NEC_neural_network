import numpy as np
import Layers
import LossFunctions
from sklearn.model_selection import train_test_split

class NeuralNet:
	def __init__(self, layers, epochs, learning_rate, momentum, fact, validation):
		"""
		params:
		@layers, int[]: array (length == number of layers) of ints (numbers of neurons in each layer)
				e.g. [3, 3, 1] corresponds to three layers of three, three and one neuron
		@epochs, int: number of epochs
		@learning_rate, double from [0, 1]: gradient descent step size
		@momentum, double from [0, 1]: 
		@fact, stirng: name of activation function, one for all layers. Choose from []"linear", "relu", "sigmoid", "tanh"]
		@validation, double from [0, 1]: 
		"""
		self.L = len(layers)				# number of layers
		self.n = layers.copy()				# an array with the number of units in each layer (including the input and output layers)
		self.epochs = epochs				# number of training epochs
		self.learning_rate = learning_rate	# GD learning rate
		self.momentum = momentum			# 
		self.validation = validation		# number of how to split training and validation data
		self.losses = np.zeros((self.epochs, 2))

		self.xi = []	# an array of arrays for the activations (ξ)
		self.w = []		# an array of matrices for the weights (w)		
		self.h = [None]*self.L				# an array of arrays for the fields (h)
		self.theta = []			# an array of arrays for the thresholds (θ)
		self.delta = []			# an array of arrays for the propagation of errors (Δ)
		self.d_w = []			# an array of matrices for the changes of the weights (δw)
		self.d_theta = []		# an array of arrays for the changes of the weights (δθ)
		self.d_w_prev = []		# an array of matrices for the previous changes of the weights, used for the momentum term (δw^(prev))
		self.d_theta_prev = []	# an array of arrays for the previous changes of the thresholds, used for the momentum term (δθ^(prev))
		self.fact = fact		# the name of the activation function that it will be used. It can be one of these four: sigmoid, relu, linear, tanh.
	
	def fit(self, X, y):
		"""
		Method to fit and train the model.
		params:
		@X, float[]: input data-set of size (n_samples, n_features)
		@y, float[]: input labels of size (n_samples)
		"""
		
		print(f"\n========== MODEL TRAIN METHOD ==========\n")

		# split the given data to train and validation
		X, X_valid, y, y_valid = train_test_split(X, y, test_size=self.validation, random_state=42)
		n_samples, n_features = X.shape
		v_samples, v_features = X_valid.shape		
		
		# initialize array of w, xi etc. based on given data and layers from constructor
		self.w.append(np.zeros((1, 1)))		# we use _w_ from idx 1, so just define the 0th (unused). Maybe it could be init as other below, but I dont want to touch it while it works :)
		self.d_w.append(np.zeros((1, 1)))
		self.d_w_prev.append(np.zeros((1, 1)))
		for lay in range(1, self.L):
			# Xavier init, they say it is better than random :)
			limit = np.sqrt(6 / (self.n[lay] + self.n[lay - 1]))
			weights = np.random.uniform(-limit, limit, size=(self.n[lay], self.n[lay - 1]))
			self.w.append(weights)
			# self.w.append(np.random.uniform(low=-0.1, high=0.1, size=(self.n[lay], self.n[lay - 1])))    # random init
			self.d_w.append(np.ones((self.n[lay], self.n[lay - 1])))
			self.d_w_prev.append(np.ones((self.n[lay], self.n[lay - 1])))

		# init of other arrays
		for lay in range(self.L):
			self.xi.append(np.zeros(self.n[lay]))
			self.theta.append(np.zeros(self.n[lay]))
			self.delta.append(np.zeros(self.n[lay]))
			self.d_theta.append(np.zeros(self.n[lay]))
			self.d_theta_prev.append(np.zeros(self.n[lay]))

		if self.fact == "linear":
			activation_function = Layers.LinearLayer()
		elif self.fact == "relu":
			activation_function = Layers.ReLULayer()
		elif self.fact == "sigmoid":
			activation_function = Layers.SigmoidLayer()
		elif self.fact == "tanh":
			activation_function = Layers.TanhLayer()
		
		output_function = Layers.LinearLayer(output=True)
		loss_function = LossFunctions.MSELoss()

		for epoch in range(self.epochs):
			print(f"Epoch {epoch} / {self.epochs}", end="\t")
			for sample in range(n_samples):
				self.xi[0] = activation_function.forward(X[sample])

				# feed-forward
				for l in range(1, self.L):
					self.h[l] = self.w[l] @ self.xi[l-1] - self.theta[l]
					self.xi[l] = activation_function.forward(self.h[l]) if l != self.L-1 else output_function.forward(self.h[l])
				prediction = self.xi[-1]
				loss = loss_function.forward(prediction, y[sample])
				self.delta[-1] = output_function.derivative(self.h[-1])*loss_function.derivative(prediction, y[sample])
				
				# go back to compute deltas
				for l in range(self.L-1, 1, -1):
					self.delta[l-1] = activation_function.derivative(self.h[l-1])*(self.delta[l] @ self.w[l])

				# computing gradient
				for l in range(1, self.L):
					self.d_w[l] = -self.learning_rate * np.outer(self.delta[l], self.xi[l-1]) + self.momentum * self.d_w_prev[l]
					self.d_theta[l] = self.learning_rate * self.delta[l] + self.momentum * self.d_theta_prev[l]
				
				# weights update
				for l in range(self.L):
					self.w[l] += self.d_w[l]
					self.theta[l] += self.d_theta[l]

				# weights prev setting
				for l in range(1, self.L):
					self.d_w_prev[l] = self.d_w[l]
					self.d_theta_prev[l] = self.d_theta[l]
			
			# try the current model on train and validation data
			train_loss = 0
			valid_loss = 0
			train_correct = 0
			valid_correct = 0

			# train
			for sample in range(n_samples):
				a_t = [None]*self.L
				z_t = [None]*self.L
				z_t[0] = X[sample].copy()
				for l in range(1, self.L):
					a_t[l] = self.w[l] @ z_t[l-1] - self.theta[l]
					z_t[l] = activation_function.forward(a_t[l]) if l != self.L-1 else output_function.forward(a_t[l])
				
				train_loss += loss_function.forward(z_t[-1], y[sample])
				train_correct = train_correct + 1 if np.round(z_t[-1]) == y[sample] else train_correct
			self.losses[epoch, 0] = train_loss / train_correct

			# validation
			for sample in range(v_samples):
				a_v = [None]*self.L
				z_v = [None]*self.L
				z_v[0] = X_valid[sample].copy()
				for l in range(1, self.L):
					a_v[l] = self.w[l] @ z_v[l-1] - self.theta[l]
					z_v[l] = activation_function.forward(a_v[l]) if l != self.L-1 else output_function.forward(a_v[l])
				
				valid_loss += loss_function.forward(z_v[-1], y_valid[sample])
				valid_correct = valid_correct + 1 if np.round(z_v[-1]) == y_valid[sample] else valid_correct
			self.losses[epoch, 1] = valid_loss / valid_correct

			# print(f"train loss: {train_loss}\ttrain average: {train_loss/n_samples}\tvalid loss: {valid_loss}\tvalid average: {valid_loss/v_samples}")
			print(f"train loss: {train_loss[0]:.3f}\ttrain accuracy: {train_correct/n_samples:.3f}\tvalid loss: {valid_loss[0]:.3f}\tvalid accuracy: {valid_correct/v_samples:.3f}")
		
	def predict(self, X):
		n_samples, n_features = X.shape
		
		predictions = []
		if self.fact == "linear":
			activation_function = Layers.LinearLayer()
		elif self.fact == "relu":
			activation_function = Layers.ReLULayer()
		elif self.fact == "sigmoid":
			activation_function = Layers.SigmoidLayer()
		elif self.fact == "tanh":
			activation_function = Layers.TanhLayer()
		output_function = Layers.LinearLayer(output=True)
		
		# for each sample, do the forward messaging and obtain the result
		for sample in range(n_samples):
			a = [None]*self.L
			z = [None]*self.L
			z[0] = X[sample]
			for l in range(1, self.L):
				a[l] = self.w[l] @ z[l-1] - self.theta[l]
				z[l] = activation_function.forward(a[l]) if l != self.L-1 else output_function.forward(a[l])

			prediction = z[-1]
			predictions.append(prediction)

		return np.array(predictions)
	
	def loss_epochs(self):
		return self.losses
