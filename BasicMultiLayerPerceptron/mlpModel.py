import numpy as np
import math

def sigmoid(z):
	return np.array([1/(np.exp(-x)+1) for x in z])


class MLPModel:
	# w^(l)_(j,k) = weight[l][j][k]
	# b^(l)_(k) = biases[l][k]
	# z^(l)_(j) = ∑(k)w^(l)_(j,k)*a^(l-1)_(k) + b^(l)_(j)
	#           = ∑(k)weight[l][j][k]*record[l-1][k] + biases[j]
	# a^(l)_(k) = record[l][k]
	def __init__(self, layersInfo, learnRate):
		self.weights = [[]]
		self.biases = [[]]
		self.learnRate = learnRate
		for i in range(len(layersInfo) - 1):
			weight = np.random.randn(layersInfo[i+1], layersInfo[i])
			# weight = np.ones((layersInfo[i + 1], layersInfo[i]))
			self.weights.append(weight)
			bias = np.random.randn(layersInfo[i+1])
			# bias = np.ones(layersInfo[i + 1])
			self.biases.append(bias)
		self.record = []
	
	
	def feedForward(self, input):
		self.record = []
		output = np.array(input)
		self.record.append(output)
		for l in range(1, len(self.weights)):
			w = self.weights[l]
			b = self.biases[l]
			output = sigmoid(w.dot(output) + b)
			self.record.append(output)
		return output
	
	def backPropagate(self, output, trueval):
		deltas = []
		for j in range(len(self.record[-1])):
		# 	size = output.shape[0]
		# 	dvt = 0
		# 	for n in range(size):
		# 		dvt += (output)
		# 	dvt /= size
			delta = (output[j] - trueval[j]) * output[j] * (1 - output[j])
			deltas.append(delta)
			self.biases[j] -= self.learnRate * delta
			k = 0
			for k in range(len(self.record[-2])):
				self.weights[-1][j][k] -= self.learnRate * delta * self.record[-2][k]
		
		l = len(self.record) - 2
		
		while l > 0:
			newdeltas = []
			j = 0
			for j in range(len(self.record[l])):
				newdelta = 0
				k = 0
				for k in range(len(self.record[l+1])):
					newdelta += self.weights[l+1][k][j] * deltas[k] * \
					            self.record[l][j] * (1 - self.record[l][j])
				newdeltas.append(newdelta)
				k = 0
				for k in range(len(self.record[l-1])):
					self.weights[l][j][k] -= self.learnRate * newdelta * self.record[l-1][k]
				self.biases[l][j] -= self.learnRate * newdelta
			deltas = newdeltas
			l-=1
	def train(self, data, trueval):
		for i in range(len(trueval)):
			output = self.feedForward(data[i])
			self.backPropagate(output, trueval[i])
			print(i)

	
if __name__ == "__main__":
	model = MLPModel([2,64,1], 0.5)
	metadata = [[0, 0], [1, 0],[0, 1],[1,1]]
	metatrue = [[0], [1], [1], [0]]
	data = []
	true = []
	print(model.feedForward([0,0]))
	for i in range(150000):
		data.append(metadata[i%4])
		true.append(metatrue[i%4])
	model.train(data, true)
	print(model.feedForward([0,0]))
	print(model.feedForward([0,1]))
	print(model.feedForward([1,0]))
	print(model.feedForward([1,1]))
	
