import numpy as np
import random


class TreeParityMachine:

    def __init__(self, name, k, l, n):
        self.name = name
        self.k = k
        self.l = l
        self.n = n

        self.input = None
        self.weights = self.init_weights()
        self.perceptron = np.ones((self.k), dtype=np.int8)
        self.output = 1
        
    def init_weights(self):
        weights = np.zeros((self.k, self.n), dtype=np.int8)
        for p_i in range(self.k):
            for x_i in range(self.n):
                weights[p_i, x_i] = random.randint(-self.l, self.l)
        return weights

    def process(self, input):
        self.input = input
        self.output = 1
        for p_i in range(self.k):
            value = 0
            for x_i in range(self.n):
                value += self.input[p_i, x_i] * self.weights[p_i, x_i]
            self.perceptron[p_i] = self.sign(value)
            self.output *= self.perceptron[p_i]
        return self.output

    def sign(self, value):
        if value <= 0:
            return -1            
        return 1

    def train(self, output):
        if self.output != output:
            return
        for p_i in range(self.k):
            if self.perceptron[p_i] == self.output:
                for x_i in range(self.n):
                    self.weights[p_i, x_i] += self.perceptron[p_i] * self.input[p_i, x_i]
                    if self.weights[p_i, x_i] > self.l:
                        self.weights[p_i, x_i] = self.l
                    elif self.weights[p_i, x_i] < -self.l:
                        self.weights[p_i, x_i] = -self.l

    def train_attacker(self, a_out, b_out):
        if self.output != a_out or self.output != b_out:
            return
        for p_i in range(self.k):
            if self.perceptron[p_i] == self.output:
                for x_i in range(self.n):
                    self.weights[p_i, x_i] += self.perceptron[p_i] * self.input[p_i, x_i]
                    if self.weights[p_i, x_i] > self.l:
                        self.weights[p_i, x_i] = self.l
                    elif self.weights[p_i, x_i] < -self.l:
                        self.weights[p_i, x_i] = -self.l

    def print(self):
        string = "{} weights -> [".format(self.name)
        for p_i in range(self.k):
            for x_i in range(self.n):
                if self.weights[p_i, x_i] < 0:
                    string += "{} ".format(self.weights[p_i, x_i])
                else:
                    string += " {} ".format(self.weights[p_i, x_i])
        string += "] "
        print(string)
