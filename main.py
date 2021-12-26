import numpy as np
import random
from tree_parity_machine import TreeParityMachine


def sync(a_weights, b_weights):
    return np.array_equal(a_weights, b_weights)

def input_generator(k, n):
    input = np.zeros((k, n), dtype=np.int8)
    for p_i in range(k):
        for x_i in range(n):       
            input[p_i, x_i] = random.choice([-1, 1])
    return input 


k = 3
l = 5
n = 10

a = TreeParityMachine("Person 1", k, l, n)
b = TreeParityMachine("Person 2", k, l, n)
e = TreeParityMachine("Attacker", k, l, n)

iteration = 0
print("Iteration = {}".format(iteration))
a.print()
b.print()
e.print()

while not sync(a.weights, b.weights):
    input = input_generator(k, n)
    a_out = a.process(input)
    b_out = b.process(input)
    e_out = e.process(input)
    a.train(b_out)
    b.train(a_out)
    e.train_attacker(a_out, b_out)
    iteration += 1

print("\nIteration = {}".format(iteration))
a.print()
b.print()
e.print()