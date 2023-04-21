from neural import *

print("\n\nTraining XOR\n\n")
xor_training_data = [([1, 1], [0]), ([1, 0], [1]), ([0, 1], [1]), ([0, 0], [0])]

xorn = NeuralNet(2, 1, 1)
xorn.train(xor_training_data)
print(xorn.test_with_expected(xor_training_data))
print()
print()

party_training_data = [([0.9, 0.6, 0.8, 0.3, 0.1], [1]), 
                       ([0.8, 0.8, 0.4, 0.6, 0.4], [1]),
                       ([0.7, 0.2, 0.4, 0.6, 0.3], [1]),
                       ([0.5, 0.5, 0.8, 0.4, 0.8], [0]),
                       ([0.3, 0.1, 0.6, 0.8, 0.8], [0]),
                       ([0.6, 0.3, 0.4, 0.3, 0.6], [0])]
party_test_data = [([1.0, 1.0, 1.0, 0.1, 0.1]),
                   ([0.5, 0.2, 0.1, 0.7, 0.7]),
                   ([0.8, 0.3, 0.3, 0.3, 0.8]),
                   ([0.8, 0.3, 0.3, 0.8, 0.3]),
                   ([0.9, 0.8, 0.8, 0.3, 0.6])]

nn = NeuralNet(5, 10, 1)
nn.train(party_training_data)

for i in nn.test(party_test_data):
    print(f"actual: {i[1]}")
