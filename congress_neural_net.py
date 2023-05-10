import csv
from typing import Tuple
from neural import *

testing_members = [[116, 15448,	71, 12,	100, 1940, -0.49, -0.205, 68, 0], #Pelosi
                   [116, 20703,	71,	23,	200, 1965, 0.455, 0.235, 789, 41], #McCarthy
                   [116, 20954,	21,	5, 100, 1958, -0.322, -0.266, 795, 6], #Quigley
                   [116, 21343,	13,	8, 100, 1970, -0.488, -0.095, 798, 6], #Jeffries
                   [116, 21949,	13,	14, 100, 1989, -0.256, -0.967, 805, 29], #AOC
                   [116, 21710,	68,	1, 200, 1966, 0.418, 0.762,	773, 57], #Cheney
                   [116, 21719,	43,	1, 200, 1982, 0.604, -0.63, 712, 93] #Gaetz
                   ]

members = ["Pelosi", "McCarthy", "Quigley", "Jeffries", "AOC", "Cheney", "Gaetz"]


def normalize_add (row, list_, train_test):
    state_icpsr = float(row[2])
    state = state_icpsr / 100
    party = 0 if int(row[4]) == 100 else 1
    birth_year = float(row[5])
    born = (birth_year - 1900) / 100
    nom_dim_1 = float(row[6])
    nom_dim_2 = float(row[7])
    num_votes = float(row[8])
    num_errors = float(row[9])
    error_prop = num_errors / num_votes
    errors_norm = num_errors / 811
    if train_test == "train":
        list_.append(([nom_dim_2,  error_prop], [party]))
    elif train_test == "test":
        list_.append(([nom_dim_2,  error_prop]))
    return list_

training_data = []

with open("H117_members.csv", "r") as f:
    read_csv = csv.reader(f)
    for row in read_csv:
        #print(row[1])
        training_data = normalize_add(row, training_data, "train")
    #print(training_data)

td = training_data
#print(td)

testing_data = []
for member in testing_members:
    testing_data = normalize_add(member, testing_data, "test")

#print(testing_data)
print()

nn = NeuralNet(2, 10, 1)
nn.train(td, iters=6000, print_interval=200)
num_tested = 0
num_correct = 0

print()

for i in nn.test_with_expected(td):
    val = 1 if i[2][0] > 0.9 else 0 if i[2][0] < 0.1 else i[2][0]
    if i[1][0] == val:
        num_correct += 1
    else:
        print(f"desired: {i[1]}, actual: {i[2]}")
    num_tested +=1

print("correct proportion:", num_correct, "out of", num_tested)

test_id = 0
for i in nn.test(testing_data):
    print(members[test_id])
    val = 1 if i[1][0] > 0.9 else 0 if i[1][0] < 0.1 else i[1][0]
    print(val)
    test_id += 1


