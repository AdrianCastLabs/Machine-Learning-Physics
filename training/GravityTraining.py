import csv
import torch
import torch.nn as nn

# load data
rows = []

with open('../data/gravity-simulation-data.csv') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        rows.append([float(x) for x in row])

data = torch.tensor(rows, dtype=torch.float32)