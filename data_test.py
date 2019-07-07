import csv
with open("example_0.txt") as f:
    reader = csv.reader(f)
    for row in reader:
