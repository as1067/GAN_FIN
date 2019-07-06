import matplotlib.pyplot as plt
import csv

loss_d = []
loss_g = []
x = []
filename = "loss1.txt"
with open(filename) as f:
    reader = csv.reader(f,delimiter="\t")
    count = 0
    for row in reader:
        print(count)
        x.append(int(count))
        loss_d.append(float(row[0]))
        loss_g.append(float(row[1]))
        count+=1
plt.scatter(x,loss_d,color = "r")
plt.scatter(x,loss_g,color = "g")

plt.show()