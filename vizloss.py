import matplotlib.pyplot as plt
import csv

loss_d = []
loss_g = []
x = []
filename = "loss9.txt"
with open(filename) as f:
    reader = csv.reader(f,delimiter="\t")
    count = 0
    for row in reader:
        print(count)
        x.append(int(count))
        loss_d.append(float(row[0]))
        loss_g.append(float(row[1]))
        count+=1
plt.plot(x,loss_d)
plt.plot(x,loss_g)
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.show()