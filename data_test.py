import csv
import numpy as np
import statistics as s
from scipy import stats
from random import sample
n_genes = 0
with open("example/example_data_0.txt") as f:
    reader = csv.reader(f)
    for row in reader:
        n_genes+=1

def get_p_vals(real,fake):
    count = 0
    for i in range(real.shape[0]):
        p = stats.ttest_ind(real[i],fake[i],equal_var=False)[1]
        if p>.05:
            count+=1
        print(p)
    print(count)

def get_stats(d):
    count = 0
    for i in range(d.shape[0]):
        m = s.mean(d[i])
        std = s.stdev(d[i])
        print(str(m)+"\t"+str(std)+"\n")
        if std >=.9 and std <= 1.1:
            count+=1
    print(count)

data = np.zeros((419,n_genes))
print(data.shape)
for i in range(419):
    with open("example/example_data_"+str(i)+".txt") as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            data[i][count] = float(row[0])
            count+=1
gene_first = data.T
real = sample(data.tolist(),100)
real = np.asarray(real)
real = real.T
print(gene_first.shape)
epoch = 3300
gen_data = np.zeros((100,n_genes))
for i in range(100):
    with open("generated_data/model14data/sample"+str(epoch)+"_"+str(i)+".txt") as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            gen_data[i][count] = float(row[0])
            count+=1
fake = gen_data.T
# get_p_vals(gene_first,fake)
get_p_vals(real,fake)
