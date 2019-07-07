import csv
import numpy as np
import statistics as s
from random import sample
n_genes = 0
with open("example/example_data_0.txt") as f:
    reader = csv.reader(f)
    for row in reader:
        n_genes+=1

def get_mean_gene(d):
    means = []
    for gene in d:
        m = s.mean(gene)
        means.append(m)
        print(str(m)+"\t"+str(s.stdev(gene)))


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
print(gene_first.shape)
get_mean_gene(gene_first)
epoch = 1000
gen_data = np.zeros((100,n_genes))
for i in range(100):
    with open("generated_data/model9data/sample"+str(epoch)+"_"+str(i)+".txt") as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            gene_first[i][count] = float(row[0])
            count+=1
