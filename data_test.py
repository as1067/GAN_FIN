import csv
import numpy as np
import statistics as s
n_genes = 0
with open("example/example_data_0.txt") as f:
    reader = csv.reader(f)
    for row in reader:
        n_genes+=1

def get_stats(d):
    for gene in d:
        print(str(s.mean(gene))+"\t"+str(s.stdev(gene)))


data = np.zeros((n_genes,419))
print(data.shape)
gene_first = data.T
print(gene_first.shape)
for i in range(419):
    with open("example/example_data_"+str(i)+".txt") as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            gene_first[i][count] = float(row[0])
            count+=1
get_stats(gene_first)

