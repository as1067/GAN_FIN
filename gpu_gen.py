import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import math
import random
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from math import sqrt, ceil
from operator import itemgetter
from copy import deepcopy
from sklearn.model_selection import KFold
from scipy import stats
from multiprocessing import Process, Queue
import sys
import argparse
import csv
import statistics
from keras import backend as K
from random import sample

random.seed(0)
np.random.seed(0)
mrna_file = "LA_mRNA.txt"
clinical_file = "LA_Clinical.txt"


def main():
    # preprocossing.
    pm = preprocessing()
    print("----------------------------------------------------------------------------------------------------")
    print("2. Step 1 : reconstructing FIs network")
    reconstructed_FIs_perfold, gene_in_reconstructed_FIs_perfold = pm.reconstruct_FIs_network()
    print(gene_in_reconstructed_FIs_perfold)
    print("----------------------------------------------------------------------------------------------------")
    print("3. Step 2,3 : Learning the network and Feature selection using PageRank")
    biomarker_perfold = []
    file = open("gpu_latest.txt", "r")
    n = int(file.read())
    # for foldnum in range(2):
    data_for_GANs = pm.mk_data_for_GANs(gene_in_reconstructed_FIs_perfold)

    score = np.zeros(pm.mRNA.shape[0])
    pm.Learning_FIsnetwork_GANs(reconstructed_FIs_perfold, data_for_GANs,n)
    file = open("gpu_latest.txt","w")
    file.write(str(n+1))
"""
	From here,Functions for preprocessing
	this step includes loading data,intersectioning data,z-scoring for each sample and t-test for each fold

"""


# read a comma-delimited text file.
def read_file(file):
    ids = []
    num2gene = {}
    gene2num = {}
    genes = {}
    num_samples = 0
    x = 0
    y = 0
    with open(file) as fop:
        # print("opened file")
        reader = csv.reader(fop, delimiter=",")
        for row in reader:
            # print(row)
            y += 1
            if y == 1:
                for gene in row:
                    x += 1
        data = np.zeros((y - 1, x - 1))
        num_samples = x
    with open(file) as fop:
        print("opened file")
        reader = csv.reader(fop, delimiter=",")
        y = 0
        for row in reader:
            # print(y)
            x = 0
            for x in range(num_samples):
                # print(gene)
                if y == 0:
                    if not x == 0:
                        ids.append(row[x])
                else:
                    if x == 0:
                        genes[row[x]] = 1
                        num2gene[y - 1] = row[x]
                        gene2num[row[x]] = y - 1
                    else:
                        # print(str(y-1)+" "+str(x-1))
                        data[y - 1][x - 1] = float(row[x])
            y += 1
    # print(ids)
    # sys.exit()
    return data, ids, genes, num2gene, gene2num


# download mRNA,CNA,methylation,SNP,clinical file(lable),FIs network and parameters.
def loading_data():
    # download mRNA,CNA,methylation and SNP.
    mRNA, ids, genes, num2gene, gene2num = read_file(mrna_file)

    # download FIS network.
    with open('FIsnetwork.txt', 'r') as fop:
        edges = []
        for line in fop:
            edges.append(line.strip().split(','))

    # download lable (lable 0: patient who has good prognosis,lable 1: patient who has bad prognosis).
    lable = {}
    sample2id = {}
    with open(clinical_file) as fop:
        reader = csv.reader(fop, delimiter=",")
        count = 0
        for row in reader:
            lable[row[0]] = int(row[1])
            sample2id[row[0]] = count
            count += 1

    # download parameters
    # 400
    n_gene_in_ttest = 400
    # 250
    n_biomarker = 250
    damping_factor = .7
    n_experiment = 1
    n_limit = 5
    return mRNA, edges, lable, n_gene_in_ttest, n_biomarker, damping_factor, n_experiment, n_limit, sample2id, genes, num2gene, gene2num


# find the intersection of genes in mRNA, CNA, methylation,SNP data and FIs network and the intersection of samples in mRNA, CNA, methylation, and SNP data.
def intersetion_data(raw_mRNA, raw_edges, raw_clinical_file, gs, num2gene, gene2num):
    print("finding intersection")
    # find the intersection of genes in mRNA, CNA, methylation, and SNP data.
    # co_gene=[x for x in raw_mRNA.index if x in raw_CNA.index]
    # co_gene=[x for x in raw_met.index if x in co_gene]
    # co_gene=[x for x in raw_snp.index if x in co_gene]
    co_gene = gs
    # find the intersection between the genes from the previous step and the genes in the FIs network.
    edge_list = []
    ppi_genes = set()
    for edge in raw_edges:
        gene1, gene2 = edge[0], edge[1]
        condition = ((gene1 in co_gene) and (gene2 in co_gene))
        if condition:
            # print("gene found")
            edge_list.append([gene1, gene2])
            ppi_genes.add(gene1)
            ppi_genes.add(gene2)
    ppi_genes = list(ppi_genes)
    genes = {}
    for gene in ppi_genes:
        genes[gene2num[gene]] = gene

    # find the intersection of samples in mRNA, CNA, methylation, and SNP data.
    # co_sample=[x for x in raw_mRNA.columns if x in raw_CNA.columns]
    # co_sample=[x for x in raw_met.columns if x in co_sample]
    # co_sample=[x for x in raw_snp.columns if x in co_sample]
    # co_sample = get_sample_id(raw_mRNA)
    # modify raw mRNA, raw CNA, raw methylation, raw SNP, and raw lable data with the intersection of the genes and the intersection of the samples.
    y = len(genes)
    x = raw_mRNA.shape[1]
    gene2num_small = {}
    num2gene = {}
    mRNA = np.zeros((y, x))
    gene_names = gs
    count = 0
    y = 0
    for gene in raw_mRNA:
        # temp = np.array([])
        if not count == 0:
            try:
                if genes[count]:
                    gene2num_small[genes[count]] = y
                    num2gene[y] = genes[count]
                    x = 0
                    for d in gene:
                        # print(d)
                        mRNA[y][x] = d
                        x += 1
                    y += 1
            except(KeyError):
                pass
        count += 1
    # print(mRNA)
    # CNA=raw_CNA.loc[ppi_genes,co_sample]
    # met=raw_met.loc[ppi_genes,co_sample]
    # snp=raw_snp.loc[ppi_genes,co_sample]
    lable = raw_clinical_file

    return mRNA, edge_list, lable, gene_names, gene2num, num2gene, gene2num_small


# normalizing data for each sample by z-scoring.
def zscore(data):
    print("zscoring")
    print(data.shape)
    # print(data)
    len_row_gene = data.shape[0]
    len_column_sample = data.shape[1]
    # column_wise = data.T
    zscored_data = np.zeros((len_row_gene, len_column_sample))
    for gene in range(len_row_gene):
        mu = statistics.mean(data[gene])
        sigma = statistics.stdev(data[gene])
        # print(str(mu)+"\t"+str(sigma))
        if sigma != 0:
            for sample in range(len_column_sample):
                x = data[gene][sample]
                zscored_data[gene][sample] = (x - mu) / sigma
    return zscored_data


# seperate patients who have bad prognosis and patients who have good prognosis.
def seperate_good_bad_patients(lable):
    good_samples = []
    bad_samples = []
    for sample in lable:
        j = lable[sample]
        if j == 0:
            good_samples.append(sample)
        elif j == 1:
            bad_samples.append(sample)
        else:
            # exception Handling
            print(
                '#########################################################################################################################')
            print('									error!lable can be only 0 or 1')
            print('									You have to stop this process!')
            print(
                '#########################################################################################################################')
    return good_samples, bad_samples


# perfom t-test comparing patients who have poor prognosis and patients who have good prognosis for each dataset.
# num2gene is Series in which data is a gene and index is a gene number.
# good_sam is the list of samples which have good prognosis and bad_sam is the list of samples which have bad prognosis.
def t_test(mRNA, num2gene, good_sam, bad_sam):
    # exception Handling
    if len(good_sam) == 0:
        print(
            '#########################################################################################################################')
        print('									Error!there is no good prognostic patient')
        print('									You have to stop this process!')
        print(
            '#########################################################################################################################')
    if len(bad_sam) == 0:
        print(
            '#########################################################################################################################')
        print('									error!there is no bad prognostic patient')
        print('										You have to stop this process!')
        print(
            '#########################################################################################################################')

    n_genes = mRNA.shape[0]

    def get_data(g, samples):
        data = []
        for sample in samples:
            data.append(mRNA[g][sample])
        return data

    # perform a t-test for each gene in mRNA data conparing a poor prognostic patient group and a good prognostic patient group.
    t_scores = []
    for i in range(n_genes):
        poor_data = get_data(i, bad_sam)
        good_data = get_data(i, good_sam)
        poor_data = np.asarray(poor_data)
        good_data = np.asarray(good_data)
        t_statistic = abs(stats.ttest_ind(poor_data.ravel(), good_data.ravel(), equal_var=False)[0])
        if np.isnan(t_statistic):
            t_statistic = 0
            t_scores.append((t_statistic, i))
        else:
            t_scores.append((t_statistic, i))

    # t_scores is DataFrame in which data is a list of the t-test statistics in mRNA data and index is a list of genes.

    # #perform a t-test for each gene in CNA data conparing a poor prognosis patient group and a good prognosis patient group.
    # t_scores2 = np.zeros(n_genes, dtype=np.float32)
    # for i in range(n_genes):
    # 	poor_data = CNA.loc[num2gene[i], bad_sam].values.astype(np.float64)
    # 	good_data = CNA.loc[num2gene[i], good_sam].values.astype(np.float64)
    # 	t_statistic = abs(stats.ttest_ind(poor_data,good_data)[0])
    # 	if np.isnan(t_statistic) :
    # 		t_statistic = 0
    # 		t_scores2[i] = t_statistic
    # 	else :
    # 		t_scores2[i] = t_statistic
    # #t_scores2 is DataFrame in which data is a list of the t-test statistics in CNA data and index is a list of genes.
    # t_scores2=DataFrame(t_scores2,index=genes)

    # #Perform a t-test for each gene in methylation data comparing a poor prognosis patient group and a good prognosis patient group.
    # t_scores3= np.zeros(n_genes, dtype=np.float32)
    # for i in range(n_genes):
    # 	poor_data = met.loc[num2gene[i], bad_sam].values.astype(np.float64)
    # 	good_data = met.loc[num2gene[i], good_sam].values.astype(np.float64)
    # 	t_statistic = abs(stats.ttest_ind(poor_data,good_data)[0])
    # 	if np.isnan(t_statistic) :
    # 		t_statistic = 0
    # 		t_scores3[i] = t_statistic
    # 	else :
    # 		t_scores3[i] = t_statistic
    # #t_scores3 is DataFrame in which data is a list of the t-test statistics in methylation data and index is a list of genes.
    # t_scores3=DataFrame(t_scores3,index=genes)

    # #Perform a t-test for each gene in SNP data conparing a poor prognosis patient group and a good prognosis patient group.
    # t_scores4= np.zeros(n_genes, dtype=np.float32)
    # for i in range(n_genes):
    # 	poor_data = snp.loc[num2gene[i], bad_sam].values.astype(np.float64)
    # 	good_data = snp.loc[num2gene[i], good_sam].values.astype(np.float64)
    # 	t_statistic = abs(stats.ttest_ind(poor_data,good_data)[0])
    # 	if np.isnan(t_statistic) :
    # 		t_statistic = 0
    # 		t_scores4[i] = t_statistic
    # 	else :
    # 		t_scores4[i] = t_statistic
    # #t_scores4 is DataFrame in which data is a list of the t-test statistics in SNP data and index is a list of genes.
    # t_scores4=DataFrame(t_scores4,index=genes)
    # print(t_scores)
    return t_scores


# perform preprocessing.
def preprocessing():
    print('1.preprocessing data...')
    # download mRNA,CNA,methylation,SNP,lable,FIs network data and parameters.
    print(' loading data...')
    raw_mRNA, raw_edges, raw_lable, n_gene_in_ttest, n_biomarker, damping_factor, n_experiment, n_limit, sample2id, genes, num2gene, gene2num = loading_data()

    # find the intersection of genes in mRNA, CNA, methylation,SNP data and FIs network and the intersection of samples from mRNA, CNA, methylation, and SNP data.
    # then modify raw mRNA, raw CNA, raw methylation, raw SNP, and raw lable data with the intersection of the genes and the intersection of the samples.

    raw_mRNA2, edge_list, lable, gene_names, gene2num, num2gene, small_gene2num = intersetion_data(raw_mRNA, raw_edges,
                                                                                                   raw_lable, genes,
                                                                                                   num2gene, gene2num)
    # raw_mRNA2 = np.asarray(raw_mRNA2)
    # normalizing data for each sample by z-scoring in mRNA, CNA, methylation and SNP data respectly.
    # mRNA = zscore(raw_mRNA2)
    mRNA = raw_mRNA2
    # CNAvalues=zscore(raw_CNA2.values.astype('float64'))
    # metvalues=zscore(raw_met2.values.astype('float64'))
    # snpvalues=zscore(raw_snp2.values.astype('float64'))
    # mRNA = DataFrame(mvalues, index=raw_mRNA2.index, columns=raw_mRNA2.columns)
    # CNA = DataFrame(CNAvalues, index=raw_CNA2.index, columns=raw_CNA2.columns)
    # met = DataFrame(metvalues, index=raw_met2.index, columns=raw_met2.columns)
    # snp = DataFrame(snpvalues, index=raw_snp2.index, columns=raw_snp2.columns)

    # #gene2num and num2gene are for mapping between genes and numbers.
    # gene2num = {}
    # num2gene = {}
    # # print(str(mRNA.index))
    # i = 0
    # for gene in gene_names:
    # 	gene2num[gene] = i
    # 	num2gene[i] = gene
    # 	i+=1

    # divide samples for 10fold validation
    good_sam, bad_sam = seperate_good_bad_patients(lable)
    # mRNA.T
    # kf = KFold(n_splits=2, random_state=None, shuffle=False)
    print(good_sam)
    good_ids = []
    bad_ids = []
    for sample in good_sam:
        good_ids.append(sample2id[sample])
    for sample in bad_sam:
        bad_ids.append(sample2id[sample])
    test_samples = []
    print("t-testing")
    mRNA_ttmp = t_test(mRNA, num2gene, good_ids, bad_ids)

    # make instance of class PM.
    Pm = PM(n_gene_in_ttest, n_biomarker, damping_factor, n_experiment, n_limit, mRNA, lable, edge_list,
            good_sam, bad_sam, test_samples, gene2num, num2gene, mRNA_ttmp, sample2id,
            small_gene2num)

    return Pm


"""
	From here,Functions for Step 1,2,3 and 4 in paper.
	step1 is recostructing FIs network.
	step2 is learning the network.
	step3 is Feature selection using PageRank.
	step4 is prognosis predicition.
"""


# PM is the class which has multi-omics Data, parameters, series to map between genes and gene numbers, samples for 10 fold validation , t-statistics for each fold and functions for Step 1, 2, 3, and 4.
class PM:
    # initialize variables.
    """
        self.n_gene_in_ttest is a parameter which is the number of genes which have high absolute values of t-statistics to be used in step 1.
        self.n_experiment is parameter which is the number of times to repeat step2 and step3.
        self.n_biomarker is parameter which is the number of genes which are selected as biomarkers.
        self.limit is parameter of step 2,3. when step2 and step3 are repeated t times(t=n_experiment), the genes that appeared b times in t times is selected as biomarkers. The b is the limit.
        self.damping_factor is damping factor using in pagerank algorithm.
        self.mRNA is mRNA data.
        self.CNA is CNA data.
        self.met is methylation data.
        self.snp is SNP data.
        self.lable is lable of samples (clinical data).
        self.edge_list is edges in FIs network.
        self.good_train_samples is a list containing lists of good prognostic training samples in each fold for 10 fold validation.
        self.bad_train_samples is a list containing lists of bad prognostic trainging samples in each fold for 10 fold validation.
        self.test_samples a list containing lists of test samples in each fold for 10 fold validation.
        self.gene2num is series for mapping from genes to gene numbers.
        self.num2gene is series for mapping from gene numbers to genes.
        self.mRNA_ttest is a list containing DataFrames that are the results of t-test in mRNA data per fold at the preprocessing stage.
        self.CNA_ttest is a list containing DataFrames that are the results of t-test in CNA data per fold at the preprocessing stage.
        self.met_ttest is a list containing DataFrames that are the results of t-test in methylation data per fold at the preprocessing stage.
        self.snp_ttest is a list containing DataFrames that are the results of t-test in SNP data per fold at the preprocessing stage.
    """

    def __init__(self, n_gene_in_ttest, n_biomarker, damping_factor, n_experiment, n_limit, mRNA, lable, edge_list,
                 good_train_samples, bad_train_samples, test_samples, gene2num, num2gene, mRNA_ttest, sample2id,
                 small_gene2num):
        self.n_gene_in_ttest = n_gene_in_ttest
        self.n_experiment = n_experiment
        self.n_biomarker = n_biomarker
        self.limit = n_limit
        self.damping_factor = damping_factor
        self.mRNA = mRNA
        # self.CNA=CNA
        # self.met=met
        # self.snp=snp
        self.lable = lable
        self.edge_list = edge_list
        self.good_train_samples = good_train_samples
        self.bad_train_samples = bad_train_samples
        self.test_samples = test_samples
        self.gene2num = gene2num
        self.num2gene = num2gene
        self.mRNA_ttest = mRNA_ttest
        self.sample2id = sample2id
        self.small_gene2num = small_gene2num

    # self.CNA_ttest=CNA_ttest
    # self.met_ttest=met_ttest
    # self.snp_ttest=snp_ttest
    # step 1. reconstruct FIs network.
    def reconstruct_FIs_network(self):
        print("reconstructing network")
        # reconstructed_network_10fold is list containing lists of the edges in reconstructed network per fold.
        # reconstructed_network_10fold = []

        # gene_in_reconstructed_network_10fold is list containing sets of the genes in reconstructed network per fold.
        # gene_in_reconstructed_network_10fold = []

        # reconstruct network per fold.
        # for foldnum in range(2):

        # reconstructed_network is a list of the edges in reconstructed network in the fold (foldnum :fold number for 10fold validation).
        reconstructed_network = []

        # gene_in_reconstructed_network is a set of the genes in reconstructed network in the fold (foldnum :fold number for 10fold validation).
        gene_in_reconstructed_network = set()

        # sort genes by t-statistics.
        test_temp = self.mRNA_ttest
        test_temp.sort(key=lambda tup: tup[0], reverse=True)
        # print(test_temp)
        # CNA_t_sort=self.CNA_ttest[foldnum].sort_values(by=0,ascending=False)
        # met_t_sort=self.met_ttest[foldnum].sort_values(by=0,ascending=False)
        # snp_t_sort=self.snp_ttest[foldnum].sort_values(by=0,ascending=False)

        # selected_by_ttest is a set of the top N genes which have high absolute values of t-statistics in mRNA, CNA, methylation, or SNP data.
        selected_by_ttest = set()
        for i in range(self.n_gene_in_ttest):
            selected_by_ttest.add(test_temp[i])
        top_genes = []
        for tup in selected_by_ttest:
            top_genes.append(self.num2gene[tup[1]])
        # print(top_genes)
        # sys.exit()
        # selected_by_ttest.update(CNA_t_sort.index[:self.n_gene_in_ttest])
        # selected_by_ttest.update(met_t_sort.index[:self.n_gene_in_ttest])
        # selected_by_ttest.update(snp_t_sort.index[:self.n_gene_in_ttest])

        # construct a network. include all edges involving at least one of the genes belonging to selected_by_ttest.
        for edge in self.edge_list:
            # print(edge[0])
            if edge[0] in top_genes or edge[1] in top_genes:
                reconstructed_network.append(edge)
                gene_in_reconstructed_network.update(edge)
            # reconstructed_network_10fold.append(reconstructed_network)
            # gene_in_reconstructed_network_10fold.append(gene_in_reconstructed_network)
        return reconstructed_network, gene_in_reconstructed_network

    # step 2-1. make data for GANs.
    # foldnum is fold number.
    # network is the gene in reconstructed network in the fold.
    def mk_data_for_GANs(self, networkgene):
        print("making data")
        # print(networkgene)
        # merge traing samples.
        trainsample = np.hstack((self.good_train_samples, self.bad_train_samples))
        random.seed(0)

        # to suffle between train samples have good prognosis and train sample have bad prognosis.
        random.shuffle(trainsample)

        # result_tmp is list containing the data for GANs.
        result_tmp = []
        # print(self.gene2num)
        # for each gene in reconstructed network, select the dataset with the largest absolute value of t-test statistic.
        for j in networkgene:
            temp = []
            i = self.small_gene2num[j]
            for sample in trainsample:
                id = self.sample2id[sample]
                temp.append(self.mRNA[i][id])
            result_tmp.append(temp)
        # result_tmp.append(self.mRNA.loc[j,trainsample].values.astype('float64'))
        # elif num==1:
        # 	result_tmp.append(self.CNA.loc[j,trainsample].values.astype('float64'))
        # elif num==2:
        # 	result_tmp.append(self.met.loc[j,trainsample].values.astype('float64'))
        # elif num==3:
        # 	result_tmp.append(self.snp.loc[j,trainsample].values.astype('float64'))
        # print(result_tmp)
        # sys.exit()
        print("finished")
        return result_tmp

    # step 2-2. learn reconstructed FIs network using GANs.
    # process_number is process number in multiprocessing.
    # edge_list is the edges of FIs network in the fold.
    # data_for_GANs is the data we made in step 2-1.
    # foldnum is the fold number.
    def Learning_FIsnetwork_GANs(self, edge_list, data_for_GANs,n):

        # creat an adjacency matrix from the reconstructed FIs network.
        def make_adjacencyMatrix_for_GANs(n_genes, edge_list):
            matrix = np.zeros([n_genes, n_genes], dtype=np.float32)
            for edge in edge_list:
                x = gene2num_forGANs[edge[0]]
                y = gene2num_forGANs[edge[1]]
                matrix[x][y] = matrix[y][x] = 1.
            return matrix

        # prepare for GANs.
        def prepare(adjacency_matrix, n_input, n_hidden, n_noise, stddev):
            reconstucted_network_adjacency_matrix = tf.constant(adjacency_matrix)

            # input.
            X = tf.placeholder(tf.float32, [None, n_input])

            # noise for generator.
            Z = tf.placeholder(tf.float32, [None, n_noise])
            # Generator weights
            gw1 = tf.Variable(tf.random_normal([n_noise, n_genes], stddev=0.01))
            gw2 = tf.Variable(tf.random_normal([n_genes, n_genes], stddev=0.01))
            gw3 = tf.Variable(tf.random_normal([n_genes, n_genes], stddev=0.01))


            # Discriminator weights
            D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
            D_W2 = tf.Variable(tf.random_normal([n_hidden, int(n_hidden/2)], stddev=0.01))
            D_W3 = tf.Variable(tf.random_normal([int(n_hidden/2), 1], stddev=0.01))

            # Set up weight summary
            tf.summary.histogram("Discriminator weights 1", D_W1)
            tf.summary.histogram("Discriminator weights 2", D_W2)
            tf.summary.histogram("Generator weights 1", gw1)
            tf.summary.histogram("Generator weights 2", gw2)


            return reconstucted_network_adjacency_matrix, X, Z, gw1,gw2,gw3, D_W1, D_W2,D_W3

        # generator of GANs.
        def generator(gw1, gw2,gw3, reconstucted_network_adjacency_matrix, noise_z):
            hidden = tf.nn.relu(tf.matmul(noise_z, reconstucted_network_adjacency_matrix * (gw1 * tf.transpose(gw1))))
            hidden2 = tf.nn.relu(tf.matmul(hidden,reconstucted_network_adjacency_matrix * (gw2 * tf.transpose(gw2))))
            output = tf.nn.relu(tf.matmul(hidden2,gw3))
            return output


        # discriminator of GANs.
        def discriminator(inputs, D_W1, D_W2,D_W3):
            hidden = tf.nn.relu(tf.matmul(inputs, D_W1))
            hidden2 = tf.nn.relu(tf.matmul(hidden, D_W2))
            output = tf.nn.relu(tf.matmul(hidden2, D_W3))
            return output

        # make random variables for generator.
        def get_noise(batch_size, n_noise):
            return np.random.normal(size=(batch_size, n_noise))

        print(' start process')

        # get a set of genes from reconstructed FIs network.
        total_gene = []
        for i in edge_list:
            total_gene.append(i[0])
            total_gene.append(i[1])
        total_gene = set(total_gene)

        # make series to map between genes and genes number only for GANs.
        gene2num_forGANs = {}
        num2gene_forGANs = {}
        for i, gene in enumerate(total_gene):
            gene2num_forGANs[gene] = i
            num2gene_forGANs[i] = gene

        # n_genes is the length of genes from the reconstructed FIs network.
        n_genes = len(total_gene)

        data_for_GANs = np.array(data_for_GANs)
        data_for_GANs = data_for_GANs.T
        print(data_for_GANs.shape)

        # creat an adjacency matrix from the reconstructed FIs network.
        adjacency_matrix = make_adjacencyMatrix_for_GANs(n_genes, edge_list)

        # set the parameters.
        tf.set_random_seed(0)
        batch_size = 1
        learning_rate = 0.0002
        epsilon = 1e-4
        LAMBDA = 10
        # reconstucted_network_adjacency_matrix is an adjacency matrix of the reconstructed FIs network.
        reconstucted_network_adjacency_matrix, X, Z, gw1,gw2,gw3, D_W1, D_W2,D_W3 = prepare(adjacency_matrix, n_genes, 512, n_genes,
                                                                               0.01)

        G = generator(gw1, gw2,gw3, reconstucted_network_adjacency_matrix, Z)

        D_gene = discriminator(G, D_W1, D_W2,D_W3)
        # D_gene = D_gene.assign( tf.where (tf.equal(D_gene, tf.constant(0)), tf.constant(epsilon), D_gene) )
        D_real = discriminator(X, D_W1, D_W2,D_W3)
        # D_real = D_real.assign( tf.where (tf.equal(D_real, tf.constant(0)), tf.constant(epsilon), D_real) )
        # loss function.
        loss_D = -tf.reduce_mean(D_real)+tf.reduce_mean(D_gene)
        D_var_list = [D_W1, D_W2,D_W3]
        G_var_list = [gw1,gw2,gw3]

        # define optimizer.
        # loss_G = -tf.reduce_mean(D_gene)
        # loss_D = tf.reduce_mean(D_gene) - tf.reduce_mean(D_real)

        alpha = tf.random_uniform(
            shape=[batch_size, 1],
            minval=0.,
            maxval=1.
        )
        differences = G-X
        interpolates = X + (alpha * differences)
        gradients = tf.gradients(discriminator(interpolates,D_W1,D_W2,D_W3), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        loss_D += LAMBDA * gradient_penalty
        loss_G =-tf.reduce_mean(D_gene)
        train_G = tf.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.5,
            beta2=0.9
        ).minimize(loss_G, var_list=G_var_list)
        train_D = tf.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.5,
            beta2=0.9
        ).minimize(loss_D, var_list=D_var_list)

        n_iter = data_for_GANs.shape[0]
        sess = tf.Session()
        writer = tf.summary.FileWriter("./logs/gan_mrna")
        summaries = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        loss_val_D, loss_val_G = 0, 0
        datas = []
        for i in range(n_iter):
            datas.append(i)
        # print("generating compare data")
        # start = process_number*10+foldnum*10
        # for i in range(start,start+10):
        # 	f = open("generated_data/compare_"+str(i)+".txt","w")
        # 	noise = get_noise(1, n_genes).astype("float32")
        # 	out = generator(G_W,reconstucted_network_adjacency_matrix,noise)
        # 	out = out.eval(session = sess)
        # 	out = out.tolist()
        # 	line = ""
        # 	for num in out:
        # 		line+=str(num)
        # 	f.write(line)
        # 	f.close()
        # perform GANs.
        # tf.train.Saver().save(sess,"checkpoint/start.txt")
        loss = open("loss"+str(n)+".txt", "w")
        print("training")
        for epoch in range(10000):
            loss_val_D_list = []
            loss_val_G_list = []
            inds = sample(datas,50)
            if epoch % 100 == 0:
                for i in range(100):
                    f = open("generated_data/model"+str(n)+"data/sample" + str(epoch) + "_" + str(i) + ".txt", "w")
                    noise = get_noise(1, n_genes)
                    out = sess.run([G], feed_dict={Z: noise})
                    # line = ""
                    # test = np.asarray(out)
                    # print(test.shape)
                    # print(len(out[0][0]))
                    for num in out[0][0]:
                        f.write(str(num) + "\n")
                    f.close()
            for i in inds:
                batch_xs = data_for_GANs[i].reshape(1, -1)
                # print(batch_xs)
                # sys.exit()
                # print(batch_xs.shape)
                # sys.exit()
                noise = get_noise(1, n_genes)
                _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
                _, loss_val_G = sess.run([train_G, loss_G], feed_dict={X:batch_xs,Z: noise})
                # _, loss_val_D, summary1 = sess.run([train_D, loss_D, summaries], feed_dict={X: batch_xs, Z: noise})
                # _, loss_val_G, summary2 = sess.run([train_G, loss_G, summaries], feed_dict={Z: noise})
                loss_val_D_list.append(loss_val_D)
                loss_val_G_list.append(loss_val_G)
                # print(loss_val_D)
                # sys.exit()
                # if i%10 == 0:
                #     writer.add_summary(summary1, (i + 209 * epoch))
                #     writer.add_summary(summary2, (i + 209 * epoch))
                # writer.add_summary(lossD,(i+209*epoch))
                # writer.add_summary(lossG,(i+209*epoch))
            loss.write(str(np.mean(loss_val_D_list)) + "\t" + str(np.mean(loss_val_G)) + "\n")
            print(str(np.mean(loss_val_D_list)) + "\t" + str(np.mean(loss_val_G)) + "\n")
            print(str(epoch))
        # tf.train.Saver().save(sess,"checkpoint/model.txt")

        print(' converge ', 'Epoch:', '%04d' % (epoch + 1), 'n_iter :',
              '%04d' % n_iter, 'D_loss : {:.4}'.format(np.mean(loss_val_D_list)),
              'G_loss : {:.4}'.format(np.mean(loss_val_G_list)))
        sess.close()

    # example = data_for_GANs[0].reshape(1,-1)
    # ex = example[0]
    # f = open("example_data.txt","w")
    # line = ""
    # for gene in ex:
    # 	line+=str(gene)+","
    # f.write(line)
    # f.close()

    # create an adjacency matrix form edge_list.
    def make_adjacencyMatrix(self, edge_list):
        n_genes = len(self.mRNA)
        matrix = np.zeros([n_genes, n_genes], dtype=np.float32)
        count = 0
        for edge in edge_list:
            x = self.small_gene2num[edge[0]]
            y = self.small_gene2num[edge[1]]
            matrix[x][y] = 1
            matrix[y][x] = 1
        return matrix


if __name__ == "__main__":
    main()
