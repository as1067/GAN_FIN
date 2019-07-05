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
    for foldnum in range(2):
        data_for_GANs = pm.mk_data_for_GANs(gene_in_reconstructed_FIs_perfold[foldnum], foldnum)

        score = np.zeros(pm.mRNA.shape[0])
        # multiprocessing.
        # output1 = Queue(); output2 = Queue(); output3 = Queue();output4 = Queue();output5 = Queue();
        # output6 = Queue(); output7 = Queue(); output8 = Queue();output9 = Queue();output10 = Queue();
        # output11 = Queue(); output12 = Queue(); output13 = Queue();output14 = Queue();output15 = Queue();
        # output16 = Queue(); output17 = Queue(); output18 = Queue();output19 = Queue();output20 = Queue();
        # process_list = []
        # Output = [output1, output2, output3,output4,output5,output6, output7, output8,output9,output10,output11, output12, output13,output14,output15,output16, output17, output18,output19,output20]

        # #To select a stable and robust feature for random initialization of weights, repeatedly experiment with the reconstructed network learning-phase using GANs and the PageRank process (t times).
        # #t is n_experiment.
        # for process_number in range(pm.n_experiment) :
        # 	process_list.append(Process(target=pm.Learning_FIsnetwork_GANs, args=(process_number, reconstructed_FIs_perfold[foldnum],data_for_GANs,foldnum, Output[process_number])))

        # for n,p in enumerate(process_list) :
        # 	p.start()
        # result_GANs=[]
        for i in range(pm.n_experiment):
            pm.Learning_FIsnetwork_GANs(i, reconstructed_FIs_perfold[foldnum], data_for_GANs, foldnum,n)
    file = open("gpu_latest.txt","w")
    file.write(str(n+1))
    # for process in process_list :
    # 	process.join()
    # select the genes that appeared more than b times in t experiments as biomarkers.
    # t is n_experiment.
    # b is limit.


# 	print("pagerank")
# 	for i in range(pm.n_experiment):
# 		pagerank_genes=pm.pagerank(result_GANs[i])
# 		for k in pagerank_genes:
# 			score[pm.small_gene2num[k]]=score[pm.small_gene2num[k]]+1
# 	biomarker=[]
# 	genes = []
# 	for i in range(pm.mRNA.shape[0]):
# 		genes.append(pm.num2gene[i])
# 	for i,j in zip(score,genes):
# 		if i >=pm.limit:
# 			biomarker.append(j)
# 	biomarker_perfold.append(biomarker)

# #save biomarker
# f = open("Lung_Adenocarcino_biomarkers.txt", 'w')
# for foldnum in range(10):
# 	f.write("\nFold Number(10 fold validation) : %d\n" % foldnum)
# 	for gene in biomarker_perfold[foldnum]:
# 		f.write("%s\t" % gene)
# f.close()
# print("----------------------------------------------------------------------------------------------------")
# print("4. Step4 : Prognosis Prediction")
# pm.auc(reconstructed_FIs_perfold,biomarker_perfold)


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
    column_wise = data.T
    zscored_data = np.zeros((len_row_gene, len_column_sample))
    for column_sample in range(len_column_sample):
        mu = statistics.mean(column_wise[column_sample])
        sigma = statistics.stdev(column_wise[column_sample])
        if mu != 0 and sigma != 0:
            for row_gene in range(len_row_gene):
                x = data[row_gene][column_sample]
                zscored_data[row_gene][column_sample] = (x - mu) / sigma
        else:
            print('Warning!z-scoring!')

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
    mRNA = zscore(raw_mRNA2)
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
    print(' divide samples for 2fold validation ')
    good_sam, bad_sam = seperate_good_bad_patients(lable)
    good_sam = np.array(good_sam)
    bad_sam = np.array(bad_sam)

    kf = KFold(n_splits=2, random_state=None, shuffle=False)

    good_train_samples = []
    bad_train_samples = []
    test_samples = []
    for good_index, bad_index in zip(kf.split(good_sam), kf.split(bad_sam)):
        good_train, good_test = good_sam[good_index[0]], good_sam[good_index[1]]
        bad_train, bad_test = bad_sam[bad_index[0]], bad_sam[bad_index[1]]
        good_train_samples.append(good_train)
        bad_train_samples.append(bad_train)
        test_tmp = np.hstack((good_test, bad_test))
        test_samples.append(test_tmp)

    # perform a t-test for each fold.
    mRNA_ttest = []

    for foldnum in range(2):
        print(' ' + str(foldnum) + 'fold ttest start')
        goodsam = good_train_samples[foldnum]
        badsam = bad_train_samples[foldnum]
        good_ids = []
        bad_ids = []
        for sample in goodsam:
            good_ids.append(sample2id[sample])
        for sample in badsam:
            bad_ids.append(sample2id[sample])
        mRNA_ttmp = t_test(mRNA, num2gene, good_ids, bad_ids)
        mRNA_ttest.append(mRNA_ttmp)
    # CNA_ttest.append(CNA_ttmp)
    # met_ttest.append(met_ttmp)
    # snp_ttest.append(snp_ttmp)

    # make instance of class PM.
    Pm = PM(n_gene_in_ttest, n_biomarker, damping_factor, n_experiment, n_limit, mRNA, lable, edge_list,
            good_train_samples, bad_train_samples, test_samples, gene2num, num2gene, mRNA_ttest, sample2id,
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
        reconstructed_network_10fold = []

        # gene_in_reconstructed_network_10fold is list containing sets of the genes in reconstructed network per fold.
        gene_in_reconstructed_network_10fold = []

        # reconstruct network per fold.
        for foldnum in range(2):

            # reconstructed_network is a list of the edges in reconstructed network in the fold (foldnum :fold number for 10fold validation).
            reconstructed_network = []

            # gene_in_reconstructed_network is a set of the genes in reconstructed network in the fold (foldnum :fold number for 10fold validation).
            gene_in_reconstructed_network = set()

            # sort genes by t-statistics.
            test_temp = self.mRNA_ttest[foldnum]
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
            reconstructed_network_10fold.append(reconstructed_network)
            gene_in_reconstructed_network_10fold.append(gene_in_reconstructed_network)
        return reconstructed_network_10fold, gene_in_reconstructed_network_10fold

    # step 2-1. make data for GANs.
    # foldnum is fold number.
    # network is the gene in reconstructed network in the fold.
    def mk_data_for_GANs(self, networkgene, foldnum):
        print("making data")
        # print(networkgene)
        # merge traing samples.
        trainsample = np.hstack((self.good_train_samples[foldnum], self.bad_train_samples[foldnum]))
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
    def Learning_FIsnetwork_GANs(self, process_number, edge_list, data_for_GANs, foldnum,n):

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
            # G_W is for generator.
            G_W = tf.Variable(tf.random_normal([n_noise, n_genes], stddev=0.01))

            # D_W1 is for discriminator.
            D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))

            # D_W2 is for discriminator.
            D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))

            # Set up weight summary
            tf.summary.histogram("Discriminator weights 1", D_W1)
            tf.summary.histogram("Discriminator weights 2", D_W2)
            tf.summary.histogram("Generator weights", G_W)

            return reconstucted_network_adjacency_matrix, X, Z, G_W, D_W1, D_W2

        # generator of GANs.
        def generator(G_W, reconstucted_network_adjacency_matrix, noise_z):
            output = tf.nn.relu(tf.matmul(noise_z, reconstucted_network_adjacency_matrix * (G_W * tf.transpose(G_W))))
            return output

        def get_weights():
            return G_W

        # discriminator of GANs.
        def discriminator(inputs, D_W1, D_W2):
            hidden = tf.nn.relu(tf.matmul(inputs, D_W1))
            output = tf.nn.sigmoid(tf.matmul(hidden, D_W2))
            return output

        # make random variables for generator.
        def get_noise(batch_size, n_noise):
            return np.random.normal(size=(batch_size, n_noise))

        print(' start process	process number : ', process_number, '	fold number :', foldnum)

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
        tf.set_random_seed(process_number)
        batch_size = 1
        learning_rate = 0.002
        epsilon = 1e-4

        # reconstucted_network_adjacency_matrix is an adjacency matrix of the reconstructed FIs network.
        reconstucted_network_adjacency_matrix, X, Z, G_W, D_W1, D_W2 = prepare(adjacency_matrix, n_genes, 512, n_genes,
                                                                               0.01)

        G = generator(G_W, reconstucted_network_adjacency_matrix, Z)

        D_gene = discriminator(G, D_W1, D_W2)
        # D_gene = D_gene.assign( tf.where (tf.equal(D_gene, tf.constant(0)), tf.constant(epsilon), D_gene) )
        D_real = discriminator(X, D_W1, D_W2)
        # D_real = D_real.assign( tf.where (tf.equal(D_real, tf.constant(0)), tf.constant(epsilon), D_real) )
        # loss function.
        loss_D = tf.reduce_mean(tf.log(tf.cosh(1-D_real)) + tf.log(tf.cosh(D_gene)))
        loss_G = tf.reduce_mean(tf.log(tf.cosh(1-D_gene)))
        D_var_list = [D_W1, D_W2]
        G_var_list = [G_W]

        # define optimizer.
        train_D = tf.train.AdamOptimizer(learning_rate).minimize(loss_D, var_list=D_var_list)
        train_G = tf.train.AdamOptimizer(learning_rate).minimize(loss_G, var_list=G_var_list)

        n_iter = data_for_GANs.shape[0]
        sess = tf.Session()
        writer = tf.summary.FileWriter("./logs/gan_mrna")
        summaries = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        loss_val_D, loss_val_G = 0, 0
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
        loss = open("loss"+str(n)+"_" + str(foldnum) + ".txt", "w")
        print("training")
        for epoch in range(100):
            loss_val_D_list = []
            loss_val_G_list = []
            for i in range(n_iter):
                batch_xs = data_for_GANs[i].reshape(1, -1)
                # print(batch_xs)
                # print(batch_xs.shape)
                # sys.exit()
                noise = get_noise(1, n_genes)
                # _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
                # _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})
                _, loss_val_D, summary1 = sess.run([train_D, loss_D, summaries], feed_dict={X: batch_xs, Z: noise})
                _, loss_val_G, summary2 = sess.run([train_G, loss_G, summaries], feed_dict={Z: noise})
                loss_val_D_list.append(loss_val_D)
                loss_val_G_list.append(loss_val_G)
                loss.write(str(loss_val_D) + "\t" + str(loss_val_G) + "\n")
                print(str(loss_val_D) + "\t" + str(loss_val_G) + "\n")
                # print(loss_val_D)
                # sys.exit()
                writer.add_summary(summary1, (i + 209 * epoch))
                writer.add_summary(summary2, (i + 209 * epoch))
                # writer.add_summary(lossD,(i+209*epoch))
                # writer.add_summary(lossG,(i+209*epoch))
                print(str(i))
        # tf.train.Saver().save(sess,"checkpoint/model.txt")

        print(' process ' + str(process_number) + ' converge ', 'Epoch:', '%04d' % (epoch + 1), 'n_iter :',
              '%04d' % n_iter, 'D_loss : {:.4}'.format(np.mean(loss_val_D_list)),
              'G_loss : {:.4}'.format(np.mean(loss_val_G_list)))
        print("generating data")
        start = process_number * 10 + foldnum * 10
        for i in range(start, start + 10):
            f = open("generated_data/sample_" + str(i) + ".txt", "w")
            noise = get_noise(1, n_genes)
            out = sess.run([G], feed_dict={Z: noise})
            line = ""
            test = np.asarray(out)
            print(test.shape)
            print(len(out[0][0]))
            for num in out[0][0]:
                line += str(num) + ","
            f.write(line)
            f.close()
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
