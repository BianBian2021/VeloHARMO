#coding=utf-8
import random

import numpy as np
import torch
import io
import utils
from models import *
from Mimo_data import load_data
from scipy import stats
from matplotlib import pyplot as plt
import time
from functools import cache
import logging
logging.basicConfig(level=logging.INFO)

codon_table = {
    "A": ["GCT", "GCC", "GCA", "GCG"],
    "C": ["TGT", "TGC"],
    "D": ["GAT", "GAC"],
    "E": ["GAA", "GAG"],
    "F": ["TTT", "TTC"],
    "G": ["GGT", "GGC", "GGA", "GGG"],
    "H": ["CAT", "CAC"],
    "I": ["ATT", "ATC", "ATA"],
    "K": ["AAA", "AAG"],
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "M": ["ATG"],
    "N": ["AAT", "AAC"],
    "P": ["CCT", "CCC", "CCA", "CCG"],
    "Q": ["CAA", "CAG"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "T": ["ACT", "ACC", "ACA", "ACG"],
    "V": ["GTT", "GTC", "GTA", "GTG"],
    "W": ["TGG"],
    "*": ["TAA", "TAG", "TGA"],
    "Y": ["TAT", "TAC"],
}




POPULATION_SIZE = 100  # maximum population
MAX_GENERATIONS = 1000  # maximum generation
MUTATION_RATE = 0.01  # mutation rate


nts = "ATCG"
codon2idx = {}
for nt1 in nts:
    for nt2 in nts:
        for nt3 in nts:
            codon2idx[nt1 + nt2 + nt3] = len(codon2idx)

aa_table = utils.aa_table
codon2aa = utils.codon2aa
aa2codon= codon_table
onehot_codon = {codon: np.eye(64)[codon2idx[codon]] for codon in codon2idx}
onehot_nt = {nts[i]: np.eye(4)[i] for i in range(len(nts))}
onehot_aa = {aa_table[i, 2]: np.eye(21)[i] for i in range(len(aa_table))}


datafile,index="/Your/work/path/E.coli.summary.add.pro.SS.txt",0
data_X, data_Y, valid_len = load_data(datafile)

list_seq = []

SS = "-STIGEBH"
onehot_ss = {SS[i]: np.eye(len(SS))[i] for i in range(len(SS))}
with open(datafile, "r") as f:
    data = f.readlines()
    name = data[4 * 0 + 0].split('>')[1].split()[0]
    seq = data[4 * 6 + 1].split()
    pro_feature = [list(map(float, e.split(",")[:-1])) for e in data[4 * 6 + 3].split()] #(L,6)
    ss = [e.split(",")[-1] for e in data[4 * 6 + 3].split()]
    ss_feature=[onehot_ss[structure] for structure in ss] #(L,)
list_seq.append(seq)


def seq2feature(seq,aa=True,nt=True,protein=False,ss=True):
    assert len(seq) % 3 == 0
    seq=[seq[i:i+3] for i in range(len(seq)//3)]
    codons = seq
    nts = "".join(codons)
    if "N" in nts:
        raise Exception("Invalid sequence")
    feature=[]
    i=0
    for codon in codons:
        per_condon_feature=onehot_codon[codon]
        if nt:
            for nt in codon:
                per_condon_feature=np.concatenate([per_condon_feature,onehot_nt[nt]]) #+4*3
        if aa:
            per_condon_feature=np.concatenate([per_condon_feature,onehot_aa[codon2aa[codon]]])#+21
        if protein:
            per_condon_feature=np.concatenate([per_condon_feature,pro_feature[i]])
        if ss:
            per_condon_feature=np.append(per_condon_feature,ss_feature[i])
        i+=1
        feature.append(per_condon_feature)
    return torch.tensor(np.array(feature)).unsqueeze(0)
codons = list_seq[0]
aas = [codon2aa[codon] for codon in codons]

target_sequence = "".join(aas)

# load the E.coli model
device = torch.device('cpu')

velo_model1 = torch.load('/Your/work/path/E.coli.fold_0_best_network.pth',map_location=device)

#codons = list_seq[0]
seqs="".join(codons)
#print(seqs)

features=seq2feature(seqs) #(1,L,V)
features= features.to(torch.float32)
label_i = velo_model1(features, torch.tensor([[features.shape[1]]]))
#print(label_i)

#load yeast model
velo_model = torch.load('/Your/work/path/yeast.fold_0_best_network.pth',map_location=device)
def generate_random_rna_sequence():
    # generate a random RNA seq
    # return ''.join(random.choices('UCAG', k=len(target_sequence)))
    return ''.join([random.choice(codon_table[AA]) for AA in target_sequence])
# def evaluate_fitness(rna_sequence):
#     # 评估RNA序�得分
#     protein_sequence = translate_rna_to_protein(rna_sequence)
#     score = sum(1 for a, b in zip(protein_sequence, target_sequence) if a == b)
#     return score
#load the mRNA seq from Ecoli to yeast model and calcualte the similarity of velocity between Ecoli and yeast
@cache
def evaluate_fitness(rna_sequence):
    feature=seq2feature(rna_sequence) #(1,L,V)
    feature= feature.to(torch.float32)
    #print(feature.dtype)
    
    with torch.no_grad():
        label = label_i.reshape(-1).cpu().numpy()
    y_hat = velo_model(feature, torch.tensor([[feature.shape[1]]])) #L
    with torch.no_grad():
        y_hat = y_hat.reshape(-1).cpu().numpy()  # (B,L,1)---->(L,B,1)--->(L*B,)
    score, p = stats.pearsonr(label, y_hat)
    return score

def translate_rna_to_protein(rna_sequence):
    # translate the RNA seq into a protein seq
    protein_sequence = ''
    for i in range(0, len(rna_sequence), 3):
        codon = rna_sequence[i:i+3]
        for amino_acid, codons in codon_table.items():
            if codon in codons:
                protein_sequence += amino_acid
                break
    return protein_sequence


def mutate(rna_sequence):
# mutate the RNA seq
    mutated_rna_sequence = list(rna_sequence)
    for i in range(0,len(rna_sequence),3):
        codon="".join(mutated_rna_sequence[i:i+3])
        if random.random() < MUTATION_RATE:
            mutated_codon=random.choice(aa2codon[codon2aa[codon]])
            mutated_rna_sequence[i:i+3] =list(mutated_codon)
    return ''.join(mutated_rna_sequence)


def plot_GA(start_seq,end_seq):
    with torch.no_grad():
        label = label_i.reshape(-1).cpu().numpy()

    feature1 = seq2feature(start_seq) # (1,L,V)
    feature1= feature1.to(torch.float32)
    y_hat_1 = velo_model(feature1, torch.tensor([[feature1.shape[1]]])) # L
    with torch.no_grad():
        y_hat_1 = y_hat_1.reshape(-1).cpu().numpy() # (B,L,1)---->(L,B,1)--->(L*B,)

    feature2 = seq2feature(end_seq) # (1,L,V)
    feature2= feature2.to(torch.float32)
    y_hat_2 = velo_model(feature2, torch.tensor([[feature2.shape[1]]])) # L
    with torch.no_grad():
        y_hat_2 = y_hat_2.reshape(-1).cpu().numpy() # (B,L,1)---->(L,B,1)--->(L*B,)

    plt.figure(figsize=(30,10))
    plt.plot(y_hat_1, linewidth=2, label='initial_seq')
    plt.plot(y_hat_2, linewidth=2,label='Harmonized_seq')
    plt.plot(label, linewidth=2,label='WT(E.coli)')
    plt.legend()
    plt.savefig('P60595.velocity_pattern_mutation_rate0.01_300generation.1000size.ss.pearson.jpg', bbox_inches='tight', pad_inches=0.2, dpi=400)
    plt.clf() # 清图。
    plt.cla() # 清坐标轴。
    plt.close() # 关窗口

def genetic_algorithm():
    population = [generate_random_rna_sequence() for _ in range(POPULATION_SIZE)]  # initialize population
    population.sort(key=lambda x: evaluate_fitness(x), reverse=True)
    start_seq=population[0]
    for generation in range(MAX_GENERATIONS):
        start_time=time.time()
        population = sorted(population, key=lambda x: evaluate_fitness(x), reverse=True)# sort the popluation
        end_time=time.time()
        logging.info("time elapse:{}".format(end_time-start_time))
        print(f"Generation {generation+1}, Best Fitness: {evaluate_fitness(population[0])}/{len(target_sequence)}")
        if evaluate_fitness(population[0]) == 1:
            break  # terminate the iteration prematurely if the optimal fitness is reached
        next_generation = population[:int(POPULATION_SIZE / 2)]  # select seq individuals with higher fitness
        start_time=time.time()
        for _ in range(int(POPULATION_SIZE / 2)):
            parent1, parent2 = random.sample(population[:int(POPULATION_SIZE / 2)], 2)  # random select parents
            child = parent1[:(len(parent1)//6)*3] + parent2[(len(parent2)//6)*3:]#crossover to generate offsing
            child = mutate(child)  #mutate the offspring
            next_generation.append(child)
        end_time=time.time()
        logging.info("time elapse:{}".format(end_time-start_time))
        population = next_generation
    best_rna_sequence = population[0]  # best mRNA seq
    end_seq=population[0]
    plot_GA(start_seq,end_seq)
    return best_rna_sequence, start_seq, end_seq

#initial_velo= velocity_pattern(best_rna_sequence)
#final_velo = velocity_pattern(best_rna_sequence)


best_rna_sequence, start_seq, end_seq = genetic_algorithm()
print("Harmonized mRNA sequence:", best_rna_sequence)
#print("initial_velo_pattern:",)
print("Corresponding protein sequence:", translate_rna_to_protein(best_rna_sequence))








