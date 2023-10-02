import os
import time
import shutil
import numpy as np


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)
aa_table = np.array([
            ["Alanine", "Ala", "A"],
            ["Arginine", "Arg", "R"],
            ["Asparagine", "Asn", "N"],
            ["Aspartic acid", "Asp", "D"],
            ["Cysteine", "Cys", "C"],
            ["Glutamine", "Gln", "Q"],
            ["Glutamic acid", "Glu", "E"],
            ["Glycine", "Gly", "G"],
            ["Histidine","His", "H"],
            ["Isoleucine", "Ile", "I"],
            ["Leucine", "Leu", "L"],
            ["Lysine", "Lys", "K"],
            ["Methionine", "Met", "M"],
            ["Phenylalanine", "Phe", "F"],
            ["Proline", "Pro", "P"],
            ["Serine", "Ser", "S"],
            ["Threonine", "Thr", "T"],
            ["Tryptophan", "Trp", "W"],
            ["Tyrosine", "Tyr", "Y"],
            ["Valine", "Val", "V"],
            ["STOP", "Stp", "*"]])

codon2aa = {
            'TTT': 'F',
            'TTC': 'F',
            'TTA': 'L',
            'TTG': 'L',

            'TCT': 'S',
            'TCC': 'S',
            'TCA': 'S',
            'TCG': 'S',

            'TAT': 'Y',
            'TAC': 'Y',
            'TAA': '*',
            'TAG': '*',

            'TGT': 'C',
            'TGC': 'C',
            'TGA': '*',
            'TGG': 'W',

            'CTT': 'L',
            'CTC': 'L',
            'CTA': 'L',
            'CTG': 'L',

            'CCT': 'P',
            'CCC': 'P',
            'CCA': 'P',
            'CCG': 'P',

            'CAT': 'H',
            'CAC': 'H',
            'CAA': 'Q',
            'CAG': 'Q',

            'CGT': 'R',
            'CGC': 'R',
            'CGA': 'R',
            'CGG': 'R',

            'ATT': 'I',
            'ATC': 'I',
            'ATA': 'I',
            'ATG': 'M',

            'ACT': 'T',
            'ACC': 'T',
            'ACA': 'T',
            'ACG': 'T',

            'AAT': 'N',
            'AAC': 'N',
            'AAA': 'K',
            'AAG': 'K',

            'AGT': 'S',
            'AGC': 'S',
            'AGA': 'R',
            'AGG': 'R',

            'GTT': 'V',
            'GTC': 'V',
            'GTA': 'V',
            'GTG': 'V',

            'GCT': 'A',
            'GCC': 'A',
            'GCA': 'A',
            'GCG': 'A',

            'GAT': 'D',
            'GAC': 'D',
            'GAA': 'E',
            'GAG': 'E',

            'GGT': 'G',
            'GGC': 'G',
            'GGA': 'G',
            'GGG': 'G'
        }
