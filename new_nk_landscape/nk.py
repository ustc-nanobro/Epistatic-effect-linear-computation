from typing import Sequence, Dict
import os
import numpy as np
import csv
import sequence_utils as s_utils
from landscape import Landscape
import pandas as pd
_filedir = os.path.dirname(__file__)
class NK(Landscape):
    def __init__(
        self,
        N,
        K,
        epi='uni',
        pos_weight='one',
        cut_off=None,
        alphabet=s_utils.AAS,
        csv_file = "data.csv"
    ):
        assert K < N
        assert epi in ['uni', 'exp', 'zipf']
        assert pos_weight in ['one', 'uni', 'exp', 'zipf']
        super().__init__(name=f"N{N}K{K}")
        self.alphabet = alphabet
        self.csv_file = csv_file
        if True:
            self._epi = epi
            self._rng = np.random.default_rng()
            if pos_weight == 'one':
                weight = np.ones(N)
            elif pos_weight == 'uni':
                weight = self._rng.uniform(0, 1, size=N)
            elif pos_weight == 'exp':
                weight = self._rng.exponential(scale=1., size=N)
            elif pos_weight == 'zipf':
                weight = self._rng.zipf(a=2, size=N).astype(float)
            weight /= weight.sum()
            print(weight)
            f_mem = {}
            epi_net = self.genEpiNet(N, K)
            print(epi_net)
            df = pd.read_csv("rhla1_3_sequence/rhla1-3.csv")
            rhla_sequence = df["Sequence"]
            rhla_sequence = rhla_sequence.to_numpy()
            split_strings = [list(s) for s in rhla_sequence]
            converted_array = np.array(split_strings)
            sequenceSpace = converted_array
            score = np.array([
                self.fitness(i, epi=epi_net, mem=f_mem, w=weight) for i in sequenceSpace
            ])
            norm_score = (score - score.min()) / (score.max() - score.min())
            if cut_off is not None:
                score = np.where(norm_score > cut_off, norm_score, cut_off)
                norm_score = (score - score.min()) / (score.max() - score.min())
            seqs = ["".join(seq) for seq in sequenceSpace]
            self._sequences = tuple(zip(seqs, norm_score))
            if not os.path.exists(os.path.dirname(self.csv_file)):
                    os.makedirs(os.path.dirname(self.csv_file))
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Sequence', 'Fitness'])
                for row in self._sequences:
                    writer.writerow(row)
            print(f"数据已保存到 {self.csv_file}")
    def _fitness_function(self, sequences: Sequence[str]) -> np.ndarray:
        return np.array(
            [self._sequences[seq] for seq in sequences],
            dtype=np.float32
        )
    def genEpiNet(self, N, K):
        return {
            i: sorted(self._rng.choice(
                [n for n in range(N) if n != i],
                K,
                replace=False
            ).tolist() + [i])
            for i in range(N)
        }
    def fitness(self, sequence, epi, mem, w):
        return np.mean([
            self.fitness_i(sequence, i, epi, mem) * w[i]
            for i in range(len(sequence))
        ])
    def fitness_i(self, sequence, i, epi, mem):
        key = tuple(zip(epi[i], sequence[epi[i]]))
        if key not in mem:
            if self._epi == 'uni':
                mem[key] = self._rng.uniform(0., 1.)
            elif self._epi == 'exp':
                mem[key] = self._rng.exponential(scale=1.)
            elif self._epi == 'zipf':
                mem[key] = float(self._rng.zipf(a=2.))
        return mem[key]
if __name__ == "__main__":
    for ruggedness in range(0, 11):
        for v in range(0,11):
            ss = NK(11,ruggedness,'exp',csv_file=f"new_exp_data/rhla1-3/k{ruggedness}/v{v}.csv")
