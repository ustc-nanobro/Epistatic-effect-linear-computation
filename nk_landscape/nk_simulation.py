import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


class NKModel:
    def __init__(
        self,
        seq_len=11,
        library="ACFGHIKLMNPQRSTVWY",
        seed=None,
        input_csv="",
        model="",
    ):
        self.seq_len = seq_len
        self.library = library
        self.seed = seed
        self.f_mem = {}
        self.epi_net = self.genEpiNet(self.seq_len, 3)  # 默认连接数为3
        self.input_seuence = input_csv
        self.model = model

    def gen_distance_subsets(self, ruggedness):
        land_K2, seq, _ = self.makeNK(
            self.seq_len, ruggedness
        )  
        if not self.seed:
            self.seed = np.array(
                [x for x in "".join([self.library[0] for x in range(self.seq_len)])]
            )  
        subsets = {
            x: [] for x in range(self.seq_len + 1)
        }  
        for seq in land_K2:  
            subsets[self.hamming(seq[0], self.seed)].append(
                seq
            )  
        return subsets

    def collapse_single(self, protein):
        return "".join([str(i) for i in protein])

    def hamming(self, str1, str2):
        return sum(c1 != c2 for c1, c2 in zip(str1, str2))

    def genEpiNet(self, N, K):
        return {
            i: sorted(
                np.random.choice(
                    [n for n in range(N) if n != i], K, replace=False
                ).tolist()
                + [i]
            )
            for i in range(N)
        }

    def fitness_i(self, sequence, i):
        epi_indices = self.epi_net[i]
        epi_values = sequence[epi_indices]
        key = tuple(zip(epi_indices, epi_values))
        if key not in self.f_mem:
            if self.model == "uniform":
                self.f_mem[key] = np.random.uniform(0, 1)
            elif self.model == "exp":
                self.f_mem[key] = np.random.exponential(scale=1.0)
            elif self.model == "normal":
                self.f_mem[key] = np.random.normal(0, 1)
            else:
                raise ValueError(
                    "Invalid model parameter. Choose from 'uniform', 'exp', 'zipf', or 'normal'."
                )
        return self.f_mem[key]

    def fitness(self, sequence):
        return np.mean(
            [self.fitness_i(sequence, i) for i in range(len(sequence))]  # ω_i
        )

    def makeNK(self, N, K):
        self.f_mem = {}
        self.epi_net = self.genEpiNet(N, K)
        df = pd.read_csv(self.input_seuence)
        rhla_sequence = df["Sequence"]
        rhla_sequence = rhla_sequence.to_numpy()
        split_strings = [list(s) for s in rhla_sequence]
        converted_array = np.array(split_strings)
        sequenceSpace = converted_array
        land = [
            (x, y)
            for x, y in zip(sequenceSpace, [self.fitness(i) for i in sequenceSpace])
        ]
        return land, sequenceSpace, self.epi_net

    def dataset_generation(
        self, directory="/Users/mac/code/NK_Benchmarking/rhla13_exp"
    ):  
        if not os.path.exists(directory):
            os.mkdir(directory)  
        for ruggedness in range(0, 1):
            for instance in range(self.seq_len):
                print("Generating data for K={} V={}".format(ruggedness, instance))
                subsets = self.gen_distance_subsets(ruggedness)
                hold = []  
                for i in subsets.values():
                    for j in i:
                        hold.append(
                            [self.collapse_single(j[0]), j[1]]
                        )  
                saved = np.array(hold)
                df = pd.DataFrame({"Sequence": saved[:, 0], "Fitness": saved[:, 1]})
                output_path = os.path.join(
                    directory, "K{0}/V{1}.csv".format(ruggedness, instance)
                )
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))
                df.to_csv(output_path)
        print("All data generated. Data is stored in: {}".format(directory))

# Example usage
if __name__ == "__main__":
    # Choose from 'uniform', 'exp', or 'normal'.
    # input_csvs:simulation sequence csv files
    nk_model = NKModel(
        seq_len=11,
        library="ACFGHIKLMNPQRSTVWY",
        seed="SFQRAQLSQHA",
        input_csv="rhla1_3_sequence/rhla1-3.csv",
        model="exp",
    )
    # The address of the saved file
    nk_model.dataset_generation(
        directory="/Users/mac/Desktop/Epistatic-effect-linear-computation/test/test"
    )
