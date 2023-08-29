import pandas as pd
import numpy as np
import os

class NKModel:
    def __init__(self, seq_len=11, library="ACFGHIKLMNPQRSTVWY", seed=None,input_csv='',model=''):
        self.seq_len = seq_len
        self.library = library
        self.seed = seed
        self.f_mem = {}
        self.epi_net = self.genEpiNet(self.seq_len, 3)  # 默认连接数为3
        self.input_seuence = input_csv
        self.model = model

    def gen_distance_subsets(self, ruggedness):
        # Implementation of gen_distance_subsets function here
        land_K2, seq, _ = self.makeNK(
            self.seq_len, ruggedness
        )  # 调用函数 makeNK，生成一个 NK 风景模型。land_K2 是生成的风景模型，seq 是生成的序列

        if not self.seed:
            self.seed = np.array(
                [x for x in "".join([self.library[0] for x in range(self.seq_len)])]
            )  # 生成种子序列，长度为 seq_len，由 library 的第一个字符重复组成。

        subsets = {
            x: [] for x in range(self.seq_len + 1)
        }  # 初始化一个空字典 subsets，用于存储不同距离子集的数据{0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        for seq in land_K2:  # 遍历生成的风景模型中的每个序列
            subsets[self.hamming(seq[0], self.seed)].append(
                seq
            )  # 将当前序列添加到 subsets 字典中，根据其与种子序列的汉明距离分配到相应的距离子集中
        return subsets
    def collapse_single(self, protein):
        """
        Takes any iterable form of a single amino acid character sequence and returns a string representing that sequence.
        
        Args:
            protein (iterable): An iterable containing amino acid characters.
        
        Returns:
            str: A string representing the sequence of amino acids.
        """
        return "".join([str(i) for i in protein])


    def hamming(self, str1, str2):
        """
        Calculates the Hamming distance between two strings.

        Args:
            str1 (str): The first string.
            str2 (str): The second string.

        Returns:
            int: The Hamming distance between the two strings.
        """
        return sum(c1 != c2 for c1, c2 in zip(str1, str2))

    def genEpiNet(self, N, K):
        """
        Generates a random epistatic network for a sequence of length N with, on average, K connections.
        
        Args:
            N (int): The length of the sequence.
            K (int): The average number of connections for each position.
        
        Returns:
            dict: A dictionary representing the epistatic network, where keys are position indices
                  and values are lists of indices representing connections.
        """
        return {
            i: sorted(
                np.random.choice([n for n in range(N) if n != i], K, replace=False).tolist()
                + [i]
            )
            for i in range(N)
        }


    def fitness_i(self, sequence, i):
        """
        Assigns a (random) fitness value to the ith amino acid that interacts with K other positions in a sequence.

        Args:
            sequence (str): The sequence of amino acids.
            i (int): The index of the amino acid to which fitness value is assigned.

        Returns:
            float: The fitness value assigned to the ith amino acid.
        """
        # We use the epistasis network to work out what the relation is
        epi_indices = self.epi_net[i]
        epi_values = sequence[epi_indices]

        # Then, we assign a random number to this interaction
        key = tuple(zip(epi_indices, epi_values))
        if key not in self.f_mem:
            if self.model == 'uniform':
                self.f_mem[key] = np.random.uniform(0, 1)
            elif self.model == 'exp':
                self.f_mem[key] = np.random.exponential(scale=1.0)
            elif self.model == 'zipf':
                self.f_mem[key] = float(np.random.zipf(a=2.))
            elif self.model == "normal":    
                 self.f_mem[key] =np.random.normal(0,1)
            else:
                raise ValueError("Invalid model parameter. Choose from 'uniform', 'exp', 'zipf', or 'normal'.")
        return self.f_mem[key]
    def fitness(self, sequence):
        """
        Obtains a fitness value for the entire sequence by summing over individual amino acids.

        Args:
            sequence (str): The sequence of amino acids.

        Returns:
            float: The fitness value of the entire sequence.
        """
        return np.mean(
            [self.fitness_i(sequence, i) for i in range(len(sequence))]  # ω_i
        )
    
    def makeNK(self, N, K):
        """Make NK landscape with above parameters"""
        self.f_mem = {}
        self.epi_net = self.genEpiNet(N, K)
        df = pd.read_csv(self.input_seuence)
        rhla_sequence = df["Sequence"]
        rhla_sequence = rhla_sequence.to_numpy()
        
        # 将字符串数组拆分成单个字符
        split_strings = [list(s) for s in rhla_sequence]
        
        # 转换为二维数组形式
        converted_array = np.array(split_strings)
        sequenceSpace = converted_array
        
        # 创建一个列表 land，其中每个元素是一个元组，包含基因型序列和其对应的表现型值。
        land = [
            (x, y)
            for x, y in zip(
                sequenceSpace, [self.fitness(i) for i in sequenceSpace]
            )
        ]
        return land, sequenceSpace, self.epi_net


    def dataset_generation(self, directory="/Users/mac/code/NK_Benchmarking/rhla13_exp"):
        """
        Generates five instances of each possible ruggedness value for the NK landscape
        
        Args:
            directory (str): The directory where the generated data will be stored.
            seq_len (int): The length of the sequences.
        """
        if not os.path.exists(directory):
            os.mkdir(directory)  # 检查指定的数据存储目录是否存在，如果不存在则创建目录。
        
        for ruggedness in range(0, 11):
            for instance in range(self.seq_len):
                print("Generating data for K={} V={}".format(ruggedness, instance))
                subsets = self.gen_distance_subsets(ruggedness)
                hold = []  # 初始化一个空列表 hold，用于临时存储每个实例的数

                for i in subsets.values():
                    for j in i:
                        hold.append(
                            [self.collapse_single(j[0]), j[1]]
                        )  # 循环遍历 subsets 中的每个子集：将子集中的每个元素（即序列和适应度）存储为 [collapse_single(j[0]), j[1]] 的形式，并添加到 hold 列表中

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
    # Choose from 'uniform', 'exp', 'zipf', or 'normal'.
    # input_csvs:simulation sequence csv files
    nk_model = NKModel(seq_len=11, library="ACFGHIKLMNPQRSTVWY", seed="SFQRAQLSQHA",input_csv="rhla1_3_sequence/rhla1-3.csv",model='zipf')
    # The address of the saved file
    nk_model.dataset_generation(directory="/Users/mac/Desktop/Epistatic-effect-linear-computation/rhla_data_simulation/rhla13_zipf")
