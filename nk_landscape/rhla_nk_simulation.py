import pandas as pd
import numpy as np
import os


def gen_distance_subsets(
    ruggedness, seq_len=11, library="ACFGHIKLMNPQRSTVWY", seed=None
):
    """
    Takes a ruggedness, sequence length, and library and produces an NK landscape then separates it
    into distances from a seed sequence.

    ruggedness [int | 0-(seq_len-1)]  : Determines the ruggedness of the landscape
    seq_len : length of all of the sequences
    library : list of possible characters in strings
    seed    : the seed sequence for which distances will be calculated

    returns ->  {distance : [(sequence,fitness)]}
    """

    land_K2, seq, _ = makeNK(
        seq_len, ruggedness
    )  # 调用函数 makeNK，生成一个 NK 风景模型。land_K2 是生成的风景模型，seq 是生成的序列

    if not seed:
        seed = np.array(
            [x for x in "".join([library[0] for x in range(seq_len)])]
        )  # 生成种子序列，长度为 seq_len，由 library 的第一个字符重复组成。

    subsets = {
        x: [] for x in range(seq_len + 1)
    }  # 初始化一个空字典 subsets，用于存储不同距离子集的数据{0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    for seq in land_K2:  # 遍历生成的风景模型中的每个序列
        subsets[hamming(seq[0], seed)].append(
            seq
        )  # 将当前序列添加到 subsets 字典中，根据其与种子序列的汉明距离分配到相应的距离子集中
    return subsets


def dataset_generation(
    directory="/Users/mac/code/NK_Benchmarking/rhla13_exp",
    seq_len=11,
):
    """
    Generates five instances of each possible ruggedness value for the NK landscape

    seq_len
    """

    if not os.path.exists(directory):
        os.mkdir(directory)  # 检查指定的数据存储目录是否存在，如果不存在则创建目录。

    for ruggedness in range(0, 11):
        for instance in range(seq_len):
            print("Generating data for K={} V={}".format(ruggedness, instance))
            subsets = gen_distance_subsets(
                ruggedness, seq_len, seed="SFQRAQLSQHA"
            )  # 来生成 NK 模型的子集，存储在 subsets 变量中
            hold = []  # 初始化一个空列表 hold，用于临时存储每个实例的数

            for i in subsets.values():
                for j in i:
                    hold.append(
                        [collapse_single(j[0]), j[1]]
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


def collapse_single(protein):
    """
    Takes any iterable form of a single amino acid character sequence and returns a string representing that sequence.
    """
    return "".join([str(i) for i in protein])


def hamming(str1, str2):
    """Calculates the Hamming distance between 2 strings"""
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))


"""
def genEpiNet(N, K):：定义了一个名为 genEpiNet 的函数它有两个参数,N 表示位点的数量,K 表示平均连接数。
return {i: ... for i in range(N)}：返回一个字典，字典的键是位点的索引（从 0 到 N-1),值是该位点连接的其他位点的列表。
sorted(np.random.choice(...))：使用 np.random.choice 从 [n for n in range(N) if n != i] 中随机选择 K 个元素，
表示该位点连接的其他位点的索引，这些位点不能是自身。
+ [i]：将位点自身的索引添加到连接列表中，以表示该位点对自身的影响。
for i in range(N)：对每个位点的索引循环执行上述步骤，生成一个字典，表示每个位点连接的其他位点。
总之，这个函数通过随机选择方式生成了一个随机的表现型网络，其中每个位点连接了平均数量为 K 的其他位点。
生成的网络被表示为一个字典，其中每个键表示一个位点，对应的值是连接的其他位点的列表。
这个函数可以通过调用 genEpiNet(N, K) 来生成随机的表现型网络，也可以根据需要进行修改和配置"""


def genEpiNet(N, K):
    """Generates a random epistatic network for a sequence of length
    N with, on average, K connections"""
    return {
        i: sorted(
            np.random.choice([n for n in range(N) if n != i], K, replace=False).tolist()
            + [i]
        )
        for i in range(N)
    }


def fitness_i(sequence, i, epi, mem):
    """Assigns a (random) fitness value to the ith amino acid that
    interacts with K other positions in a sequence,"""
    # we use the epistasis network to work out what the relation is
    key = tuple(zip(epi[i], sequence[epi[i]]))
    # then, we assign a random number to this interaction
    if key not in mem:
        mem[key] = np.random.uniform(0, 1)
        # mem[key] = np.random.exponential(scale=1.0)
        # mem[key] = float(np.random.zipf(a=2.))
    return mem[key]


def fitness(sequence, epi, mem):
    """Obtains a fitness value for the entire sequence by summing
    over individual amino acids"""
    # print(sequence)
    # print(epi)
    # print(mem)
    return np.mean(
        [fitness_i(sequence, i, epi, mem) for i in range(len(sequence))]  # ω_i
    )


def makeNK(N, K):
    """Make NK landscape with above parameters"""
    f_mem = {}
    epi_net = genEpiNet(
        N, K
    )  # 调用函数 genEpiNet，生成一个随机的表现型网络（epistatic network），其中每个位点连接了平均数量为 K 的其他位点。
    df = pd.read_csv("rhla1-3/rhla1-3.csv")
    rhla_1_6 = df["Sequence"]
    rhla_1_6 = rhla_1_6.to_numpy()
    # 将字符串数组拆分成单个字符
    split_strings = [list(s) for s in rhla_1_6]

    # 转换为二维数组形式
    converted_array = np.array(split_strings)
    sequenceSpace = converted_array  # 调用函数 all_genotypes，生成给定位点数量和氨基酸集合下的所有可能基因型组合。
    # 创建一个列表 land，其中每个元素是一个元组，包含基因型序列和其对应的表现型值。
    land = [
        (x, y)
        for x, y in zip(
            sequenceSpace, [fitness(i, epi=epi_net, mem=f_mem) for i in sequenceSpace]
        )
    ]
    return land, sequenceSpace, epi_net
def main():
    dataset_generation()

if __name__ == "__main__":
    main()
    
