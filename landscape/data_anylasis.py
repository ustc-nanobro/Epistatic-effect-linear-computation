import pandas as pd
import numpy as np
from landscape import Protein_Landscape
class ProteinLandscapeAnalyzer:
    def __init__(self, csv_folder,csv_output):
        self.csv_folder = csv_folder
        self.csv_output = csv_output

    def analyze(self, k):
        test_results = []
        for v in range(10):
            
            csv_path = f"{self.csv_folder}/K{k}/V{v}.csv"
            test = Protein_Landscape(csv_path=csv_path).get_rs_ex()
            test_results.append(test[-4:])  # Extract the last four elements

        linear_slope_elements, linear_RMSE_elements, extrema_ruggedness_elements, RS_ruggedness_elements = zip(
            *test_results
        )

        df = pd.DataFrame({
            "linear_slope_elements": linear_slope_elements,
            "linear_RMSE_elements": linear_RMSE_elements,
            "extrema_ruggedness_elements": extrema_ruggedness_elements,
            "RS": RS_ruggedness_elements,
        })

        output_file = f"{self.csv_output}/k{k}.xlsx"
        df.to_excel(output_file, index=False)
        print(f"Data written to {output_file}")

        e_rs_std_deviation = np.std(extrema_ruggedness_elements)
        e_rs_mean = np.mean(extrema_ruggedness_elements)

        rs_std_deviation = np.std(RS_ruggedness_elements)
        rs_mean = np.mean(RS_ruggedness_elements)
        rs_median_nk = np.median(RS_ruggedness_elements)
        rs_extremes_removed = np.sort(RS_ruggedness_elements)[1:-1]  # Remove extremes
        rs_average_without_extremes = np.mean(rs_extremes_removed)

        output_file = f"{self.csv_output}/result.txt"
        with open(output_file, "a") as file:
            file.write(f"k={k} Standard Deviation: {rs_std_deviation}\n")
            file.write(f"k={k} Median: {rs_median_nk}\n")
            file.write(f"k={k} Mean: {rs_mean}\n")
            file.write(f"k={k} Mean without extremes: {rs_average_without_extremes}\n")
            file.write(f"k={k} Extrema RS Mean: {e_rs_mean}\n")

        print(f"Median and mean saved to {output_file}")

if __name__ == "__main__":
    # csv_folder = "/Users/mac/Desktop/Epistatic-effect-linear-computation/rhla_data_simulation/rhla16_exp"
    # result_output = "/Users/mac/Desktop/Epistatic-effect-linear-computation/rhla_analysis/rhla1_6_exp_result"
    # analyzer = ProteinLandscapeAnalyzer(csv_folder,result_output)
    
    # for k in range(11):
    #     analyzer.analyze(k)

    # 创建一个包含不同目录路径的列表
    csv_folders = [
        "/Users/mac/Desktop/Epistatic-effect-linear-computation/rhla_data_simulation/rhla16_exp",
        "/Users/mac/Desktop/Epistatic-effect-linear-computation/rhla_data_simulation/rhla13_exp",
        "/Users/mac/Desktop/Epistatic-effect-linear-computation/rhla_data_simulation/rhla16_uniform",
        "/Users/mac/Desktop/Epistatic-effect-linear-computation/rhla_data_simulation/rhla13_uniform",
        "/Users/mac/Desktop/Epistatic-effect-linear-computation/rhla_data_simulation/rhla16_zipf",
        "/Users/mac/Desktop/Epistatic-effect-linear-computation/rhla_data_simulation/rhla13_zipf",
        "/Users/mac/Desktop/Epistatic-effect-linear-computation/rhla_data_simulation/rhla16_normal",
        "/Users/mac/Desktop/Epistatic-effect-linear-computation/rhla_data_simulation/rhla13_normal",
    ]

    # 创建一个包含不同输出路径的列表
    result_outputs = [
        "/Users/mac/Desktop/Epistatic-effect-linear-computation/rhla_analysis/rhla1_6_exp_result",
        "/Users/mac/Desktop/Epistatic-effect-linear-computation/rhla_analysis/rhla1_3_exp_result",
        "/Users/mac/Desktop/Epistatic-effect-linear-computation/rhla_analysis/rhla1_6_uniform_result",
        "/Users/mac/Desktop/Epistatic-effect-linear-computation/rhla_analysis/rhla1_3_uniform_result",
        "/Users/mac/Desktop/Epistatic-effect-linear-computation/rhla_analysis/rhla1_6_zipf_result",
        "/Users/mac/Desktop/Epistatic-effect-linear-computation/rhla_analysis/rhla1_3_zipf_result",
        "/Users/mac/Desktop/Epistatic-effect-linear-computation/rhla_analysis/rhla1_6_normal_result",
        "/Users/mac/Desktop/Epistatic-effect-linear-computation/rhla_analysis/rhla1_3_normal_result",
    ]

    # 遍历每个目录路径和输出路径
    for csv_folder, result_output in zip(csv_folders, result_outputs):
        # 创建分析器对象并进行分析
        analyzer = ProteinLandscapeAnalyzer(csv_folder, result_output)
        for k in range(11):
            analyzer.analyze(k)

