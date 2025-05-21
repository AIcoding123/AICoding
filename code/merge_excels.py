import os
import pandas as pd


def merge_excels(input_dir, output_file):
    # 获取目录下所有xlsx文件
    files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]
    all_data = []

    for file in files:
        file_path = os.path.join(input_dir, file)

        # 读取Excel文件，默认读取第一个sheet
        df = pd.read_excel(file_path)

        # 可以添加一列记录来源文件名，方便追溯
        df['source_file'] = file

        all_data.append(df)

        # 合并所有DataFrame
    merged_df = pd.concat(all_data, ignore_index=True)

    # 保存到新的Excel文件
    merged_df.to_excel(output_file, index=False)
    print(f'已合并{len(files)}个文件，保存至 {output_file}')


if __name__ == "__main__":
    input_directory = r"E:\fairy\运河杯\大创\开始写\大语言模型\编码\coding data"  # 输入文件夹路径
    output_filepath = r"E:\fairy\运河杯\大创\开始写\大语言模型\编码\coding data\Merged_output.xlsx"  # 合并后输出文件路径

    merge_excels(input_directory, output_filepath)