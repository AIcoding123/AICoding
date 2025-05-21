import os

import numpy as np
import pandas as pd

def merge_excel_files(glm4_folder, sliding_folder):
    # 存储合并后的数据框
    merged_art = []
    merged_science = []

    # 处理 art 文件
    for i in range(1, 4):  # prompt1, prompt2, prompt3
        glm4_file = os.path.join(glm4_folder, f'art_prompt{i}_combined_data.xlsx')
        sliding_file = os.path.join(sliding_folder, f'art_prompt{i}_combined_data.xlsx')

        # 读取 Excel 文件
        glm4_df = pd.read_excel(glm4_file)
        sliding_df = pd.read_excel(sliding_file)

        # 构建合并数据框，只横向连接 IFIAS_Code 列
        combined_df = pd.DataFrame()
        combined_df['课堂对话'] = glm4_df['课堂对话']
        combined_df['Finegrained'] = glm4_df['Finegrained']

        # 为 IFIAS_Code 列重命名并添加到合并数据框中
        combined_df[f'prompt{i}glm4'] = glm4_df['Finegrained_Code']
        combined_df[f'prompt{i}slidingglm4'] = sliding_df['Finegrained_Code']

        # 添加到合并数据列表中
        merged_art.append(combined_df)

    # 处理 science 文件
    for i in range(1, 4):  # prompt1, prompt2, prompt3
        glm4_file = os.path.join(glm4_folder, f'science_prompt{i}_combined_data.xlsx')
        sliding_file = os.path.join(sliding_folder, f'science_prompt{i}_combined_data.xlsx')

        # 读取 Excel 文件
        glm4_df = pd.read_excel(glm4_file)
        sliding_df = pd.read_excel(sliding_file)

        # 构建合并数据框，只横向连接 IFIAS_Code 列
        combined_df = pd.DataFrame()
        combined_df['课堂对话'] = glm4_df['课堂对话']
        combined_df['Finegrained'] = glm4_df['Finegrained']

        # 为 IFIAS_Code 列重命名并添加到合并数据框中
        combined_df[f'prompt{i}glm4'] = glm4_df['Finegrained_Code']
        combined_df[f'prompt{i}slidingglm4'] = sliding_df['Finegrained_Code']

        # 添加到合并数据列表中
        merged_science.append(combined_df)

    # 将合并后的结果保存到新的 Excel 文件
    art_merged_result = pd.concat(merged_art, axis=1)  # 横向连接
    science_merged_result = pd.concat(merged_science, axis=1)  # 横向连接

    # 保存合并的结果
    art_merged_result.to_excel('Merged_Art_Results.xlsx', index=False)
    science_merged_result.to_excel('Merged_Science_Results.xlsx', index=False)

    print("合并结果已保存为 Excel 文件。")

# 定义文件夹路径
glm4_folder = 'resultdataset/qwen72b'
sliding_folder = 'resultdataset/slidingwindowqwen72b'

# 调用函数
# merge_excel_files(glm4_folder, sliding_folder)
# import pandas as pd

# 读取Excel文件
# file_path = 'EvaluateDataset/Qwen2-7B-Instructvoting/Merged_Art_Results.xlsx'  # 替换为你的文件名
# df = pd.read_excel(file_path)


# 定义函数计算vote
# def calculate_vote(row):
#     counts = row.value_counts()
#
#     # 判断最大值和数量
#     if counts.get(12, 0) > 0:  # 12 不计
#         counts = counts.drop(12)
#
#         # 找到出现次数最多的值和对应的索引
#     max_count = counts.max()
#     max_values = counts[counts == max_count].index.tolist()
#
#     if len(max_values) == 1:
#         return max_values[0]  # 只有一个最大值，直接返回
#     else:
#         # 如果有多个最大值，返回第一个的值
#         return max_values[0]
#
#     # 应用函数计算vote
#
# df['vote'] = df.apply(calculate_vote, axis=1)
#
# # 导出结果到新的Excel文件
# output_file_path = 'output_file.xlsx'  #可以修改为你想要的文件名
# df.to_excel(output_file_path, index=False)
##############################################Science
import pandas as pd

# 读取Excel文件
file_path = 'resultdataset/qwen72bvoting/Merged_Art_Results.xlsx'  # 替换为你的文件名
df = pd.read_excel(file_path)


# 定义函数计算vote
# 定义函数计算vote
def calculate_vote(row):
    # 将行数据转换为数值类型，强制转换错误的值为NaN
    numeric_row = pd.to_numeric(row, errors='coerce')

    # 过滤掉非常大的数字（比如>1e20），并且不计算12
    filtered_row = numeric_row[(numeric_row < 1e20) & (numeric_row != 12)]

    counts = filtered_row.value_counts()

    if counts.empty:  # 如果没有有效的计数
        return 0  # 或者返回你喜欢的默认值

    max_count = counts.max()
    max_values = counts[counts == max_count].index.tolist()

    if len(max_values) == 1:
        return max_values[0]  # 只有一个最大值，直接返回
    elif max_values:  # 如果有最大值，但不止一个
        return max_values[0]  # 返回第一个最大值
    return 0  # 如果没有有效的最大值，返回默认值


# 应用函数计算vote
df['vote'] = df.apply(calculate_vote, axis=1)

# 导出结果到新的Excel文件
output_file_path = 'output_file_art.xlsx'  # 可以修改为你想要的文件名
df.to_excel(output_file_path, index=False)