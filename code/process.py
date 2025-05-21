import os
import pandas as pd

def merge_files_in_category(folder_path: str, category: str, prompt: str) -> pd.DataFrame:
    """
    合并指定类别（Art或Science）的Excel文件，按提示分类。

    :param folder_path: 文件夹路径
    :param category: 类别（"Art" 或 "Science"）
    :param prompt: 提示（如"prompt1"、"prompt2"、"prompt3"）
    :return: 合并后的DataFrame
    """
    data_frames = []

    for num in range(1, 11):  # 对于1到10的系列
        file_name = f'Result{category}{num}_{prompt}_qwen72b.xlsx'
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):  # 检查文件是否存在
            df = pd.read_excel(file_path)
            df['Category'] = category  # 添加类别列
            df['Prompt'] = prompt  # 添加提示列
            data_frames.append(df)
            print(f"读取文件: {file_name}")

    # 合并所有DataFrame
    return pd.concat(data_frames, ignore_index=True)

def merge_all_files(folder_path: str, prompts: list) -> pd.DataFrame:
    """
    合并所有Excel文件，按提示分类，不区分类别。

    :param folder_path: 文件夹路径
    :param prompts: 提示列表（["prompt1", "prompt2", "prompt3"]）
    :return: 合并后的DataFrame
    """
    data_frames = []

    for prompt in prompts:
        for category in ['Art', 'Science']:  # 遍历Art和Science
            combined_category_data = merge_files_in_category(folder_path, category, prompt)
            data_frames.append(combined_category_data)

    # 合并所有DataFrame
    return pd.concat(data_frames, ignore_index=True)

# 使用示例
folder_path = 'processfine-data/qwen72b'  # 设置文件夹路径
prompts = ['prompt1', 'prompt2', 'prompt3']  # 提示列表

# 导出每个prompt的合并数据
for prompt in prompts:
    # 不区分类别的合并数据
    prompt_combined_data = merge_all_files(folder_path, [prompt])
    prompt_combined_data.to_excel(f'{prompt}_combined_data.xlsx', index=False)
    print(f'{prompt}合并后的数据已导出到: {prompt}_combined_data.xlsx')

    # 仅Art的合并数据
    art_combined = merge_files_in_category(folder_path, 'Art', prompt)
    art_combined.to_excel(f'art_{prompt}_combined_data.xlsx', index=False)
    print(f'{prompt}的Art合并后的数据已导出到: art_{prompt}_combined_data.xlsx')

    # 仅Science的合并数据
    science_combined = merge_files_in_category(folder_path, 'Science', prompt)
    science_combined.to_excel(f'science_{prompt}_combined_data.xlsx', index=False)
    print(f'{prompt}的Science合并后的数据已导出到: science_{prompt}_combined_data.xlsx')