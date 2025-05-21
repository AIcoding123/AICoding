import pandas as pd
import os

# 定义文件夹路径
# folder_path = 'newdata'
folder_path = 'processdata'
# 分类    编码     表述
# 1      教师接受情感       0
# 2      教师表扬或者鼓励  0
# 3      教师采纳学生观点  0
# 4      教师提问  0
# 5      教师讲授  0
# 6      教师指令  0
# 7      教师批评或维护权威  0
# 8      学生被动应答  1
# 9      学生主动说话  1
# 10     教师操纵技巧  2
# 11     学生操纵技能  2
# 遍历所有Art和Science的文件
def ten_if ():
    for subject in ['Art', 'Science']:
        for i in range(1, 11):
            file_name = f"{subject}{i}.xlsx"
            file_path = os.path.join(folder_path, file_name)

            # 检查文件是否存在
            if os.path.exists(file_path):
                # 读取Excel文件
                df = pd.read_excel(file_path)
                # 打印“成员3”这一列的内容
                if 'Finegrained' in df.columns:
                    # print(f"文件: {file_name} 的 '成员3' 列:")
                    # print(df['成员3'])
                    # print(f"文件: {file_name} 中 '成员3' 列包含值为10的行.")
                    # 检查是否有值为10的行
                    if (df['Finegrained'] == 10).any():
                        print(f"文件: {file_name} 中 '成员3' 列包含值为10的行.")

                    print()
                else:
                    print(f"文件: {file_name} 中没有 '成员3' 这一列.")
            else:
                print(f"文件: {file_name} 不存在.")

def deleteTen():
    # 定义文件夹路径
    input_folder = 'newdata'  # 输入文件夹
    output_folder = 'processdata'  # 输出文件夹

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历所有Art和Science的文件
    for subject in ['Art', 'Science']:
        for i in range(1, 11):
            file_name = f"{subject}{i}.xlsx"
            file_path = os.path.join(input_folder, file_name)

            # 检查文件是否存在
            if os.path.exists(file_path):
                # 读取Excel文件
                df = pd.read_excel(file_path)

                # 检查是否有“成员3”列
                if '成员3' in df.columns:
                    # 删除“成员3”列中值为10的行
                    df = df[df['成员3'] != 10]

                    # 将“成员3”列更名为“Finegrained”
                    df.rename(columns={'成员3': 'Finegrained'}, inplace=True)

                    # 存储到新的Excel文件
                    output_file_path = os.path.join(output_folder, file_name)
                    df.to_excel(output_file_path, index=False)  # 不保存索引

                    print(f"处理文件: {file_name}，已保存到 {output_folder}")
                else:
                    print(f"文件: {file_name} 中没有 '成员3' 这一列.")
            else:
                print(f"文件: {file_name} 不存在.")

def finegrainedTocoarsegrained():
    # 定义输入和输出文件夹路径
    input_folder = 'processdata'  # 输入文件夹，这里使用之前处理的文件夹
    output_folder = 'Dataset'  # 输出文件夹

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历所有Art和Science的文件
    for subject in ['Art', 'Science']:
        for i in range(1, 11):
            file_name = f"{subject}{i}.xlsx"
            file_path = os.path.join(input_folder, file_name)

            # 检查文件是否存在
            if os.path.exists(file_path):
                # 读取Excel文件
                df = pd.read_excel(file_path)

                # 检查是否有“Finegrained”列
                if 'Finegrained' in df.columns:
                    # 根据“Finegrained”列的值添加“Coarsegrained”列
                    def assign_coarsegrained(value):
                        if 1 <= value <= 7:
                            return 0
                        elif 8 <= value <= 9:
                            return 1
                        elif 11 <= value <= 12:
                            return 2
                        else:
                            return None  # 其他值可以是None或其他默认值

                    df['Coarsegrained'] = df['Finegrained'].apply(assign_coarsegrained)

                    # 存储到新的Excel文件
                    output_file_path = os.path.join(output_folder, file_name)
                    df.to_excel(output_file_path, index=False)  # 不保存索引

                    print(f"处理文件: {file_name}，已保存到 {output_folder}")
                else:
                    print(f"文件: {file_name} 中没有 'Finegrained' 这一列.")
            else:
                print(f"文件: {file_name} 不存在.")

def delete_chengyuan12():
    dataset_folder = 'Dataset'  # Dataset文件夹

    # 遍历所有Art和Science的文件
    for subject in ['Art', 'Science']:
        for i in range(1, 11):
            file_name = f"{subject}{i}.xlsx"
            file_path = os.path.join(dataset_folder, file_name)

            # 检查文件是否存在
            if os.path.exists(file_path):
                # 读取Excel文件
                df = pd.read_excel(file_path)

                # 检查并删除“成员1”和“成员2”这两列
                columns_to_drop = ['成员1', '成员2']
                df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

                # 重新保存到Dataset文件夹
                df.to_excel(file_path, index=False)  # 不保存索引

                print(f"处理文件: {file_name}，已删除 '成员1' 和 '成员2' 列并重新保存。")
            else:
                print(f"文件: {file_name} 不存在.")

def xiu1112To1011():
    dataset_folder = 'Dataset'  # Dataset文件夹

    # 遍历所有Art和Science的文件
    for subject in ['Art', 'Science']:
        for i in range(1, 11):
            file_name = f"{subject}{i}.xlsx"
            file_path = os.path.join(dataset_folder, file_name)

            # 检查文件是否存在
            if os.path.exists(file_path):
                # 读取Excel文件
                df = pd.read_excel(file_path)

                # 删除“Finegrained”列中值为0的行
                df = df[df['Finegrained'] != 0]

                # 将“Finegrained”列中值为11的行改为10
                df.loc[df['Finegrained'] == 12, 'Finegrained'] = 11

                # 重新保存到Dataset文件夹
                df.to_excel(file_path, index=False)  # 不保存索引

                print(f"处理文件: {file_name}，已删除 'Finegrained' 中值为0的行，12已改为11，重新保存。")
            else:
                print(f"文件: {file_name} 不存在.")

def humanfinegrainedTocoarsegrained():
    # # 定义输入和输出文件夹路径
    # input_folder = 'processdata'  # 输入文件夹，这里使用之前处理的文件夹
    output_folder = 'EvaluateDataset/human'  # 输出文件夹
    #
    # # 确保输出文件夹存在
    # os.makedirs(output_folder, exist_ok=True)


    file_name = f"EvaluateDataset/human/All_Data.xlsx"
    # file_path = os.path.join(input_folder, file_name)
    file_name1 = f"All_Data_1.xlsx"
    # 检查文件是否存在
    if os.path.exists(file_name):
        # 读取Excel文件
        df = pd.read_excel(file_name)

        # 检查是否有“Finegrained”列
        if '成员1' in df.columns:
            # 根据“Finegrained”列的值添加“Coarsegrained”列
            def assign_coarsegrained(value):
                if 1 <= value <= 7:
                    return 0
                elif 8 <= value <= 9:
                    return 1
                elif 11 <= value <= 12:
                    return 2
                else:
                    return None  # 其他值可以是None或其他默认值

            df['Coarsegrained_成员1'] = df['成员1'].apply(assign_coarsegrained)
            if '成员2' in df.columns:
                # 根据“Finegrained”列的值添加“Coarsegrained”列
                def assign_coarsegrained(value):
                    if 1 <= value <= 7:
                        return 0
                    elif 8 <= value <= 9:
                        return 1
                    elif 11 <= value <= 12:
                        return 2
                    else:
                        return None  # 其他值可以是None或其他默认值

                df['Coarsegrained_成员2'] = df['成员2'].apply(assign_coarsegrained)
            if '成员3' in df.columns:
                # 根据“Finegrained”列的值添加“Coarsegrained”列
                def assign_coarsegrained(value):
                    if 1 <= value <= 7:
                        return 0
                    elif 8 <= value <= 9:
                        return 1
                    elif 11 <= value <= 12:
                        return 2
                    else:
                        return None  # 其他值可以是None或其他默认值

                df['Coarsegrained_成员3'] = df['成员3'].apply(assign_coarsegrained)

            # 存储到新的Excel文件
            output_file_path = os.path.join(output_folder, file_name1)
            df.to_excel(output_file_path, index=False)  # 不保存索引

            print(f"处理文件: {file_name}，已保存到 {output_folder}")
        else:
            print(f"文件: {file_name} 中没有 'Finegrained' 这一列.")
    else:
        print(f"文件: {file_name} 不存在.")
def cul_chengyuan12():
    dataset_folder = 'newdata'  # Dataset文件夹
    art_dataframes = []
    science_dataframes = []

    # 遍历所有Art和Science的文件
    for subject in ['Art', 'Science']:
        for i in range(1, 11):
            file_name = f"{subject}{i}.xlsx"
            file_path = os.path.join(dataset_folder, file_name)

            # 确保文件存在
            if os.path.exists(file_path):
                # 读取Excel文件
                df = pd.read_excel(file_path)

                # 根据科目分类存储
                if subject == 'Art':
                    art_dataframes.append(df)
                elif subject == 'Science':
                    science_dataframes.append(df)

                    # 合并DataFrame
    all_art_df = pd.concat(art_dataframes, ignore_index=True) if art_dataframes else pd.DataFrame()
    all_science_df = pd.concat(science_dataframes, ignore_index=True) if science_dataframes else pd.DataFrame()

    # 合并所有数据
    all_data_df = pd.concat([all_art_df, all_science_df], ignore_index=True)

    # 导出Excel文件
    all_art_df.to_excel('All_Art.xlsx', index=False)
    all_science_df.to_excel('All_Science.xlsx', index=False)
    all_data_df.to_excel('All_Data.xlsx', index=False)

    print("Excel文件已成功导出。")

if __name__ == "__main__":
    # cul_chengyuan12()
    humanfinegrainedTocoarsegrained()
    # deleteTen()
    # ten_if()
    # finegrainedTocoarsegrained()
    # import re
    #
    # coding_result = "{ 'coding': 0 }"  # 示例输出
    #
    # # 使用正则表达式提取 coding 的值
    # match = re.search(r'\{\s*\'coding\'\s*:\s*(\d)\s*\}', coding_result)
    #
    # if match:
    #     coding_value = match.group(1)  # 提取匹配的数字部分
    #     print(f"提取的编码值: {coding_value}")  # 输出提取的编码值
    # else:
    #     print("未找到编码值。")
    # 定义文件夹路径
    # deleteTen()


