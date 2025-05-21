import re
import pandas as pd
import requests
import json
import os

# 定义文件夹路径和文件名
dataset_folder = 'Dataset'
file_name = 'Art1.xlsx'
file_path = os.path.join(dataset_folder, file_name)

# 设置代理
proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
}


def Coding_zero_shot_prompt4Coarsegrained(text):
    """
    通用编码函数，可以编码

    参数:
    text - 要编码的文本

    返回:
    编码结果
    """
    url = "https://api.siliconflow.cn/v1/chat/completions"
    # prompt = "请将下面课堂文本进行编码，如果编码文本是教师发出，则输出：{coding = 0},如果编码文本发出者是学生，则输出：{coding = 1}，如果编码文本中明确使用了一些技术硬件产品，则输出{coding = 2}，一定要以JSON格式输出。" + f"注意！需要编码的文本是：“{text}”"
    # prompt = (
    #     "\n注意：需要编码的文本是：\"{text}\"。"
    #     "注意：需要精确判断这个文本的发出主体，如果该句既有学生发出又有教师发出，则算作学生发出的。反之，比如提问学生，算教师发出的"
    #     "注意：如果存在技术或者看某个产品来辅助学习，则属于文本中包含具体的技术或硬件产品"
    #     "不是输出Python代码。仅需要判断我现在给你的。请将下面的课堂文本进行编码。以JSON格式回复，格式如下："
    #     "{ 'coding': <数字> }，其中："
    #     "- 如果文本是以教师口吻发出的，请返回 { 'coding': 0 }；"
    #     "- 如果文本是以学生口吻发出的，请返回 { 'coding': 1 }；"
    #     "- 如果文本中包含具体的技术或硬件产品，请返回 { 'coding': 2 }。"
    #     "\n注意：需要编码的文本是：\"{text}\"。"
    # )
    prompt1 = (
        "\n注意：需要编码的文本是：\"{text}\"。"
        "注意：需要精确判断这个文本的发出主体。"
        "如果文本中既有学生的发出又有教师的发出，则整体算作学生发出的。"
        "如果文本是教师在提问或引导学生讨论，则应视为教师发出的。"
        "注意：如果存在具体的技术或硬件产品的引用，"
        "则该文本会被判断为包含技术内容。"
        "不是输出Python代码。仅需要判断我现在给你的。请将下面的课堂文本进行编码。以JSON格式回复，格式如下："
        "{ 'coding': <数字> }，其中："
        "- 如果文本是以教师口吻发出的，请返回 { 'coding': 0 }；"
        "- 如果文本是以学生口吻发出的，请返回 { 'coding': 1 }；"
        "- 如果文本中包含具体的技术或硬件产品，请返回 { 'coding': 2 }。"
        "\n注意：需要编码的文本是：\"{text}\"。"
    )
    prompt2 = (
        "\n注意：需要编码的文本是：\"{text}\"。"
        "判断文本的发出主体："
        "1. 如果句子是教师在陈述或者提问学生的，则视为教师口吻。"
        "2. 如果句子是学生直接表达自己的想法或提问，则视为学生口吻。"
        "3. 如果文本中提到具体的技术如ipad这种电子产品，则标记为技术内容。"
        "请不要输出任何Python代码，仅需进行编码。"
        "以JSON格式回复，格式如下："
        "{ 'coding': <数字> }，其中："
        "- 教师口吻返回 { 'coding': 0 }；"
        "- 学生口吻返回 { 'coding': 1 }；"
        "- 包含技术或硬件产品返回 { 'coding': 2 }。"
        "\n需要编码的文本是：\"{text}\"。"
    )
    prompt = (
        "\n注意：需要编码的文本是：\"{text}\"。"
        "判断文本的发出主体："
        "1. 如果句子是教师在陈述、提问或引导学生的，则视为教师口吻。"
        "2. 如果句子是学生直接表达自己的想法或提问，则视为学生口吻。"
        "3. 只有当文本中明确提到特定的技术或硬件产品（如具体的工具、设备名称等）时，才标记为技术内容。"
        "请不要输出任何Python代码，仅需进行编码。"
        "以JSON格式回复，格式如下："
        "{ 'coding': <数字> }，其中："
        "- 教师口吻返回 { 'coding': 0 }；"
        "- 学生口吻返回 { 'coding': 1 }；"
        "- 包含具体技术或硬件产品返回 { 'coding': 2 }。"
        "\n需要编码的文本是：\"{text}\"。"
    )
    payload = {
        "model": "THUDM/glm-4-9b-chat",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "max_tokens": 512,
        "temperature": 0,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"}
    }

    headers = {
        "Authorization": "Bearer sk-psxspuuwacnnlezwuktzbvevyerrhrncqzrepaukfojrvthy",
        "Content-Type": "application/json"
    }

    response = requests.post(url, proxies=proxies, json=payload, headers=headers)
    response_json = response.json()

    # 提取编码结果
    content = response_json["choices"][0]["message"]["content"].strip()

    return content


if __name__ == "__main__":
    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 检查是否有“课堂对话”列
    if '课堂对话' in df.columns:
        # 初始化“glm4”列
        df['glm4'] = None

        # 遍历“课堂对话”列中的每一行
        for index, row in df.iterrows():
            text = row['课堂对话']
            print(text)
            if pd.notna(text):  # 确保文本不为空
                coding_result = Coding_zero_shot_prompt4Coarsegrained(text)
                print(coding_result)
                # 使用正则表达式提取 coding 的值
                match = re.search(r'\{\s*\'coding\'\s*:\s*(\d)\s*\}', coding_result)
                print(match)
                if match:
                    # 提取到的值
                    coding_value = match.group(1)
                    print(coding_value)
                    df.at[index, 'glm4'] = coding_value  # 将提取的值放入“glm4”列

        # 重新保存到Dataset文件夹
        df.to_excel(file_path, index=False)  # 不保存索引
        print(f"处理完成，结果已保存到 {file_path}")
    else:
        print("文件中没有 '课堂对话' 这一列。")