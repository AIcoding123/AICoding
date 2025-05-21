import requests
import pandas as pd
from tqdm import tqdm
import os
import time
from datetime import datetime

def generate_response(prompt, system_prompt=None, max_new_tokens=64, temperature=0.1, top_p=0.9):
    """
    生成模型响应，使用API调用
    参数:
    prompt: 用户输入的提示
    system_prompt: 系统提示，用于设置模型角色
    max_new_tokens: 最大生成的token数量
    temperature: 温度参数，控制生成的随机性
    top_p: top-p采样参数
    返回:
    response: 模型生成的回复
    """
    # 设置API请求参数
    print("generate_response")
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-psxspuuwacnnlezwuktzbvevyerrhrncqzrepaukfojrvthy",  # 替换为您的API密钥
        "Content-Type": "application/json"
    }
    # 创建提示
    if system_prompt is None:
        system_prompt = "你是一位资深课堂教研专家，请根据IFIAS编码框架对课堂内容进行编码。仅需要给出粗粒度的编码结果，即教师语言为0，学生语言为1，技术为2"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    # 构建请求负载Qwen/Qwen2.5-7B-Instruct
    payload = {
        "model": "THUDM/chatglm3-6b",#deepseek-ai/DeepSeek-R1-Distill-Qwen-7B#THUDM/glm-4-9b-chat
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": 1
    }
    # 发送请求
    response = requests.post(url, json=payload, headers=headers)
    response_json = response.json()
    # 提取生成的文本
    response_text = response_json["choices"][0]["message"]["content"].strip()
    print("response_text11", response_text)
    return response_text
def batch_generate_responses(prompts, system_prompt=None, batch_size=1, max_new_tokens=64, temperature=0.1, top_p=0.9):
    """
    批量生成模型响应，使用API调用
    参数:
    prompts: 提示列表
    system_prompt: 系统提示，用于设置模型角色
    batch_size: 批处理大小
    max_new_tokens: 最大生成的token数量
    temperature: 温度参数，控制生成的随机性
    top_p: top-p采样参数
    返回:
    responses: 响应列表
    """
    print("batch_generate_responses")
    responses = []
    # 分批处理
    for i in tqdm(range(0, len(prompts), batch_size), desc="批量处理中"):
        batch_prompts = prompts[i:i + batch_size]
        batch_responses = []

        for prompt in batch_prompts:
            print("prompt:", prompt)
            response = generate_response(
                prompt,
                system_prompt,
                max_new_tokens,
                temperature,
                top_p
            )
            batch_responses.append(response)
            # 为了避免API限制，在每次请求之间添加短暂延迟
            time.sleep(0.2)
        responses.extend(batch_responses)
    return responses

def create_ifias_prompt(classroom_text, ifias_framework):
    """
    创建IFIAS编码提示
    参数:
    classroom_text: 课堂对话文本
    ifias_framework: IFIAS编码框架说明
    返回:
    prompt: 格式化的提示
    """
    print("create_ifias_prompt")
    print("classroom_text",classroom_text)
    # prompt = f"""你是一位资深课堂教研专家，请根据IFIAS编码框架对课堂内容进行编码。仅需要给出粗粒度的编码结果，即教师语言为0，学生语言为1，技术为2
    #
    #     IFIAS编码是一种课堂交互分析系统，它包含以下分类：
    #     {ifias_framework}
    #     不是输出Python代码。仅需要判断我现在给你的。请将下面的课堂文本进行编码。以JSON格式回复，格式如下：{ 'coding': <数字> }
    #     你只需要输出编码结果，不要包括其他内容。
    #     上下文会被放在<context></context>中，要进行编码的课堂内容文本会被放在<text2code></text2code>中。
    #
    #     请严格按照以下步骤执行任务：
    #     1. 完整阅读<context></context>标记内的全部课堂内容上下文
    #     2. 定位<text2code></text2code>标记中的待编码文本片段
    #     3. 根据上下文语境与IFIAS编码手册精确匹配文本特征与编码规则，得出编码
    #     4. 按照要求输出编码结果
    #
    #     下面是你需要进行编码的课堂内容文本及其上下文：
    #     <context>{classroom_text_shangwen}<text2code>{classroom_text}</text2code>{classroom_text_xiawen}</context>
    # """
    prompt = f"""你是一位资深课堂教研专家，请根据IFIAS编码框架对课堂内容进行编码。仅需要给出粗粒度的编码结果，即教师语言为0，学生语言为1，技术为2  

            IFIAS编码是一种课堂交互分析系统，它包含以下分类：  
            {ifias_framework}  

            你只需要输出编码结果，不要包括其他内容。  
            要进行编码的课堂内容文本会被放在<text2code></text2code>中。  

            请严格按照以下步骤执行任务：  
            1. 完整阅读<text2code></text2code>标记中的待编码文本片段    
            2. 根据上下文语境与IFIAS编码手册精确匹配文本特征与编码规则，得出编码  
            3. 按照要求输出编码结果。  <code>数字</code>

            下面是你需要进行编码的课堂内容文本及其上下文：  
            <text2code>{classroom_text}</text2code>  
        """
    print(prompt)
    return prompt

def extract_code_from_response(response):
    """
    从模型响应中提取IFIAS编码

    参数:
    response: 模型响应

    返回:
    jiang: 提取的编码数字
    """
    # 清理响应文本，尝试提取编码数字
    print("extract_code_from_response")
    response = response.strip()
    print("response",response)
    # 如果回复只有一个数字，直接返回
    if response.isdigit():
        return int(response)
        # 尝试从文本中提取数字编码
    import re
    digits = re.findall(r'\b\d+\b', response)
    if digits:
        return int(digits[0])
        # 如果没有找到数字，返回None
    return None

def process_excel_file(excel_file, text_column, batch_size=1,j=1, output_file=None):
    """
    处理Excel文件中的课堂对话并进行IFIAS编码
    参数:
    excel_file: Excel文件路径
    text_column: 包含课堂对话的列名
    batch_size: 批处理大小
    output_file: 输出文件路径
    返回:
    result_df: 包含编码结果的DataFrame
    """
    # 读取Excel文件
    print("process_excel_file")
    df = pd.read_excel(excel_file)
    # 确保指定的列存在
    if text_column not in df.columns:
        raise ValueError(f"列 '{text_column}' 在Excel文件中不存在")
        # IFIAS编码框架说明
    if j == 1:
        print(j)
        # IFIAS编码框架说明
        ###########1##########
        ifias_framework = """
            IFIAS 编码框架说明：
            教师语言、
            学生语言、
            技术。
            """
    elif j == 2:
        print(j)
        #############2###############
        ifias_framework = """
        IFIAS 编码框架说明：

        教师语言:
        教师语言行为包括教师对学生情感的接纳、对学生的表扬和鼓励、在课堂中提出问题、指导学生的学习、以及在必要时进行的批评。这些行为旨在促进学生的参与和理解，营造良好的学习氛围。

        学生语言:
        学生语言行为反映了学生在课堂中的参与程度，包括被动应答和主动发言。积极的学生语言表现有助于提升课堂互动和学习效果。

        技术:
        技术在教学中的应用包括教师和学生使用各种技术工具，以增强教学效果和学习体验。教师利用技术工具进行信息传递和课堂管理，学生利用技术进行自主学习和知识探索。

        """
    elif j == 3:
        print(j)
        #############3###############
        ifias_framework = """  
            IFIAS 编码框架说明：  
            教师语言:  
            教师语言行为包括教师对学生情感的接纳、对学生的表扬和鼓励、在课堂中提出问题、指导学生的学习、以及在必要时进行的批评。这些行为旨在促进学生的参与和理解，营造良好的学习氛围。  
            示例：  
            - 大桥洞与小桥洞分别有什么作用？ <code>0</code>  # 教师提问 
            - 同学们通过这节课的讲解 <code>0</code>  # 教师总结  

            学生语言:  
            学生语言行为反映了学生在课堂中的参与程度，包括被动应答和主动发言。积极的学生语言表现有助于提升课堂互动和学习效果。  
            示例：  
            - 我感觉我们中国建造的桥都非常的雄伟。 <code>1</code>  # 学生主动发言  
            - 请你来说。这个桥叫港珠澳大桥， <code>1</code>  # 教师提问，学生被动表达  


            技术:  
            技术在教学中的应用包括教师和学生使用各种技术工具，以增强教学效果和学习体验。教师利用技术工具进行信息传递和课堂管理，学生利用技术进行自主学习和知识探索。  
            - 老师想先带大家去中国桥梁博物馆 <code>2</code>  # 使用技术  
            - 请同学们拿出iPad <code>2</code>  # 使用技术  
            """
    # ifias_framework = """
    #     教师语言:
    #     1 教师接受情感
    #     2 教师表扬或者鼓励
    #     3 教师采纳学生观点
    #     4 教师提问
    #     5 教师讲授
    #     6 教师指令
    #     7 教师批评或维护权威
    #     学生语言:
    #     8 学生被动应答
    #     9 学生主动说话
    #     技术:
    #     10 教师操纵技术
    #     11 学生操纵技术
    # """
    ##################2#################
    # ifias_framework = """
    #     IFIAS 编码框架说明：
    #     教师语言:
    #     1 教师接受情感：教师识别并回应学生的情感状态，创造一个支持和理解的学习环境，以鼓励学生的积极参与。
    #     2 教师表扬或者鼓励：教师通过表扬和鼓励的方式激发学生的自信心，提升其学习动力，使学生愿意参与课堂活动。
    #     3 教师采纳学生观点：教师重视学生提出的观点，包容并鼓励学生表达，有助于培养学生的批判性思维和参与感。
    #     4 教师提问：教师通过提问激发学生的思维，引导学生深入理解课程内容，并促进讨论与互动。
    #     5 教师讲授：教师通过直接的知识传播，传授理论和技能，帮助学生在特定领域建立基础知识。
    #     6 教师指令：教师通过明确的指令指导学生的学习活动，设定课堂规则，确保学习有序进行。
    #     7 教师批评或维护权威：教师在课堂管理中进行必要的批评，维持课堂纪律和学习秩序，确保所有学生集中注意力。
    #
    #     学生语言:
    #     8 学生被动应答：学生在课堂中对教师提问或指令做出被动回应，通常表现为简单的“是”或“否”回答，缺乏主动性。
    #     9 学生主动说话：学生积极参与课堂讨论，主动表达自己的观点和理解，展现出学习热情和思考能力。
    #
    #     技术:
    #     10 教师操纵技术：教师利用各种技术工具（如投影仪、学习管理系统等）来增强教学效果，提高学生的学习体验。
    #     11 学生操纵技术：学生使用技术工具（如笔记本电脑、平板、软件应用等）进行学习，促进自主学习和探索。
    # """
    #############3###############
    # ifias_framework = """
    # IFIAS 编码框架说明：
    #
    # 教师语言:
    # 教师语言行为包括教师对学生情感的接纳、对学生的表扬和鼓励、在课堂中提出问题、指导学生的学习、以及在必要时进行的批评。这些行为旨在促进学生的参与和理解，营造良好的学习氛围。
    #
    # 学生语言:
    # 学生语言行为反映了学生在课堂中的参与程度，包括被动应答和主动发言。积极的学生语言表现有助于提升课堂互动和学习效果。
    #
    # 技术:
    # 技术在教学中的应用包括教师和学生使用各种技术工具，以增强教学效果和学习体验。教师利用技术工具进行信息传递和课堂管理，学生利用技术进行自主学习和知识探索。
    #
    # """
    ###########4##########
    # ifias_framework = """
    #     IFIAS 编码框架说明：
    #     教师语言、
    #     学生语言、
    #     技术。
    #     """
    ##############5################
    # ifias_framework = """
    # IFIAS 编码框架说明：
    # 教师语言:
    # 教师语言行为包括教师对学生情感的接纳、对学生的表扬和鼓励、在课堂中提出问题、指导学生的学习、以及在必要时进行的批评。这些行为旨在促进学生的参与和理解，营造良好的学习氛围。
    # 示例：
    # - 大桥洞与小桥洞分别有什么作用？ { 'coding': 0 }  # 教师提问
    # - 同学们通过这节课的讲解 { 'coding': 0 }  # 教师总结
    #
    # 学生语言:
    # 学生语言行为反映了学生在课堂中的参与程度，包括被动应答和主动发言。积极的学生语言表现有助于提升课堂互动和学习效果。
    # 示例：
    # - 我感觉我们中国建造的桥都非常的雄伟。 { 'coding': 1 }  # 学生主动发言
    # - 请你来说。这个桥叫港珠澳大桥， { 'coding': 1 }  # 教师提问，学生被动表达
    #
    #
    # 技术:
    # 技术在教学中的应用包括教师和学生使用各种技术工具，以增强教学效果和学习体验。教师利用技术工具进行信息传递和课堂管理，学生利用技术进行自主学习和知识探索。
    # - 老师想先带大家去中国桥梁博物馆 { 'coding': 2 }  # 使用技术
    # - 请同学们拿出iPad { 'coding': 2 }  # 使用技术
    # """

    # 创建提示列表
    prompts = []
    for text in df[text_column]:
        print("text",text)
        if pd.notna(text):  # 处理NaN值
            prompt = create_ifias_prompt(text, ifias_framework)
            prompts.append(prompt)
        else:
            prompts.append(None)
            # 过滤掉None值
    valid_indices = [i for i, p in enumerate(prompts) if p is not None]
    valid_prompts = [p for p in prompts if p is not None]
    # 批量生成响应
    if valid_prompts:
        system_prompt = "你是一位资深课堂教研专家，请根据IFIAS编码框架对课堂内容进行编码。仅需要给出粗粒度的编码结果，即教师语言为0，学生语言为1，技术为2。"
        ############batch_generate_responses########
        responses = batch_generate_responses(
            valid_prompts,
            system_prompt,
            batch_size,
            max_new_tokens=64,  # IFIAS编码回答通常很短
            temperature=0.1,  # 低温度使输出更确定性
            top_p=0.9
        )
    else:
        responses = []
        # 提取编码结果
    codes = [extract_code_from_response(response) for response in responses]
    print("codes",codes)
    # 将编码结果放回原始位置
    all_codes = [None] * len(df)
    for idx, code in zip(valid_indices, codes):
        all_codes[idx] = code

        # 创建结果DataFrame
    result_df = df.copy()
    result_df['IFIAS_Code'] = all_codes

    # 保存结果
    if output_file:
        result_df.to_excel(output_file, index=False)
        print(f"结果已保存到 {output_file}")

    return result_df


if __name__ == "__main__":
    # Excel文件路径
    # excel_file = "Dataset/Art1.xlsx"  # 请替换为实际的Excel文件路径
    # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # output_file = f"output/glm4/Art1ResultCoarse_{current_time}.xlsx"
    # # 包含课堂对话的列名
    # text_column = "课堂对话"  # 请替换为实际的列名
    # # 批处理大小
    # batch_size = 2  # 根据GPU内存调整
    # # 处理Excel文件
    # print(f"正在处理Excel文件: {excel_file}")
    # ##########process_excel_file#########
    # result_df = process_excel_file(
    #     excel_file,
    #     text_column,
    #     batch_size,
    #     output_file
    # )
    # # 显示一些统计信息
    # code_counts = result_df['IFIAS_Code'].value_counts().sort_index()
    # print("\nIFIAS编码统计:")
    # for code, count in code_counts.items():
    #     if pd.notna(code):  # 处理NaN值
    #         code_int = int(code)
    #         code_description = {
    #             1: "学生",
    #             2: "技术"
    #         }.get(code_int, "教师")
    #         print(f"编码 {code_int} ({code_description}): {count}个")'Art',
    input_folder = 'Dataset/'
    for subject in ['Science']:
        print(subject)
        for i in range(9, 11):
            print(i)
            for j in range(1, 4):
                print(j)
                file_name = f"{subject}{i}.xlsx"
                # Excel文件路径
                # excel_file = "../data/Chinese1.xlsx"  # 请替换为实际的Excel文件路径

                excel_file = os.path.join(input_folder, file_name)
                # 输出文件路径
                # 获取当前时间并格式化为字符串
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                # 创建包含时间戳的文件路径
                output_file = f"output/suc/Result{subject}{i}prompt{j}deepseek.xlsx"


                # 包含课堂对话的列名
                text_column = "课堂对话"  # 请替换为实际的列名
                # 批处理大小
                batch_size = 2  # 根据GPU内存调整
                # 处理Excel文件
                print(f"正在处理Excel文件: {excel_file}")
                ##########process_excel_file#########
                result_df = process_excel_file(
                    excel_file,
                    text_column,
                    batch_size,
                    j,
                    output_file
                )
                # 显示一些统计信息
                code_counts = result_df['IFIAS_Code'].value_counts().sort_index()
                print("\nIFIAS编码统计:")
                for code, count in code_counts.items():
                    if pd.notna(code):  # 处理NaN值
                        code_int = int(code)
                        code_description = {
                            1: "学生",
                            2: "技术"
                        }.get(code_int, "教师")
                        print(f"编码 {code_int} ({code_description}): {count}个")

                # # 显示一些统计信息
                # code_counts = result_df['IFIAS_Code'].value_counts().sort_index()
                # print("\nIFIAS编码统计:")
                # for code, count in code_counts.items():
                #     if pd.notna(code):  # 处理NaN值
                #         code_int = int(code)
                #         code_description = {
                #             1: "学生",
                #             2: "技术",
                #         }.get(code_int, "教师")
                #         print(f"编码 {code_int} ({code_description}): {count}个")