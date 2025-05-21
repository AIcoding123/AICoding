import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import time
from datetime import datetime

def setup_model(model_path, use_yarn=False):
    """
    加载DeepSeek-R1-Distill-Qwen-7B模型和分词器

    参数:
    model_path: 模型名称或路径
    use_yarn: 是否启用YaRN以处理长文本

    返回:
    model: 加载的模型
    tokenizer: 加载的分词器
    """
    # 设置模型加载参数
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        "device_map": "auto",
        "trust_remote_code": True
    }

    # 如果需要处理超过32K的长文本，可以启用YaRN
    if use_yarn:
        model_kwargs["rope_scaling"] = {
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768
        }

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    return model, tokenizer

def generate_response(model, tokenizer, prompt, system_prompt=None, max_new_tokens=64, temperature=0.1, top_p=0.9):
    """
    生成模型响应

    参数:
    model: 加载的模型
    tokenizer: 加载的分词器
    prompt: 用户输入的提示
    system_prompt: 系统提示，用于设置模型角色
    max_new_tokens: 最大生成的token数量
    temperature: 温度参数，控制生成的随机性
    top_p: top-p采样参数

    返回:
    response: 模型生成的回复
    """
    # 准备对话消息
    if system_prompt is None:
        print("111")
        system_prompt = "You are DeepSeek. You are a helpful assistant."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 将输入转换为模型可接受的格式
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成响应
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0
        )

    # 只获取新生成的token
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 解码生成的token为文本
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def batch_generate_responses(model, tokenizer, prompts, system_prompt=None, batch_size=1, max_new_tokens=64,
                             temperature=0.1, top_p=0.9):
    """
    批量生成模型响应

    参数:
    model: 加载的模型
    tokenizer: 加载的分词器
    prompts: 提示列表
    system_prompt: 系统提示，用于设置模型角色
    batch_size: 批处理大小
    max_new_tokens: 最大生成的token数量
    temperature: 温度参数，控制生成的随机性
    top_p: top-p采样参数

    返回:
    responses: 响应列表
    """
    responses = []

    # 分批处理
    for i in tqdm(range(0, len(prompts), batch_size), desc="批量处理中"):
        batch_prompts = prompts[i:i + batch_size]
        batch_responses = []

        for prompt in batch_prompts:
            response = generate_response(
                model,
                tokenizer,
                prompt,
                system_prompt,
                max_new_tokens,
                temperature,
                top_p
            )
            batch_responses.append(response)
            print(response)
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
    # prompt = f"""你是一位资深课堂教研专家，你的任务是根据课堂内容上下文，对上下文中指定的一段课堂内容文本进行IFIAS编码。
    #
    #     IFIAS编码是一种课堂交互分析系统，它包含以下分类：
    #     {ifias_framework}
    #
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
    #     <context><text2code>{classroom_text}</text2code></context>
    #     """
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
    response = response.strip()

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


def process_excel_file(excel_file, text_column, model, tokenizer, batch_size=1, j=1,output_file=None):
    """
    处理Excel文件中的课堂对话并进行IFIAS编码

    参数:
    excel_file: Excel文件路径
    text_column: 包含课堂对话的列名
    model: 加载的模型
    tokenizer: 加载的分词器
    batch_size: 批处理大小
    output_file: 输出文件路径

    返回:
    result_df: 包含编码结果的DataFrame
    """
    # 读取Excel文件
    df = pd.read_excel(excel_file)

    # 确保指定的列存在
    if text_column not in df.columns:
        raise ValueError(f"列 '{text_column}' 在Excel文件中不存在")
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
            - 大桥洞与小桥洞分别有什么作用？ { 'coding': 0 }  # 教师提问 
            - 同学们通过这节课的讲解 { 'coding': 0 }  # 教师总结  

            学生语言:  
            学生语言行为反映了学生在课堂中的参与程度，包括被动应答和主动发言。积极的学生语言表现有助于提升课堂互动和学习效果。  
            示例：  
            - 我感觉我们中国建造的桥都非常的雄伟。 { 'coding': 1 }  # 学生主动发言  
            - 请你来说。这个桥叫港珠澳大桥， { 'coding': 1 }  # 教师提问，学生被动表达  


            技术:  
            技术在教学中的应用包括教师和学生使用各种技术工具，以增强教学效果和学习体验。教师利用技术工具进行信息传递和课堂管理，学生利用技术进行自主学习和知识探索。  
            - 老师想先带大家去中国桥梁博物馆 { 'coding': 2 }  # 使用技术  
            - 请同学们拿出iPad { 'coding': 2 }  # 使用技术  
            """
    # 创建提示列表
    prompts = []
    for text in df[text_column]:
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
        system_prompt = "你是一位资深课堂教研专家，请根据IFIAS编码框架对课堂内容进行准确编码。"
        responses = batch_generate_responses(
            model,
            tokenizer,
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
    # 指定模型的本地路径
    local_model_path = "../../models/Qwen2.5-7B-Instruct"  # 请替换为实际的模型存放路径
    # local_model_path = "../../models/DeepSeek-R1-Distill-Qwen-7B"  # 修改为DeepSeek模型路径
    input_folder = '../Dataset'
    for k in [7, 8]:
        print(k)
        if k == 1:
            local_model_path = "../../models/Baichuan2-7B-Chat"  # no
            print(local_model_path)
        elif k == 2:
            local_model_path = "../../models/DeepSeek-R1-Distill-Qwen-7B"  # can low
        elif k == 3:
            local_model_path = "../../models/gemma-3-4b-it"  # no can
        elif k == 4:
            local_model_path = "../../models/glm-4-9b-chat"  # can low
        elif k == 5:
            local_model_path = "../../models/Meta-Llama-3.1-8B-Instruct"  # no can
        elif k == 6:
            local_model_path = "../../models/Mistral-7B-Instruct-v0.3"  # no can
        elif k == 7:
            local_model_path = "../../models/Qwen2.5-7B-Instruct"  # can quick
        elif k == 8:
            local_model_path = "../../models/Yi-1.5-9B-Chat"  # can low not good
        for subject in ['Art','Science']:
            print(subject)
            for i in range(1,11):
                print(i)
                for j in range(1,4):
                    print(j)
                    file_name = f"{subject}{i}.xlsx"
                    # Excel文件路径
                    # excel_file = "../data/Chinese1.xlsx"  # 请替换为实际的Excel文件路径


                    excel_file = os.path.join(input_folder,file_name)
                    # 输出文件路径
                    # 获取当前时间并格式化为字符串
                    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                    # 创建包含时间戳的文件路径
                    output_file = f"../output/suc/Result{subject}{i}prompt{j}model{k}_{current_time}.xlsx"

                    # 包含课堂对话的列名
                    text_column = "课堂对话"  # 请替换为实际的列名

                    # 批处理大小
                    batch_size = 2  # 根据GPU内存调整

                    # 加载模型和分词器
                    print("正在加载模型...")
                    model, tokenizer = setup_model(local_model_path)

                    # 处理Excel文件
                    print(f"正在处理Excel文件: {excel_file}")
                    result_df = process_excel_file(
                        excel_file,
                        text_column,
                        model,
                        tokenizer,
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
                                2: "技术",
                            }.get(code_int, "教师")
                            print(f"编码 {code_int} ({code_description}): {count}个")