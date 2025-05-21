import pandas as pd
from tqdm import tqdm
import os
import time
from openai import OpenAI
from datetime import datetime
import re


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
    # url = "https://api.siliconflow.cn/v1/chat/completions"
    # headers = {
    #     "Authorization": "Bearer sk-psxspuuwacnnlezwuktzbvevyerrhrncqzrepaukfojrvthy",
    #     "Content-Type": "application/json"
    # }
    client = OpenAI(
        base_url='https://api-inference.modelscope.cn/v1/',
        api_key='2ff7fb15-03ab-49bc-a321-8e7582d0359d',  # ModelScope Token
    )

    # 创建提示
    # if system_prompt is None:
    #     system_prompt = "你是一位资深课堂教研专家，请根据提供的编码框架对课堂内容进行编码。仅需要输出编码数字，不要包括其他内容。"
    #
    # messages = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": prompt}
    # ]
    #
    # payload = {
    #     "model": "Qwen/Qwen2.5-7B-Instruct",
    #     "messages": messages,
    #     "max_tokens": max_new_tokens,
    #     "temperature": temperature,
    #     "top_p": top_p,
    #     "n": 1
    # }
    response = client.chat.completions.create(
        model='Qwen/Qwen2.5-72B-Instruct',  # ModelScope Model-Id
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role':
                    'user',
                'content': [{
                    'type': 'text',
                    'text': prompt,
                }],
            }],
        temperature=temperature,  # 这里添加了temperature参数，值可以根据需要调整
        stream=False
    )

    # 发送请求
    # response = requests.post(url, json=payload, headers=headers)
    # response_json = response.json()
    # 提取生成的文本
    # response_text = response_json["choices"][0]["message"]["content"].strip()
    # print("response_text", response_text)
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def batch_generate_responses(prompts, system_prompt=None, batch_size=1, max_new_tokens=64, temperature=0.1, top_p=0.9):
    """
    批量生成模型响应
    """
    print("batch_generate_responses")
    responses = []
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
            time.sleep(0.2)
        responses.extend(batch_responses)
    return responses


def get_prompt_framework(coarse_code, prompt_version):
    """
    根据粗粒度编码和提示版本返回对应的提示框架
    参数:
    coarse_code: 粗粒度编码 (0, 1, 或 2)
    prompt_version: 提示版本 (1, 2, 或 3)
    返回:
    prompt_framework: 对应的提示框架文本
    """
    if coarse_code == 0:  # 教师语言
        if prompt_version == 1:
            return """
            教师接受情感1
            教师表扬或者鼓励2
            教师采纳学生观点3
            教师提问4
            教师讲授5
            教师指令6
            教师批评或维护权威7
            """
        elif prompt_version == 2:
            return """
            教师接受情感1：教师对学生表达的情感反应和认可，以增强学生的情感归属感和课堂氛围。
            教师表扬或者鼓励2：教师对学生的积极表现给予赞扬或激励，以促进学生的自信心和参与感。
            教师采纳学生观点3：教师认真听取并考虑学生的意见或建议，显示出对学生思考的重视。
            教师提问4：教师通过提问来引导学生思考和讨论，促进课堂互动和深入学习。
            教师讲授5：教师直接传授知识或技能，通过讲解来传递信息。
            教师指令6：教师发出具体的指示或任务，以指导学生的行为或学习过程。
            教师批评或维护权威7：教师对学生的错误或不当行为进行批评，或者通过展示权威性来维持课堂纪律。
            """
        elif prompt_version == 3:
            return """
            教师接受情感1：教师对学生表达的情感反应和认可，以增强学生的情感归属感和课堂氛围。
            例子：嗯，是的。非常的雄伟壮观。
            教师表扬或者鼓励2：教师对学生的积极表现给予赞扬或激励，以促进学生的自信心和参与感。
            例子：有的同学学习习惯非常好，他会仔细的先看清楚学习任务，然后再开始。
            教师采纳学生观点3：教师认真听取并考虑学生的意见或建议，显示出对学生思考的重视。
            例子：请坐，语气上的一个斟酌啊可以循循善诱，也有委婉含蓄，当然也有略微的一个示意。啊，很好。
            教师提问4：教师通过提问来引导学生思考和讨论，促进课堂互动和深入学习。
            例子：同学们认识了这几座桥梁，有什么想说的吗？请你来说。
            教师讲授5：教师直接传授知识或技能，通过讲解来传递信息。
            例子：同学们通过这节课的讲解，介绍赵州桥，不仅让我们感受到了赵州桥的雄伟美观，设计巧妙，更让我们感受到了劳动人民的智慧和才干。
            教师指令6：教师发出具体的指示或任务，以指导学生的行为或学习过程。
            例子：老师来考考你们，请你和同桌来做一做动作。第一个，相互抵着，回首遥望。
            教师批评或维护权威7：教师对学生的错误或不当行为进行批评，或者通过展示权威性来维持课堂纪律。
            例子：同学们请安静，我们先静静的看一看我们写的这些方法。
            """

    elif coarse_code == 1:  # 学生语言
        if prompt_version == 1:
            return """
            学生被动应答8
            学生主动说话9
            """
        elif prompt_version == 2:
            return """
            学生被动应答8：学生在教师提问后以被动的方式作出反应，通常表现为简短的回答或在教师的引导下进行回应。
            学生主动说话9：学生积极参与课堂讨论，主动表达自己的观点、问题或想法，展现出对学习内容的兴趣和参与感。
            """
        elif prompt_version == 3:
            return """
            学生被动应答8：学生在教师提问后以被动的方式作出反应，通常表现为简短的回答或在教师的引导下进行回应。
            例子：赵州桥
            学生主动说话9：学生积极参与课堂讨论，主动表达自己的观点、问题或想法，展现出对学习内容的兴趣和参与感。
            例子：请你说。我觉得我们这个桥是别的国家都是比不过的，非常的自豪，因为这是我们国家建造的，对，这是中国制造。
            """

    elif coarse_code == 2:  # 技术
        if prompt_version == 1:
            return """
            教师操纵技术10
            学生操纵技术11
            """
        elif prompt_version == 2:
            return """
            教师操纵技术10：教师运用各种技术设备和工具（如演示文稿、视频、互动软件等）来支持教学过程，提高课堂的互动性和学生的学习效果。
            学生操纵技术11：学生使用技术设备和工具（如电脑、平板、软件等）进行学习活动，表现出独立完成任务或参与互动的能力。
            """
        elif prompt_version == 3:
            return """
            教师操纵技术10：教师运用各种技术设备和工具（如演示文稿、视频、互动软件等）来支持教学过程，提高课堂的互动性和学生的学习效果。
            例子：正式上课前，贺老师想先带大家去中国桥梁博物馆看一看，欢迎来到武汉长江大桥，这是新中国成立以后第一座公铁两用大桥，从此长江天堑变通途。
            学生操纵技术11：学生使用技术设备和工具（如电脑、平板、软件等）进行学习活动，表现出独立完成任务或参与互动的能力。
            例子：请同学们拿出iPad，请你在讨论区填上你想输入在横线里面的词，你还可以看看讨论区里你赞同的其他的观点，可以为他点赞。
            """

    return ""

def create_ifias_prompt(classroom_text, prompt_framework, coarse_code):
    """
    创建IFIAS编码提示（无上下文版本）
    """
    print("create_ifias_prompt")
    print("classroom_text", classroom_text)

    code_range = {
        0: "1-7",
        1: "8-9",
        2: "10-11"
    }.get(coarse_code, "")

    prompt = f"""你是一位资深课堂教研专家，请根据以下编码框架对课堂内容进行精确编码。仅需要输出编码数字，不要包括其他内容。

        编码框架说明：
        {prompt_framework}

        请直接对以下文本进行编码（无需考虑上下文）：
        1. 阅读文本内容
        2. 根据编码框架匹配最合适的编码
        3. 仅输出{code_range}范围内的数字

        待编码文本：
        {classroom_text}
    """
    return prompt


def extract_code_from_response(response):
    """
    从模型响应中提取编码数字
    """
    print("extract_code_from_response")
    response = response.strip()
    print("response", response)

    if response.isdigit():
        return int(response)

    digits = re.findall(r'\b\d+\b', response)
    if digits:
        return int(digits[0])
    return None


def process_excel_file(excel_file, text_column, text_column1, batch_size=1, prompt_version=1, output_file=None):
    """
    处理Excel文件（无上下文版本）
    """
    print("process_excel_file")
    df = pd.read_excel(excel_file)

    if text_column not in df.columns or text_column1 not in df.columns:
        raise ValueError(f"指定的列 '{text_column}' 或 '{text_column1}' 在Excel文件中不存在")

    prompts = []
    coarse_codes = []

    for i in range(len(df[text_column])):
        text = df[text_column].iloc[i]
        coarse_code = df[text_column1].iloc[i]

        if pd.notna(text) and pd.notna(coarse_code):
            prompt_framework = get_prompt_framework(coarse_code, prompt_version)
            prompt = create_ifias_prompt(text, prompt_framework, coarse_code)
            prompts.append(prompt)
            coarse_codes.append(coarse_code)
        else:
            prompts.append(None)
            coarse_codes.append(None)

    valid_indices = [i for i, p in enumerate(prompts) if p is not None]
    valid_prompts = [p for p in prompts if p is not None]

    if valid_prompts:
        responses = batch_generate_responses(
            valid_prompts,
            "你是一位资深课堂教研专家，请根据提供的编码框架对课堂内容进行精确编码。仅需要输出编码数字，不要包括其他内容。",
            batch_size
        )
    else:
        responses = []

    codes = [extract_code_from_response(response) for response in responses]
    print("codes", codes)

    all_codes = [None] * len(df)
    for idx, code in zip(valid_indices, codes):
        all_codes[idx] = code

    result_df = df.copy()
    result_df['Finegrained_Code'] = all_codes

    if output_file:
        result_df.to_excel(output_file, index=False)
        print(f"结果已保存到 {output_file}")

    return result_df


if __name__ == "__main__":
    input_folder = 'Dataset/'
    for subject in ['Art']:  # 只处理Science类别'Art','Science'
        print(subject)
        for i in range(10, 11):  # 处理1-10号文件
            print(i)
            for prompt_version in range(3, 4):  # 测试3种提示版本
                print("Prompt版本:", prompt_version)
                file_name = f"{subject}{i}.xlsx"
                excel_file = os.path.join(input_folder, file_name)

                output_file = f"output1/qwen72b/Result{subject}{i}_prompt{prompt_version}_qwen72b.xlsx"
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                result_df = process_excel_file(
                    excel_file,
                    "课堂对话",
                    "Coarsegrained",
                    batch_size=2,
                    prompt_version=prompt_version,
                    output_file=output_file
                )

                code_counts = result_df['Finegrained_Code'].value_counts().sort_index()
                print("\n编码统计:")
                for code, count in code_counts.items():
                    if pd.notna(code):
                        print(f"编码 {code}: {count}个")