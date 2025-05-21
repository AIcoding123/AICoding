import requests
import json
import time
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
# 设置代理
proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
}

def Coding_zero_shot_prompt4Coarsegrained(text, text_type="basic"):
    """
    通用编码函数，可以编码

    参数:
    text - 要编码的文本
    text_type - 文本类型，可以是"basic"或"enhanced"

    返回:
    翻译后的文本
    """
    url = "https://api.siliconflow.cn/v1/chat/completions"

    # 根据文本类型构建不同的提示
    if text_type == "basic":
        prompt = "请将下面课堂文本进行编码，如果编码文本是教师发出，则输出：{coding = 0},如果编码文本发出者是学生，则输出：{coding = 1}，如果编码文本中明确使用了一些技术硬件产品，则输出{coding = 2}"+ f"{text}"
    else:  # title
        prompt = f"请将下面课堂文本进行编码：\n\n{text}"

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
        "stop": None,
        "temperature": 0,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"},
        "tools": [
            {
                "type": "function",
                "function": {
                    "description": "<string>",
                    "name": "<string>",
                    "parameters": {},
                    "strict": False
                }
            }
        ]
    }
    headers = {
        "Authorization": "Bearer sk-psxspuuwacnnlezwuktzbvevyerrhrncqzrepaukfojrvthy",
        "Content-Type": "application/json"
    }


    response = requests.request("POST", url, proxies=proxies, json=payload, headers=headers)
    response_json = json.loads(response.text)
    content = response_json["choices"][0]["message"]["content"]


    return content

if __name__ == "__main__":
    
    Coding_zero_shot_prompt4Coarsegrained()