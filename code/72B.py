from openai import OpenAI

client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1/',
    api_key='44709aec-78aa-492d-869b-b7423dfa8e4e', # ModelScope Token
)

response = client.chat.completions.create(
	model='Qwen/Qwen2.5-VL-72B-Instruct', # ModelScope Model-Id
    messages=[{
        'role':
            'user',
        'content': [{
            'type': 'text',
            'text': '描述这幅图',
        }, {
            'type': 'image_url',
            'image_url': {
                'url':
                    'https://c-ssl.duitang.com/uploads/blog/202009/06/20200906181700_XTQtN.jpeg',
            },
        }],
    }],
    stream=False
)

print(response.choices[0].message.content)