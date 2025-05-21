import pandas as pd
from sklearn.metrics import cohen_kappa_score

def calculate_kappa(file_path):
    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 填充空白数据为0
    df.fillna(0, inplace=True)

    # 取出“成员1”和“成员2”两列
    # 尝试将数据转为整数类型（或按照实际类别类型调整）
    try:
        member1_scores = df['Coarsegrained_成员2'].astype(int)
        member2_scores = df['Coarsegrained_成员3'].astype(int)
    except Exception as e:
        print("数据转换为整数时出错，请确认‘成员1’和‘成员2’列内容为数字类别或能转换为整数")
        raise e

    # 计算Cohen's Kappa值
    kappa = cohen_kappa_score(member1_scores, member2_scores)

    return kappa

if __name__ == "__main__":
    excel_path = r"E:\fairy\运河杯\大创\开始写\大语言模型\编码\coding data\All_Data_1.xlsx"
    kappa_value = calculate_kappa(excel_path)
    print(f"‘成员1’和‘成员2’之间的Kappa值为: {kappa_value}")