# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
#
# # 读取 Excel 文件
# data = pd.read_excel('EvaluateDataset/human/All_Data_1.xlsx')
# print(data.head())
#
# # 清除包含空白的行
# # data.dropna(subset=['Coarsegrained_成员2', 'Coarsegrained_成员3'], inplace=True)
# # 将包含空白的单元格替换为0
# data['成员2'].replace('', 0, inplace=True)
# data['成员3'].replace('', 0, inplace=True)
# # 删除 y_pred 不是 0, 1, 2 的行
# # data = data[data['Coarsegrained_成员2'].isin([0, 1, 2])]
# # data = data[data['Coarsegrained_成员3'].isin([0, 1, 2])]
# data = data[data['成员2'].isin([0, 1, 2,3,4,5,6,7,8,9,10,11])]
# data = data[data['成员3'].isin([0, 1, 2,3,4,5,6,7,8,9,10,11])]
# # 获取真实标签和预测标签
# y_true = data['成员3'].astype(int)
# y_pred = data['成员2'].astype(int)
#
# # 检查 y_true 和 y_pred 的唯一值
# unique_y_true = np.unique(y_true)
# unique_y_pred = np.unique(y_pred)
#
# print("Unique values in y_true:", unique_y_true)
# print("Unique values in y_pred:", unique_y_pred)
#
# # 检查长度是否一致
# if len(y_true) != len(y_pred):
#     raise ValueError("y_true and y_pred must have the same length.")
#
# # 检查是否存在未定义的标签
# undefined_labels = set(unique_y_true) - set(unique_y_pred)
# if undefined_labels:
#     raise ValueError(f"Found undefined labels in y_true that are not in y_pred: {undefined_labels}")
#
# # 计算混淆矩阵
# cm = confusion_matrix(y_true, y_pred)
# print("Confusion Matrix:")
# print(cm)
#
# # 计算分类报告，包含准确率、精确率、F1-score 和召回率
# report = classification_report(y_true, y_pred, output_dict=True)
# accuracy = report['accuracy']
# precision = report['macro avg']['precision']
# recall = report['macro avg']['recall']
# f1_score = report['macro avg']['f1-score']
#
# print(f'Accuracy: {accuracy:.4f}')
# print(f'Precision: {precision:.4f}')
# print(f'Recall: {recall:.4f}')
# print(f'F1 Score: {f1_score:.4f}')
#
# # 绘制混淆矩阵
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix")
#
# # 保存混淆矩阵图像
# plt.savefig("confusion_matrix.png")
# print("Confusion matrix saved as 'confusion_matrix.png'")

##############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# 读取 Excel 文件
data = pd.read_excel('EvaluateDataset/human/All_Science_1.xlsx')
print(data.head())

# 将包含空白的单元格替换为0
data['成员2'] = data['成员2'].replace('', 0)
data['成员3'] = data['成员3'].replace('', 0)

# 用 0 填充 NaN 值
data['成员2'] = data['成员2'].fillna(0)
data['成员3'] = data['成员3'].fillna(0)

# 将数据类型转换为整数
data['成员2'] = data['成员2'].astype(int)
data['成员3'] = data['成员3'].astype(int)

# 删除 y_pred 不是 0-11 的行
data = data[data['成员2'].isin(range(12))]  # [0, 1, 2, ..., 11]
data = data[data['成员3'].isin(range(12))]

# 获取真实标签和预测标签
y_true = data['成员3'].astype(int)
y_pred = data['成员2'].astype(int)

# 检查 y_true 和 y_pred 的唯一值
unique_y_true = np.unique(y_true)
unique_y_pred = np.unique(y_pred)

print("Unique values in y_true:", unique_y_true)
print("Unique values in y_pred:", unique_y_pred)

# 检查是否存在未定义的标签
undefined_labels = set(unique_y_true) - set(unique_y_pred)
if undefined_labels:
    print(f"Warning: Found undefined labels in y_true that are not in y_pred: {undefined_labels}")
    # 筛选 y_true 和 y_pred，保留只在 y_pred 中的标签
    y_true = y_true[y_true.isin(unique_y_pred)]
    y_pred = y_pred[y_true.index]

# 检查长度是否一致
if len(y_true) != len(y_pred):
    raise ValueError("y_true and y_pred must have the same length.")

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# 计算分类报告
report = classification_report(y_true, y_pred, output_dict=True)
accuracy = report['accuracy']
precision = report['macro avg']['precision']
recall = report['macro avg']['recall']
f1_score = report['macro avg']['f1-score']

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1_score:.4f}')

# 绘制混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")

# 保存混淆矩阵图像
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as 'confusion_matrix.png'")