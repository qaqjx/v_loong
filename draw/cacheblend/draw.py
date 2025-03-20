import pandas as pd

# 第一组数据 (F1 和 time)
data1 = {
    'dataset': "Musique",
    'model': ['Mistral-7B-Instruct-v0.2', 'cacheblend+Mistral-7B-Instruct-v0.2'],
    'F1': [0.2925, 0.2743],
    'time': [0.973, 0.306]
}

df1 = pd.DataFrame(data1)

# 第二组数据 (F1 和 time)
data2 = {
    'dataset': "Wikimqa",
    'model': ['Mistral-7B-Instruct-v0.2', 'cacheblend+Mistral-7B-Instruct-v0.2'],
    'F1': [0.2536, 0.2501],
    'time': [0.9898, 0.3121]
}

df2 = pd.DataFrame(data2)

# 第三组数据 (R1 和 Time)
data3 = {    
    'dataset': "samsum",
    'model': ['Mistral-7B-Instruct-v0.2', 'cacheblend+Mistral-7B-Instruct-v0.2'],
    'R1': [0.3840, 0.4069],
    'time': [1.7805, 0.6585]
}

df3 = pd.DataFrame(data3)

# 合并所有数据
df1['metric'] = 'F1'
df1.rename(columns={'F1': 'value'}, inplace=True)

df2['metric'] = 'F1'
df2.rename(columns={'F1': 'value'}, inplace=True)

df3['metric'] = 'R1'
df3.rename(columns={'R1': 'value'}, inplace=True)

# 将所有数据合并到一个 DataFrame
combined_df = pd.concat([df1, df2, df3], ignore_index=True)


import matplotlib.pyplot as plt
import seaborn as sns

# 设置图形大小
plt.figure(figsize=(12, 8))

# 使用 seaborn 绘制分组柱状图
sns.barplot(
    data=combined_df,
    x='dataset',
    y='value',
    hue='model',
    palette='Set2',
    errorbar=None
)

# 添加标题和标签
plt.title('Generation Quality')
plt.xlabel('Dataset')
plt.ylabel('Value')
plt.legend(title='Model')

# 显示图形
# plt.tight_layout()
plt.show()
plt.savefig('Generation_Quality.png')


# 使用 seaborn 绘制分组柱状图
sns.barplot(
    data=combined_df,
    x='dataset',
    y='time',
    hue='model',
    palette='Set2',
    errorbar=None
)

# 添加标题和标签
plt.title('TTft')
plt.xlabel('Dataset')
plt.ylabel('time')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), title='Model')

# 显示图形
# plt.tight_layout()
plt.show()
plt.savefig('time.png')

