# encoding=utf-8
import json
import random
import pandas as pd

def shffle_data(input_file):
    # 读取JSON文件，指定编码为UTF-8
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 检查数据是否为列表
    if isinstance(data, list):
        # 随机打乱列表元素顺序
        random.shuffle(data)

        # 将打乱顺序后的列表写入新的JSON文件，保留原始格式
        with open(input_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
    else:
        print("JSON数据不是一个数组。")

def read_excel(input_file):
    # 读取Excel文件
    df = pd.read_excel(input_file)  # 替换为你的Excel文件名

    # 将数据转换为两个列表
    text_list = df['text'].tolist()
    sql_list = df['sql'].tolist()

    # 打印列表以验证
    return text_list, sql_list

def generate_json(db_id, input_list, output_list, history, instruction):
    # 创建一个包含固定值的字典
    data = {
        'db_id': db_id,
        'input': input_list,
        'output': output_list,
        'history': history,
        'instruction': instruction
    }
    return data

def add_to_json_file(filename, new_data):
    # 读取现有的JSON文件
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            existing_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []  # 如果文件不存在或JSON解析失败，则创建一个空列表

    # 将新的数据添加到现有数据中
    existing_data.append(new_data)

    # 将更新后的数据写回JSON文件
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, indent=4, ensure_ascii=False)

excel_file = "sql_excel.xlsx"
db_id_value = "zhly_nhjk"
history_value = []
instruction_value = ""

text_values, sql_values = read_excel(excel_file)


for text, sql in zip(text_values, sql_values):

    modified_text = "###输入:\n" + text + "\n\n###回答:"
    # 生成新的JSON数据
    new_json_data = generate_json(db_id_value, modified_text, sql, history_value, instruction_value)

    # 将新的JSON数据添加到现有的JSON文件中
    json_filename = "text.json"
    add_to_json_file(json_filename, new_json_data)

shffle_data("text.json")
