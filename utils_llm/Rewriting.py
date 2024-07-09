import pandas as pd
import json

def data_to_markdown_table(data):
    """
    Convert a list of dictionaries into a Markdown-formatted table with columns displayed in reverse order.

    Args:
        data (list[dict]): A list of dictionaries where each dictionary represents a row in the table.
            All dictionaries must have the same keys (which become column names).

    Returns:
        str: The Markdown-formatted table.
    """
    # Ensure data is not empty
    if not data:
        return "No data to display."

    # Convert data to a Pandas DataFrame for easy manipulation and conversion to Markdown
    df = pd.DataFrame(data)

    # Remove unnecessary prefixes from column names if present
    df.columns = [col.replace('T1.', '').replace('T2.', '') for col in df.columns]

    # Reverse the order of columns
    df = df[df.columns[::-1]]

    # Generate the Markdown table
    markdown_table = df.to_markdown(index=False)

    return markdown_table

# # Example usage with your provided data
# data = [{'SUM(T1.kwh)': 222.4, 'T1.stats_time': '2024-04-16', 'T2.floor_number': 'F1'}, {'SUM(T1.kwh)': 273.11, 'T1.stats_time': '2024-04-17', 'T2.floor_number': 'F1'}, {'SUM(T1.kwh)': 314.49, 'T1.stats_time': '2024-04-18', 'T2.floor_number': 'F1'}, {'SUM(T1.kwh)': 288.66, 'T1.stats_time': '2024-04-19', 'T2.floor_number': 'F1'}, {'SUM(T1.kwh)': 226.41, 'T1.stats_time': '2024-04-20', 'T2.floor_number': 'F1'}, {'SUM(T1.kwh)': 228.37, 'T1.stats_time': '2024-04-21', 'T2.floor_number': 'F1'}, {'SUM(T1.kwh)': 167.34, 'T1.stats_time': '2024-04-22', 'T2.floor_number': 'F1'}, {'SUM(T1.kwh)': 160.58, 'T1.stats_time': '2024-04-16', 'T2.floor_number': 'F2'}, {'SUM(T1.kwh)': 169.98, 'T1.stats_time': '2024-04-17', 'T2.floor_number': 'F2'}, {'SUM(T1.kwh)': 177.21, 'T1.stats_time': '2024-04-18', 'T2.floor_number': 'F2'}, {'SUM(T1.kwh)':165.96, 'T1.stats_time': '2024-04-19', 'T2.floor_number': 'F2'}, {'SUM(T1.kwh)': 100.4, 'T1.stats_time': '2024-04-20', 'T2.floor_number': 'F2'}, {'SUM(T1.kwh)': 98.77, 'T1.stats_time': '2024-04-21', 'T2.floor_number': 'F2'}, {'SUM(T1.kwh)': 91.54, 'T1.stats_time': '2024-04-22', 'T2.floor_number': 'F2'}]
# data2 = [{'SUM(T1.kwh)': 2067.6, 'T2.floor_number': 'F1'}, {'SUM(T1.kwh)': 1161.62, 'T2.floor_number': 'F2'}, {'SUM(T1.kwh)': 178.0, 'T2.floor_number': 'F3'}, {'SUM(T1.kwh)': 363.14, 'T2.floor_number': 'F4'}, {'SUM(T1.kwh)': 346.38, 'T2.floor_number': 'F5'}]
# data3 = [{'统计日期': '2024-04-13', '当日能耗（kWh）': 0.0}, {'统计日期': '2024-04-14', '当日能耗（kWh）': 0.0}, {'统计日期': '2024-04-15', '当日能耗（kWh）': 0.0}, {'统计日期': '2024-04-16', '当日能耗（kWh）': 0.0}, {'统计日期': '2024-04-17', '当日能耗（kWh）': 0.0}, {'统计日期': '2024-04-18', '当日能耗（kWh）': 0.0}, {'统计日期': '2024-04-19', '当日能耗（kWh）': 0.0}]
# table_str_1 = data_to_markdown_table(data)
# table_str_2 = data_to_markdown_table(data2)
# table_str_3 = data_to_markdown_table(data3)
# table_str_4 = data_to_markdown_table([{'total': 100.2}])
# print(f'{table_str_1}\n')
# print(f'{table_str_2}\n')
# print(f'{table_str_3}\n')
# print(f'{table_str_4}\n')
import json
import re
from  get_deepseek import get_llm

def duplicate_elements_and_write(input_file_path, output_file_path):
    # 读取JSON文件，指定编码为UTF-8
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 复制每个元素4次
    duplicated_data = [item for item in data for _ in range(5)]

    # 将复制后的数据写入到新的JSON文件中，指定编码为UTF-8
    output_file_path_tem = "tem.json"
    with open(output_file_path_tem, 'w', encoding='utf-8') as file:
        json.dump(duplicated_data, file, indent=4, ensure_ascii=False)

    print(f"Data duplicated and written to {output_file_path}")

    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 提取每个元素的"input"字段中的文本，去除标记
    extracted_texts = []
    for item in data:
        if 'input' in item:
            # 使用正则表达式去除标记
            cleaned_text = re.sub(r'###输入:\s*|###回答:\s*', '', item['input']).strip()
            extracted_texts.append(cleaned_text)

    # 将提取的文本写入到TXT文件中
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for text in extracted_texts:
            file.write(text + '\n')  # 每个文本后面跟一个换行符


def process_file_with_llm(input_file_path, output_file_path, iterations=3):
    # 读取文本文件的每一行
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 对每一行调用LLM函数进行同义改写
    processed_lines = []
    for line in lines:
        # 去除行末的换行符
        line = line.strip()
        # 进行同义改写，保留每次改写的结果
        for _ in range(iterations):
            line = get_llm(line)
            processed_lines.append(line)

    # 将改写后的结果保存到新的文本文件中
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for line in processed_lines:
            file.write(line + '\n')


input_file_path = 'db_hub_cn_0523_filter.json'  # 替换为你的输入JSON文件路径
output_file_path = 'output.txt'  # 替换为你的输出JSON文件路径
duplicate_elements_and_write(input_file_path, output_file_path)

input_txt_path = 'test.txt'  # 替换为你的输入TXT文件路径
output_txt_path = 'results.txt'  # 替换为你的输出TXT文件路径
process_file_with_llm(input_txt_path, output_txt_path)
