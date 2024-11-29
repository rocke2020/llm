import pandas as pd
import json
import re
from openai import OpenAI


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
    df.columns = [col.replace("T1.", "").replace("T2.", "") for col in df.columns]

    # Reverse the order of columns
    df = df[df.columns[::-1]]

    # Generate the Markdown table
    markdown_table = df.to_markdown(index=False)

    return markdown_table

def rewrite(query):
    # 调用deepseek v2
    client = OpenAI(
        api_key="sk-eba54da4994d4c9bab70408237163fab",
        base_url="https://api.deepseek.com",
    )

    prompt = "你将扮演一个问题改写小助手，你仅需根据我给你的问题给出相应的同义改写即可，注意一定要与原句子语义完全一致，且意图清晰"

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
        stream=False,
    )
    return response.choices[0].message.content


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
