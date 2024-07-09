import json

# 读取原始JSON文件
with open('original.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 确保数据是一个数组
if isinstance(data, list):
    # 创建一个新的数组来存储重新排序后的元素
    new_data = []

    # 遍历原始数组中的每个元素
    for item in data:
        # 创建一个新的字典来存储重新排序后的键
        new_item = {
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item.get("output", ""),
            "system":"",
            "history": item.get("history", [])
        }
        # 将新的字典添加到新的数组中
        new_data.append(new_item)

    # 将新的数组写入新的JSON文件
    with open('new.json', 'w', encoding='utf-8') as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)
else:
    print("The JSON data is not an array.")