import json
import re


def extract_table_names(sql_statement):
    # 正则表达式匹配SQL语句中的表名，包括可能存在的模式名
    pattern = r'(?:FROM|JOIN|UPDATE|INSERT INTO)\s+(?:\w+\.)?(\w+)'

    # 使用findall方法找到所有匹配的表名
    matches = re.findall(pattern, sql_statement, re.IGNORECASE)
    for item in matches:
        if item == 'FROM':
            matches.remove('FROM')
    # 提取表名部分并去除重复
    table_names = set(matches)

    return table_names


def replace_table_names(sql_statement, ddl_data):
    # 提取SQL语句中的表名
    table_names = extract_table_names(sql_statement)

    # 创建一个新的SQL语句，用于替换表名
    sql_statement_list = []

    # 遍历表名并替换
    for table_name in table_names:
        # 检查表名是否存在于DDL数据中
        if table_name in ddl_data:
            new_name = ddl_data[table_name]
            sql_statement_list.append(new_name)
        else:
            raise ValueError(f"Table '{table_name}' not found in DDL data.")

    return sql_statement_list


if __name__ == '__main__':
    file_name = './data/test_db_hub_cn_0603.json'
    ddl_file_name = './data/DDL_EMS.json'
    query_ddl_file_name = './data/query_ddl.json'

    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)

    with open(ddl_file_name, 'r', encoding='utf-8') as file:
        ddl_data = json.load(file)


    for item in data:
        sql_statement = item['output']
        ddl_json = {}
        try:
            new_sql_statement = replace_table_names(sql_statement, ddl_data)
            pattern = r"###输入:\n(.*?)\n\n###回答:"
            # 使用正则表达式搜索并提取匹配的部分
            match = re.search(pattern, item["input"])
            extracted_part = match.group(1)
            ddl_json["query"] = extracted_part
            for i in range(len(new_sql_statement)):
                ddl_json["ddl" + str(i + 1)] = new_sql_statement[i]
        except Exception as e:
            print(sql_statement)

        with open(query_ddl_file_name, 'r+', encoding='utf-8') as file:
            try:
                # 尝试读取现有JSON数据
                existing_data = json.load(file)
            except json.JSONDecodeError:
                # 如果文件为空或内容不是有效的JSON，初始化一个空字典
                existing_data = {}
            existing_data.update(ddl_json)
            json.dump(existing_data, file, indent=4, ensure_ascii=False)
            file.write(',\n')

    with open(query_ddl_file_name, 'r+', encoding='utf-8') as file:
        content = file.read()
        file.seek(0, 0)
        file.write('[')
        file.write('\n')
        content = content.rstrip(', \n\t')
        file.write(content)
        file.write('\n')
        file.write(']')