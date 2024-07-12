import json
import re
import random

# 示例JSON数组
with open('0711_suowei_sql.json', 'r', encoding='utf-8') as file:
    json_array = json.load(file)

original_count = len(json_array)
print(f"改变前的条数: {original_count}")

# 正则表达式模式
pattern = re.compile(r'(东莞星河城|厦门乾照光电有限公司)')

# 提取符合条件的JSON对象
filtered_json_array = [item for item in json_array if pattern.search(item['input'])]

# 限制结果数量为300条
filtered_json_array = filtered_json_array[:300]

results = []
for item in filtered_json_array:
    instruction = "\n##指令 我想让你充当一个示例数据库前面的SQL终端。, 你只需要把sql命令发给我。下面是一个描述任务的指令, 写一个适当的回复来完成请求。\n\n##建表语句为：\nCREATE TABLE `suowei_table_project` (\n  `id` bigint NOT NULL AUTO_INCREMENT COMMENT 'id',\n  `name` varchar(100)  DEFAULT NULL COMMENT '项目名称',\n  `create_time` datetime DEFAULT NULL,\n  `update_time` datetime DEFAULT NULL,\n  `project_id` varchar(100) DEFAULT NULL COMMENT '项目id',\n  `project_describe` varchar(100) DEFAULT NULL COMMENT '项目描述',\n  `energy_saving_rate` varchar(100) DEFAULT NULL COMMENT '节能率',\n  `energy_radio` varchar(100) DEFAULT NULL COMMENT '节电量',\n  `electric_radio` varchar(100) DEFAULT NULL COMMENT '节能比例',\n  `volatility` varchar(100) DEFAULT NULL COMMENT '电量波动率',\n  `project_short_id` varchar(100) DEFAULT NULL COMMENT '项目短id',\n  PRIMARY KEY (`id`)\n) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE= utf8mb4_general_ci COMMENT='所为项目表';\n\nCREATE TABLE `suowei_table_system` (\n  `id` bigint NOT NULL AUTO_INCREMENT,\n  `system_id` varchar(100)   DEFAULT NULL COMMENT '系统id',\n  `system_name` varchar(100) DEFAULT NULL COMMENT '系统名称',\n  `system_short_id` varchar(100)   DEFAULT NULL COMMENT '系统短id',\n  `system_type` varchar(100)   DEFAULT NULL COMMENT '系统类型',\n  `project_id` varchar(100)   DEFAULT NULL COMMENT '所属项目id',\n  `create_time` datetime DEFAULT NULL,\n  `update_time` datetime DEFAULT NULL,\n  `system_describe` varchar(100) DEFAULT NULL COMMENT '描述',\n  `energy_saving_rate` varchar(255) DEFAULT NULL COMMENT '节能率',\n  PRIMARY KEY (`id`)\n) ENGINE=InnoDB AUTO_INCREMENT=10 DEFAULT CHARSET=utf8mb4 COLLATE= utf8mb4_general_ci COMMENT='所为系统表';\n\nCREATE TABLE `suowei_table_group` (\n  `id` bigint NOT NULL AUTO_INCREMENT,\n  `group_id` varchar(100)   DEFAULT NULL COMMENT '分组id',\n  `group_name` varchar(100)   DEFAULT NULL COMMENT '分组名称',\n  `project_id` varchar(100)   DEFAULT NULL COMMENT '项目id',\n  `system_id` varchar(100) DEFAULT NULL COMMENT '系统id',\n  `create_time` datetime DEFAULT NULL,\n  `update_time` datetime DEFAULT NULL,\n  `group_type` varchar(100) DEFAULT NULL COMMENT '分组类型',\n  PRIMARY KEY (`id`)\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE= utf8mb4_general_ci COMMENT='设备分组信息';\n\nCREATE TABLE `suowei_table_device_relation` (\n  `id` bigint NOT NULL AUTO_INCREMENT,\n  `device_id` varchar(100) DEFAULT NULL COMMENT '设备id',\n  `project_id` varchar(100) DEFAULT NULL COMMENT '项目id',\n  `system_id` varchar(100) DEFAULT NULL COMMENT '系统id',\n  `group_id` varchar(100) DEFAULT NULL COMMENT '分组id',\n  `create_time` datetime DEFAULT NULL,\n  `update_time` varchar(100) DEFAULT NULL,\n  PRIMARY KEY (`id`)\n) ENGINE=InnoDB AUTO_INCREMENT=15 DEFAULT CHARSET=utf8mb4 COLLATE= utf8mb4_general_ci;\n\nCREATE TABLE `suowei_table_device` (\n  `id` bigint NOT NULL AUTO_INCREMENT COMMENT 'id',\n  `device_id` varchar(100)   DEFAULT NULL COMMENT '设备id',\n  `name` varchar(255)   DEFAULT NULL COMMENT '设备名称',\n  `type` varchar(100)   DEFAULT NULL COMMENT '设备类型',\n  `model` varchar(100)   DEFAULT NULL COMMENT '设备型号',\n  `description` text   COMMENT '设备描述',\n  `create_time` datetime DEFAULT NULL COMMENT '创建时间',\n  `update_time` datetime DEFAULT NULL COMMENT '更新时间',\n  `device_short_id` varchar(100) DEFAULT NULL COMMENT '设备短id',\n  `meter_id` varchar(100) DEFAULT NULL COMMENT '关联的计量表id',\n  PRIMARY KEY (`id`)\n) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE= utf8mb4_general_ci COMMENT='设备信息表';\n\nCREATE TABLE `suowei_table_device_meter` (\n  `id` bigint NOT NULL AUTO_INCREMENT,\n  `meter_id` varchar(100) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '计量表id',\n  `meter_type` varchar(50) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '计量表类型（电表、水表、能量计）',\n  `action_type` varchar(50) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '作用类型（独立电表、共享电表）',\n  `meter_name` varchar(100) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '计量表名称',\n  `system_id` varchar(100) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '所属系统id',\n  `project_id` varchar(100) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '所属项目id',\n  `create_time` datetime DEFAULT NULL COMMENT '创建时间',\n  `update_time` datetime DEFAULT NULL COMMENT '更新时间',\n  PRIMARY KEY (`id`)\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='设备计量表信息表';\n\nCREATE TABLE `suowei_stats_day_meter` (\n  `id` bigint NOT NULL AUTO_INCREMENT,\n  `device_id` varchar(100) DEFAULT NULL COMMENT '设备id',\n  `dosage` decimal(10,2) DEFAULT NULL COMMENT '能耗',\n  `stats_time` date DEFAULT NULL COMMENT '统计时间',\n  `create_time` datetime DEFAULT NULL,\n  `update_time` datetime DEFAULT NULL,\n  PRIMARY KEY (`id`)\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE= utf8mb4_general_ci COMMENT='所为日汇总表（按天统计每个设备的用能）';\n\nCREATE TABLE `suowei_stats_month_meter` (\n  `id` bigint NOT NULL AUTO_INCREMENT,\n  `device_id` varchar(100) DEFAULT NULL COMMENT '设备id',\n  `dosage` decimal(10,2) DEFAULT NULL COMMENT '能耗',\n  `month` int DEFAULT NULL COMMENT '月份',\n  `year` int DEFAULT NULL COMMENT '年份',\n  `create_time` datetime DEFAULT NULL,\n  `update_time` datetime DEFAULT NULL,\n  PRIMARY KEY (`id`)\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE= utf8mb4_general_ci COMMENT='所为月汇总表（按月统计每个设备的用能）';\n\n ##要求 \n我将给你一个表间层级关联关系，你需要按需连接到具体的层级\n ##项目\n你当前处于{project}项目\n\n##表间层级连接关系为： \nsuowei_table_project project join suowei_table_system sys on project.project_id  = sys.project_id \nleft join suowei_table_group stg on stg.system_id = sys.system_id and stg.project_id = project.project_id \njoin suowei_table_device_relation relation on stg.group_id = relation.group_id or relation.system_id = sys.system_id\njoin suowei_table_device device on relation.device_id  = device.device_id\njoin suowei_table_device_meter meter on device.meter_id = meter.meter_id\n\n##问题：\n"
    if '东莞星河城' in item['input']:
        project = "'厦门乾照光电有限公司'"
    if '厦门乾照光电有限公司' in item['input']:
        project = "'东莞星河城'"
    instruction = instruction.replace("{project}", project)
    item['instruction'] = instruction
    results.append(item)

# 将results中的结果添加到 json_array中
json_array.extend(results)

# 打乱顺序
random.shuffle(json_array)

# 输出成新的json文件
with open('train_suowei_0711.json', 'w', encoding='utf-8') as file:
    json.dump(json_array, file, ensure_ascii=False, indent=4)

print(f"改变后的条数: {len(json_array)}")

