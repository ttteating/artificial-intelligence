import json, random, os
os.chdir('d:\\program\\artificial intelligence')

CONDITIONS = ['B0', 'B1', 'A0', 'A1', 'A2', 'M0', 'M1', 'H']
LABELS = ['Compliant', 'Non-Compliant', 'Not-Applicable']
roles_desc = [
    '合同履行相关',
    '违约责任条款',
    '付款条款',
    '交付时间',
    '质量标准',
    '争议解决',
    '知识产权',
    '保密条款',
    '违约金规定',
    '终止条款'
]

# 1. 生成场景基础文本
scenario_text = """合同编号：SC-2024-01892
合同类型：软件服务协议
合同金额：人民币壹拾万元整
合同期限：自2024年01月01日至2024年12月31日

本合同涉及软件产品的许可、实施、验收及后续维护服务。双方就软件的交付标准、服务水平、知识产权归属、违约责任等事项达成一致。现进行合同合规审查，重点检查条款的合法性、完整性及执行风险。"""

# 2. 生成条款文本 (18个条款)
n_clauses = 18
clauses = {}
for i in range(1, n_clauses + 1):
    clause_key = 'clause_' + str(i).zfill(3)
    clause_text = '条款' + str(i) + ': ' + random.choice(roles_desc) + '。本条款旨在规范双方权利义务，具体包括但不限于履行标准、违约后果及争议处理机制。具体内容涉及合同履行过程中的关键条款规定，重点关注各方的权益保障和风险分配。'
    clauses[clause_key] = clause_text

# 3. 随机选择条件和标签
condition = random.choice(CONDITIONS)
answer = random.choice(LABELS)

# 4. 生成搜索参数
search_params = {
    'sid': 'tr' + str(random.randint(1000, 9999)).zfill(4),
    'condition': condition,
    'explain': True,
    'explain_mode': random.choice(['template', 'real', 'auto'])
}

# 5. 构建完整场景数据
scenario = {
    'scenario_text': scenario_text,
    'clauses': clauses,
    'condition': condition,
    'answer': answer,
    'search_params': search_params,
    'created_at': '2024-01-15 10:30:00'
}

# 6. 保存文件
with open('mock_new_review_scenario.json', 'w', encoding='utf-8') as f:
    json.dump(scenario, f, ensure_ascii=False, indent=2)

with open('mock_new_review_params.json', 'w', encoding='utf-8') as f:
    json.dump(search_params, f, ensure_ascii=False, indent=2)

print('模拟数据生成完成！')
print('场景SID:', search_params['sid'])
print('条件:', condition)
print('判定:', answer)
print('条款数量:', n_clauses)
print('文件已保存：')
print(' - mock_new_review_scenario.json')
print(' - mock_new_review_params.json')