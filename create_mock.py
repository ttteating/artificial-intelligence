import json, os
os.chdir('d:\\program\\artificial intelligence')

# Generate clauses
clauses = {}
for i in range(1, 21):
    key = "clause_" + str(i).zfill(3)
    clauses[key] = "条款" + str(i) + "文本"
with open("mock_clauses.json", "w", encoding="utf-8") as f:
    json.dump(clauses, f, ensure_ascii=False, indent=2)
print("mock_clauses.json generated")