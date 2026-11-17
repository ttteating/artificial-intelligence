import json
import random
from datetime import datetime

CONDITIONS = ['B0', 'B1', 'A0', 'A1', 'A2', 'M0', 'M1', 'H']
LABELS = ['Compliant', 'Non-Compliant', 'Not-Applicable']
roles = ['drafter', 'reviewer', 'approver', 'signatory', 'obligor', 'obligee']

def generate_clauses(n=20):
    return {f"clause_{i:03d}": f"条款 {i} 的文本内容涉及合同履行和法律规定" for i in range(1, n+1)}