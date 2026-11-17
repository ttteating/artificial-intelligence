import type { Label } from '@/api/types'

export const labelCN: Record<string, string> = {
  Compliant: '合规',
  'Non-Compliant': '不合规',
  'Not-Applicable': '不适用',
  Unlabeled: '未标注',
}

export function labelClass(label?: string | null): 'ok' | 'bad' | 'na' {
  if (label === 'Compliant') return 'ok'
  if (label === 'Non-Compliant') return 'bad'
  return 'na'
}

export const roleMeta: Record<string, { cn: string; color: string }> = {
  OBL: { cn: '义务主句', color: '#3f3f46' },
  EXC: { cn: '例外句', color: '#c0362c' },
  COND: { cn: '条件句', color: '#4f4fc4' },
  DEF: { cn: '定义句', color: '#0f766e' },
  SCOPE: { cn: '引用句', color: '#7c3aed' },
  REMEDY: { cn: '救济句', color: '#a16207' },
}

export const edgeMeta: Record<string, { cn: string; color: string; dash?: string }> = {
  ENTRY: { cn: '入口', color: '#a1a1aa' },
  DEFINES: { cn: '定义', color: '#0f766e' },
  COMPLEMENTS: { cn: '补充', color: '#71717a' },
  EXCEPTIONS: { cn: '例外', color: '#c0362c' },
  DEPENDS_ON: { cn: '依赖', color: '#4f4fc4' },
  CONFLICTS: { cn: '冲突', color: '#a16207', dash: '5 4' },
  SEQUENCE: { cn: '顺序', color: '#a8a29e' },
  CITES: { cn: '引用', color: '#7c3aed', dash: '2 3' },
}

export const conditionMeta: Record<string, { name: string; kernel: string }> = {
  B0: { name: 'BFS 盲目基准', kernel: 'BFS' },
  B1: { name: 'BFS + 规则', kernel: 'BFS' },
  A0: { name: 'A* + 手工 h', kernel: 'A*' },
  A1: { name: 'A* + GA h', kernel: 'A*' },
  A2: { name: 'A* + GA h + LLM', kernel: 'A*' },
  M0: { name: 'MCTS 基准', kernel: 'MCTS' },
  M1: { name: 'MCTS + LLM + 分类器', kernel: 'MCTS' },
  H: { name: '混合架构', kernel: 'A*' },
}

export const statusMeta: Record<string, { cn: string; color: string }> = {
  expanded: { cn: '已扩展', color: '#3f3f46' },
  frontier: { cn: '边界', color: '#a1a1aa' },
  pruned: { cn: '被剪', color: '#d4d4d8' },
  goal: { cn: '目标', color: '#0e8a6e' },
  deadend: { cn: '死路', color: '#c0362c' },
}

export const flagMeta: Record<string, { cn: string; color: string }> = {
  success_path: { cn: '成功路径', color: '#0e8a6e' },
  redundant: { cn: '冗余分支', color: '#a1a1aa' },
  llm_rejected: { cn: 'LLM 驳回', color: '#c0362c' },
  llm_misjudge: { cn: 'LLM 误判', color: '#a16207' },
  rule_pruned: { cn: '规则剪枝', color: '#71717a' },
}

export const ruleMeta: Record<string, string> = {
  R1: '无关扩展阻断',
  R2: '深链截断',
  R3: '冲突边双证据',
  R4: '例外未解不下结论',
  R5: '环检测',
  R6: '失效条款排除',
}

export function labelOf(value?: string | null): Label | string {
  return labelCN[value ?? ''] ? value! : (value ?? '—')
}
