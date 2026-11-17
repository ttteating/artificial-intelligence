<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import type { Condition } from '@/api/types'
import { useReviewStore } from '@/stores/review'
import { useSystemStore } from '@/stores/system'
import VerdictBar from '@/components/review/VerdictBar.vue'
import EvidenceChain from '@/components/review/EvidenceChain.vue'
import SignalPanel from '@/components/review/SignalPanel.vue'
import ExplanationCard from '@/components/review/ExplanationCard.vue'
import TraceTable from '@/components/review/TraceTable.vue'
import ClausePanel from '@/components/review/ClausePanel.vue'
import ClauseGraph from '@/components/graph/ClauseGraph.vue'
import SearchTree from '@/components/tree/SearchTree.vue'
import EmptyState from '@/components/ui/EmptyState.vue'
import { conditionMeta } from '@/utils/meta'
import { buildChain } from '@/utils/review'
import { useToast } from '@/utils/toast'

const route = useRoute()
const router = useRouter()
const store = useReviewStore()
const system = useSystemStore()
const { toast } = useToast()

const tab = ref<'graph' | 'tree' | 'clauses' | 'trace'>('graph')
const conditions: Condition[] = ['B0', 'B1', 'A0', 'A1', 'A2', 'M0', 'M1', 'H']

const sid = computed(() => String(route.params.sid ?? ''))
const condition = computed<Condition>(() => (route.query.condition as Condition) || 'H')
const budget = computed(() => Number(route.query.budget) || 16)
const explain = computed(() => route.query.explain !== '0')
const explainMode = computed(() => (route.query.explain_mode as 'template' | 'real' | 'auto') || 'template')

const result = computed(() => store.result)
const chain = computed(() => buildChain(store.result, store.graph))
const suf = computed(() => result.value?.explanation?.sufficiency ?? null)
const scenarioText = computed(() => result.value?.meta.scenario ?? store.scenario?.text ?? '')
const title = computed(() => store.titles[sid.value] || (sid.value.startsWith('cus_') ? '自定义合同' : '数据集场景'))

const selectedNid = computed(() => {
  const sel = store.selection
  if (!sel) return null
  if (sel.kind === 'step') return chain.value[sel.index - 1]?.to ?? null
  if (sel.kind === 'node') return sel.nid
  if (sel.kind === 'clause') return store.graph?.nodes.find((n) => n.clause === sel.clause)?.nid ?? null
  if (sel.kind === 'tree') return result.value?.tree?.find((n) => n.tid === sel.tid)?.node_id ?? null
  return null
})

const selectedTid = computed(() => {
  const sel = store.selection
  const tree = result.value?.tree ?? []
  if (!sel) return null
  if (sel.kind === 'tree') return sel.tid
  if (sel.kind === 'step') {
    const step = chain.value[sel.index - 1]
    if (!step) return null
    const hit = [...tree].reverse().find((n) => n.action
      && n.action.src === step.from && n.action.edge_type === step.edge && n.action.dst === step.to)
    return hit?.tid ?? null
  }
  if (sel.kind === 'node') return tree.find((n) => n.node_id === sel.nid)?.tid ?? null
  return null
})

const activeStep = computed(() => (store.selection?.kind === 'step' ? store.selection.index : null))

function onStep(index: number): void {
  store.select({ kind: 'step', index })
}
function onNode(nid: string): void {
  const idx = chain.value.findIndex((s) => s.to === nid)
  store.select(idx >= 0 ? { kind: 'step', index: idx + 1 } : { kind: 'node', nid })
}
function onTree(tid: number): void {
  const node = result.value?.tree?.find((n) => n.tid === tid)
  if (node?.action) {
    const idx = chain.value.findIndex((s) => s.to === node.action!.dst
      && s.from === node.action!.src && s.edge === node.action!.edge_type)
    if (idx >= 0) { store.select({ kind: 'step', index: idx + 1 }); return }
  }
  store.select({ kind: 'tree', tid })
}
function onClause(clause: string): void {
  tab.value = 'clauses'
  store.select({ kind: 'clause', clause })
}

function applyCondition(next: Condition): void {
  router.replace({ query: { ...route.query, condition: next } })
}
function applyBudget(next: string): void {
  router.replace({ query: { ...route.query, budget: next } })
}
function rerun(): void {
  store.run(sid.value, condition.value, budget.value, explain.value, explainMode.value)
}
function printPage(): void {
  window.print()
}
async function copyLink(): Promise<void> {
  try {
    await navigator.clipboard.writeText(window.location.href)
    toast('已复制可复现链接')
  } catch {
    toast('复制失败，请手动复制地址栏')
  }
}

let lastKey = ''
function load(): void {
  const key = `${sid.value}|${condition.value}|${budget.value}|${explain.value}|${explainMode.value}`
  if (key === lastKey) return
  lastKey = key
  void store.run(sid.value, condition.value, budget.value, explain.value, explainMode.value)
}

watch([sid, condition, budget, explain, explainMode], () => load(), { immediate: true })
watch(() => system.ready, (ready) => { if (ready) load() })
watch(sid, () => { store.selection = null; tab.value = 'graph' })

onBeforeUnmount(() => store.reset())
</script>

<template>
  <div class="content">
    <div v-if="!system.ready && !system.error" class="banner info" style="margin-bottom: 14px">
      服务正在加载 4,700 个场景与条款图（首次约 10–60 秒）。页面会在就绪后自动发起推理。
    </div>
    <div v-if="system.error" class="banner bad" style="margin-bottom: 14px">
      无法连接推理服务：{{ system.error }}。请确认 <code>python -m api</code> 已启动。
    </div>

    <div class="page-head">
      <div class="grow">
        <div class="page-kicker">合同审查 · {{ sid }}</div>
        <h1 class="page-title">{{ title }}</h1>
        <p v-if="scenarioText" class="page-sub clamp-2">{{ scenarioText }}</p>
      </div>
      <div class="row wrap no-print">
        <select class="select" style="width: 132px" :value="condition" @change="applyCondition(($event.target as HTMLSelectElement).value as Condition)">
          <option v-for="c in conditions" :key="c" :value="c">{{ c }} · {{ conditionMeta[c].kernel }}</option>
        </select>
        <select class="select" style="width: 96px" :value="String(budget)" @change="applyBudget(($event.target as HTMLSelectElement).value)">
          <option v-for="b in [4, 8, 12, 16, 24, 32, 48]" :key="b" :value="b">预算 {{ b }}</option>
        </select>
        <button class="btn" type="button" :disabled="store.loading" @click="rerun">重跑</button>
        <button class="btn ghost" type="button" @click="copyLink">复制链接</button>
        <button class="btn ghost" type="button" @click="printPage">导出</button>
      </div>
    </div>

    <div v-if="store.loading" class="stack" style="gap: 16px">
      <div class="verdict">
        <div class="sk" style="width: 62px; height: 62px; border-radius: 50%" />
        <div class="grow">
          <div class="sk sk-line" style="width: 140px; height: 20px" />
          <div class="sk sk-line" style="width: 220px; margin-top: 10px" />
        </div>
        <div class="muted small">{{ store.stage || '推理中' }} · {{ (store.elapsed / 1000).toFixed(1) }}s</div>
      </div>
      <div class="review-grid">
        <div class="review-col"><div class="sk" style="height: 260px" /></div>
        <div class="review-col"><div class="sk" style="height: 520px" /></div>
        <div class="review-col"><div class="sk" style="height: 360px" /></div>
      </div>
    </div>

    <div v-else-if="store.error" class="banner bad">
      推理失败：{{ store.error }}
      <button class="btn sm" style="margin-left: 10px" type="button" @click="rerun">重试</button>
    </div>

    <template v-else-if="result">
      <VerdictBar :result="result" :budget="budget" />

      <div class="review-grid">
        <div class="review-col">
          <div class="panel">
            <div class="panel-head">
              <span class="panel-title">证据链</span>
              <span class="tiny faint">{{ chain.length }} 步</span>
            </div>
            <div v-if="suf" class="tiny muted" style="line-height: 1.7; margin-bottom: 10px">
              强相关条款 {{ suf.relevant_clauses.length }} 条，已审查 {{ suf.covered_clauses.length }} 条；
              {{ suf.exception_required ? (suf.exception_ok ? '例外已解开' : '例外未解开') : '本场景不强制解开例外' }}。
            </div>
            <EvidenceChain :steps="chain" :active="activeStep" @select="onStep" />
            <div v-if="!chain.length" class="empty" style="padding: 24px 10px">
              <div class="empty-title">预算内没有形成有效证据链</div>
              <div class="empty-text">搜索未同时满足覆盖、例外与置信度三项测试。可提高扩展预算或更换条件重跑。</div>
            </div>
          </div>
        </div>

        <div class="review-col">
          <div>
            <div class="tabs no-print">
              <button class="tab" :class="{ on: tab === 'graph' }" type="button" @click="tab = 'graph'">条款图</button>
              <button class="tab" :class="{ on: tab === 'tree' }" type="button" @click="tab = 'tree'">
                搜索树 <span class="count">{{ result.tree?.length ?? 0 }}</span>
              </button>
              <button class="tab" :class="{ on: tab === 'clauses' }" type="button" @click="tab = 'clauses'">
                原文条款 <span class="count">{{ result.meta.n_clauses }}</span>
              </button>
              <button class="tab" :class="{ on: tab === 'trace' }" type="button" @click="tab = 'trace'">
                轨迹表 <span class="count">{{ result.trace?.length ?? 0 }}</span>
              </button>
            </div>

            <div v-show="tab === 'graph'" style="margin-top: 12px">
              <ClauseGraph
                v-if="store.graph" :graph="store.graph" :path="result.path"
                :selected-nid="selectedNid" @select="onNode"
              />
              <div v-else class="sk" style="height: 520px" />
            </div>

            <div v-show="tab === 'tree'" style="margin-top: 12px">
              <SearchTree :nodes="result.tree" :selected-tid="selectedTid" @select="onTree" />
            </div>

            <div v-show="tab === 'clauses'" style="margin-top: 12px">
              <ClausePanel
                :clauses="result.meta.clauses" :graph="store.graph"
                :covered="suf?.covered_clauses ?? []" :relevant="suf?.relevant_clauses ?? []"
                :selected="store.selection?.kind === 'clause' ? store.selection.clause : null"
                @select="onClause"
              />
            </div>

            <div v-show="tab === 'trace'" style="margin-top: 12px">
              <TraceTable :rows="result.trace" />
            </div>
          </div>

          <ExplanationCard :explanation="result.explanation" @cite="onClause" />
        </div>

        <div class="review-col sticky-col">
          <SignalPanel :result="result" :budget="budget" />
          <div class="panel">
            <div class="panel-head"><span class="panel-title">案件信息</span></div>
            <div class="stack" style="gap: 6px">
              <div class="row between small"><span class="muted">场景</span><span class="mono">{{ result.sid }}</span></div>
              <div class="row between small"><span class="muted">条款数</span><span>{{ result.meta.n_clauses }}</span></div>
              <div class="row between small"><span class="muted">强相关</span><span>{{ result.meta.n_relevant }}</span></div>
              <div class="row between small"><span class="muted">需解开例外</span><span>{{ result.meta.needs_exc ? '是' : '否' }}</span></div>
              <div class="row between small">
                <span class="muted">数据集标注</span>
                <span>{{ result.meta.gold === 'Unlabeled' ? '—' : result.meta.gold }}</span>
              </div>
              <div class="row between small"><span class="muted">图版本</span><span class="mono">{{ store.graph?.graph_version ?? '—' }}</span></div>
            </div>
            <div class="divider" />
            <p class="tiny faint" style="line-height: 1.7">
              结果为辅助审查输出，需人工复核。LLM 不输出合规标签，只参与动作校验、路径打分与事后解释。
            </p>
          </div>
        </div>
      </div>
    </template>

    <EmptyState
      v-else-if="!store.loading" title="没有结果"
      text="请从场景库或新建审查进入；直接访问该地址时会自动运行一次默认条件。"
    />
  </div>
</template>
