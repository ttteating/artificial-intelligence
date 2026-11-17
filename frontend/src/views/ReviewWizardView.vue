<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { endpoints } from '@/api/endpoints'
import type { Condition, ScenarioListItem } from '@/api/types'
import { useReviewStore } from '@/stores/review'
import { useSystemStore } from '@/stores/system'
import { conditionMeta } from '@/utils/meta'
import { labelCN, labelClass } from '@/utils/meta'
import { num } from '@/utils/format'

const router = useRouter()
const store = useReviewStore()
const system = useSystemStore()

const source = ref<'paste' | 'dataset'>('paste')
const title = ref('')
const scenarioText = ref('')
const clauses = ref<Array<{ text: string }>>([{ text: '' }, { text: '' }])
const creating = ref(false)
const createError = ref('')
const created = ref<{ sid: string; nodes: number; edges: number; relevant: string[]; sim: Record<string, number> } | null>(null)

const split = ref<'dev' | 'test'>('dev')
const q = ref('')
const list = ref<ScenarioListItem[]>([])
const total = ref(0)
const listLoading = ref(false)
const selectedSid = ref('')

const condition = ref<Condition>('H')
const budget = ref(16)
const explain = ref(true)
const explainMode = ref<'template' | 'real' | 'auto'>('template')

const conditions: Condition[] = ['B0', 'B1', 'A0', 'A1', 'A2', 'M0', 'M1', 'H']
const canRun = computed(() => {
  if (!system.ready) return false
  if (source.value === 'dataset') return Boolean(selectedSid.value)
  return Boolean(created.value?.sid)
})

let debounce: number | undefined
function fetchList(): void {
  window.clearTimeout(debounce)
  debounce = window.setTimeout(async () => {
    listLoading.value = true
    try {
      const res = await endpoints.scenarios({ split: split.value, q: q.value || undefined, limit: 8 })
      list.value = res.items
      total.value = res.total
    } catch {
      list.value = []
      total.value = 0
    } finally {
      listLoading.value = false
    }
  }, 260)
}

watch([split, q], fetchList)
onMounted(() => { system.start(); fetchList() })

function addClause(): void {
  if (clauses.value.length < 10) clauses.value.push({ text: '' })
}
function removeClause(index: number): void {
  if (clauses.value.length > 1) clauses.value.splice(index, 1)
}
function useSample(): void {
  title.value = '数据处理分包（示例）'
  scenarioText.value = '供应商在没有取得客户书面同意的情况下，将客户个人数据分包给位于境外的第三方处理，且未在约定的三十日内通知客户。'
  clauses.value = [
    { text: '供应商不得将本协议项下的数据处理义务分包给任何第三方，除非事先取得客户的书面同意。' },
    { text: '尽管有前述约定，供应商为提供技术支持之目的，可将必要数据提供给其关联方，但应提前三十日书面通知客户。' },
    { text: '任何违反本条的分包行为均构成实质性违约，客户有权立即终止本协议并要求赔偿。' },
  ]
  created.value = null
  createError.value = ''
}

async function create(): Promise<void> {
  createError.value = ''
  const text = scenarioText.value.trim()
  const filled = clauses.value.map((c) => c.text.trim()).filter(Boolean)
  if (text.length < 10) { createError.value = '请填写至少 10 个字符的场景描述'; return }
  if (!filled.length) { createError.value = '请至少填写一条合同条款'; return }
  creating.value = true
  created.value = null
  try {
    const payload: Record<string, string> = {}
    filled.forEach((t, i) => { payload[`evidence_${i + 1}`] = t })
    const res = await endpoints.createScenario({
      scenario_text: text,
      clauses: payload,
      title: title.value.trim() || undefined,
      persist: false,
    })
    created.value = {
      sid: res.sid, nodes: res.n_nodes, edges: res.n_edges,
      relevant: res.relevant, sim: res.clause_sim,
    }
    store.rememberTitle(res.sid, title.value.trim() || `自定义合同 ${res.sid}`)
  } catch (err) {
    createError.value = err instanceof Error ? err.message : String(err)
  } finally {
    creating.value = false
  }
}

function run(): void {
  const sid = source.value === 'dataset' ? selectedSid.value : created.value?.sid
  if (!sid) return
  router.push({
    path: `/review/${sid}`,
    query: {
      condition: condition.value,
      budget: String(budget.value),
      explain: explain.value ? '1' : '0',
      explain_mode: explainMode.value,
    },
  })
}
</script>

<template>
  <div class="content">
    <div class="page-head">
      <div class="grow">
        <div class="page-kicker">新建审查</div>
        <h1 class="page-title">把一份合同放进条款图，跑一次多跳推理</h1>
        <p class="page-sub">
          从客户合同或数据集场景出发，系统在 facet 级条款图上扩展「义务 → 例外 → 冲突 → 判定」，
          输出三分类结论与可点击追溯的证据链。
        </p>
      </div>
    </div>

    <div v-if="!system.ready && !system.error" class="banner info" style="margin-bottom: 16px">
      服务正在加载语料与条款图（首次约 10–60 秒）。就绪前无法运行推理，但可以先填合同。
    </div>
    <div v-if="system.error" class="banner bad" style="margin-bottom: 16px">
      无法连接推理服务：{{ system.error }}
    </div>

    <div class="split-2">
      <div class="stack" style="gap: 18px">
        <section class="card">
          <div class="card-head">
            <span class="tag mono">01</span>
            <div class="grow">
              <div class="card-title">载入案件</div>
              <div class="card-sub">客户自己的合同，或 ACE 数据集里的既有场景</div>
            </div>
            <div class="seg">
              <button type="button" :class="{ on: source === 'paste' }" @click="source = 'paste'">粘贴合同</button>
              <button type="button" :class="{ on: source === 'dataset' }" @click="source = 'dataset'">数据集场景</button>
            </div>
          </div>

          <div v-if="source === 'paste'" class="card-body stack" style="gap: 14px">
            <div class="row between">
              <span class="hint">条款 1–10 条，单条 ≤4,000 字符；默认只在内存，不落盘。</span>
              <button class="btn ghost sm" type="button" @click="useSample">填入示例</button>
            </div>

            <div class="field">
              <label class="label" for="title">案件名称（可选）</label>
              <input id="title" v-model="title" class="input" placeholder="例如：某 SaaS 采购主协议 · 数据分包条款" />
            </div>

            <div class="field">
              <label class="label" for="scenario">场景描述</label>
              <textarea
                id="scenario" v-model="scenarioText" class="textarea" rows="4"
                placeholder="描述待审事实：谁做了什么、缺少什么同意/通知、造成了什么后果…"
              />
              <span class="hint">{{ scenarioText.length }} / 20,000</span>
            </div>

            <div class="field">
              <div class="row between">
                <span class="label">合同条款</span>
                <button class="btn ghost sm" type="button" :disabled="clauses.length >= 10" @click="addClause">+ 添加条款</button>
              </div>
              <div v-for="(clause, i) in clauses" :key="i" class="row" style="align-items: flex-start; margin-top: 8px">
                <span class="tag mono" style="margin-top: 8px">evidence_{{ i + 1 }}</span>
                <textarea
                  v-model="clause.text" class="textarea grow" rows="2"
                  :placeholder="i === 0 ? '粘贴第一条条款原文…' : '再粘贴一条相关条款（例外、定义、责任限制…）'"
                />
                <button class="btn ghost icon" type="button" :disabled="clauses.length <= 1" title="删除" @click="removeClause(i)">×</button>
              </div>
            </div>

            <div v-if="createError" class="banner bad">{{ createError }}</div>

            <div class="row">
              <button class="btn primary" type="button" :disabled="creating || !system.ready" @click="create">
                {{ creating ? '正在建图…' : '登记并构建条款图' }}
              </button>
              <span v-if="created" class="small muted">
                已建图：{{ created.nodes }} 个 facet / {{ created.edges }} 条边 · sid
                <span class="mono">{{ created.sid }}</span>
              </span>
            </div>

            <div v-if="created" class="panel" style="background: var(--surface-2)">
              <div class="panel-head"><span class="panel-title">条款相关度</span></div>
              <div v-for="(sim, clause) in created.sim" :key="clause" class="metric-row">
                <span class="metric-label mono">{{ clause }}</span>
                <span class="metric-value">{{ num(sim, 3) }}</span>
              </div>
              <p class="tiny faint" style="margin-top: 6px">
                相关度基于训练集拟合的 TF-IDF；若全部偏低，说明合同用词与场景差异大，结论需更谨慎复核。
              </p>
            </div>
          </div>

          <div v-else class="card-body stack" style="gap: 12px">
            <div class="row wrap">
              <div class="seg">
                <button type="button" :class="{ on: split === 'dev' }" @click="split = 'dev'">dev</button>
                <button type="button" :class="{ on: split === 'test' }" @click="split = 'test'">test</button>
              </div>
              <input v-model="q" class="input grow" style="max-width: 280px" placeholder="按 sid 搜索，例如 tr00632" />
              <span class="tiny faint">共 {{ total }} 条</span>
            </div>

            <div class="table-wrap" style="max-height: 340px">
              <table class="tbl">
                <thead><tr><th>sid</th><th>标注</th><th class="n">条款</th><th class="n">跳数</th><th>H 结果</th></tr></thead>
                <tbody>
                  <tr
                    v-for="item in list" :key="item.sid" class="clickable"
                    :class="{ 'is-highlight': selectedSid === item.sid }"
                    @click="selectedSid = item.sid"
                  >
                    <td class="mono">{{ item.sid }}</td>
                    <td>
                      <span class="tag" :class="labelClass(item.gold)">{{ labelCN[item.gold] ?? item.gold }}</span>
                    </td>
                    <td class="n">{{ item.n_clauses }}</td>
                    <td class="n">{{ item.oracle_hops ?? '—' }}</td>
                    <td>
                      <span v-if="item.runs.H" class="tag" :class="item.runs.H.correct ? 'ok' : 'bad'">
                        {{ item.runs.H.correct ? '正确' : '错误' }} · {{ num(item.runs.H.conf, 2) }}
                      </span>
                      <span v-else class="faint">—</span>
                    </td>
                  </tr>
                  <tr v-if="!list.length"><td colspan="5" class="faint">{{ listLoading ? '加载中…' : '没有匹配场景' }}</td></tr>
                </tbody>
              </table>
            </div>
            <p class="tiny faint">点击一行选中；场景原文与全部条款会在结果页左侧展示。</p>
          </div>
        </section>

        <section class="card">
          <div class="card-head">
            <span class="tag mono">02</span>
            <div class="grow">
              <div class="card-title">配置推理策略</div>
              <div class="card-sub">八个条件对应主表里的八组消融，默认 H（规则 + 分类器 + GA h + LLM）</div>
            </div>
          </div>
          <div class="card-body">
            <div class="grid cols-4" style="gap: 10px">
              <button
                v-for="c in conditions" :key="c" type="button"
                class="condition-card" :class="{ on: condition === c }" @click="condition = c"
              >
                <div class="row between">
                  <span class="condition-id">{{ c }}</span>
                  <span v-if="c === 'H'" class="tag accent" style="height: 18px; padding: 0 5px">推荐</span>
                </div>
                <span class="condition-desc">{{ conditionMeta[c].name }}</span>
              </button>
            </div>

            <div class="divider" />

            <div class="row wrap" style="gap: 20px; align-items: flex-start">
              <div class="field" style="min-width: 220px; flex: 1">
                <label class="label" for="budget">扩展预算 · {{ budget }}</label>
                <input id="budget" v-model.number="budget" class="range" type="range" min="1" max="48" step="1" />
                <span class="hint">主表 H 平均只用到 4.53 个扩展；预算越低，越考验启发式排序。</span>
              </div>
              <div class="field" style="min-width: 200px">
                <span class="label">推理说明</span>
                <label class="switch"><input v-model="explain" type="checkbox" /> 生成证据链解释</label>
                <select v-model="explainMode" class="select" :disabled="!explain">
                  <option value="template">template · 离线确定性</option>
                  <option value="real">real · 需 LLM_API_KEY</option>
                  <option value="auto">auto · 有 key 用真实</option>
                </select>
              </div>
            </div>
          </div>
        </section>

        <section class="card">
          <div class="card-head">
            <span class="tag mono">03</span>
            <div class="grow">
              <div class="card-title">运行推理</div>
              <div class="card-sub">现场执行与主表实验完全相同的装配路径</div>
            </div>
          </div>
          <div class="card-body row wrap">
            <button class="btn primary" type="button" :disabled="!canRun" @click="run">运行推理</button>
            <span v-if="!system.ready" class="small muted">等待服务就绪…</span>
            <span v-else-if="!canRun" class="small muted">请先登记合同或选择一个数据集场景</span>
            <span v-else class="small muted">
              条件 {{ condition }} · 预算 {{ budget }} · {{ explain ? '含解释' : '不含解释' }}
            </span>
          </div>
        </section>
      </div>

      <aside class="stack" style="position: sticky; top: calc(var(--topbar) + 20px)">
        <div class="panel">
          <div class="panel-head"><span class="panel-title">这一页会做什么</span></div>
          <ol class="small muted" style="padding-left: 16px; line-height: 1.9">
            <li>把场景与条款登记进语料，按句切分为 facet 节点；</li>
            <li>用规则锚点连接成图（例外 / 定义 / 引用 / 冲突…）；</li>
            <li>按所选条件在图上搜索，直到满足三项目标测试；</li>
            <li>给出三分类结论，并把证据链、图、树三者联动展示。</li>
          </ol>
          <div class="divider" />
          <p class="tiny faint" style="line-height: 1.7">
            自定义合同默认只在内存中登记，服务重启即释放；勾选持久化才会写入
            <code>data/custom/</code>。
          </p>
        </div>
        <div class="panel">
          <div class="panel-head"><span class="panel-title">服务状态</span></div>
          <div class="stack" style="gap: 6px">
            <div class="row between small"><span class="muted">就绪</span><span>{{ system.ready ? '是' : '否' }}</span></div>
            <div class="row between small"><span class="muted">场景数</span><span>{{ system.health?.n_scenarios?.toLocaleString() ?? '—' }}</span></div>
            <div class="row between small"><span class="muted">图版本</span><span class="mono">{{ system.health?.graph_version || '—' }}</span></div>
            <div class="row between small"><span class="muted">LLM 模式</span><span class="mono">{{ system.health?.llm_mode || '—' }}</span></div>
          </div>
        </div>
      </aside>
    </div>
  </div>
</template>
