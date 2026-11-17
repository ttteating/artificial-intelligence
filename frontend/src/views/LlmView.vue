<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts'
import { endpoints } from '@/api/endpoints'
import { renderMarkdown } from '@/utils/markdown'
import { num, pct } from '@/utils/format'

interface RejectRow {
  condition: string
  scenarios: number
  llm_vetoes: number
  false_kills: number
  false_kill_rate: number
  suspicious?: number
}

const report = ref<Awaited<ReturnType<typeof endpoints.llmReport>> | null>(null)
const error = ref('')
const tab = ref<'cost' | 'real' | 'a2' | 'smoke'>('cost')
const chartEl = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null

const rejectRows = computed<RejectRow[]>(() => {
  const rows = (report.value?.reject?.rows ?? []) as unknown as RejectRow[]
  return rows.filter((r) => r && r.condition)
})
const reference = computed(() => report.value?.reject?.reference ?? 0)

const md = computed(() => {
  const r = report.value
  if (!r) return ''
  if (tab.value === 'cost') return r.cost_md
  if (tab.value === 'real') return r.real_vs_mock_md
  if (tab.value === 'a2') return r.real_vs_mock_a2_md
  return r.smoke_md
})

function renderChart(): void {
  if (!chartEl.value || !rejectRows.value.length) return
  chart ??= echarts.init(chartEl.value)
  chart.setOption({
    grid: { left: 50, right: 20, top: 30, bottom: 30 },
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category', data: rejectRows.value.map((r) => r.condition),
      axisLine: { lineStyle: { color: '#eaeaec' } }, axisTick: { show: false },
      axisLabel: { fontSize: 11, color: '#71717a' },
    },
    yAxis: {
      type: 'value', max: 1, axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%`, fontSize: 10, color: '#a1a1aa' },
      splitLine: { lineStyle: { color: '#f4f4f5' } },
    },
    series: [{
      type: 'bar', barWidth: 28,
      itemStyle: { color: '#d4d4d8', borderRadius: [3, 3, 0, 0] },
      label: { show: true, position: 'top', formatter: (p: { value: number }) => pct(p.value), fontSize: 10, color: '#71717a' },
      data: rejectRows.value.map((r) => r.false_kill_rate),
      markLine: {
        silent: true, symbol: 'none',
        lineStyle: { color: '#c0362c', type: 'dashed' },
        data: [{ yAxis: 0.2 }],
        label: { formatter: '20%', fontSize: 10, color: '#c0362c' },
      },
    }],
  })
}

async function load(): Promise<void> {
  try {
    report.value = await endpoints.llmReport()
    await nextTick()
    renderChart()
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

function onResize(): void { chart?.resize() }
onMounted(async () => { await load(); window.addEventListener('resize', onResize) })
watch(tab, () => { /* markdown 直接切换，无需重绘 */ })
onBeforeUnmount(() => {
  window.removeEventListener('resize', onResize)
  chart?.dispose()
})
</script>

<template>
  <div class="content">
    <div class="page-head">
      <div class="grow">
        <div class="page-kicker">LLM 治理</div>
        <h1 class="page-title">LLM 只做三件事，且必须可审计</h1>
        <p class="page-sub">
          动作合法性校验、路径 reward、事后解释。LLM 不输出合规标签；解释文本一旦出现标签词即整段作废并回退模板。
        </p>
      </div>
    </div>

    <div v-if="error" class="banner bad">{{ error }}</div>

    <template v-else-if="report">
      <div class="grid cols-3">
        <div class="kpi">
          <span class="kpi-label">A2 误杀率</span>
          <span class="kpi-value">{{ pct(rejectRows.find((r) => r.condition === 'A2')?.false_kill_rate ?? 0) }}</span>
          <span class="kpi-note">合法动作被 LLM 驳回的比例</span>
        </div>
        <div class="kpi">
          <span class="kpi-label">H 误杀率</span>
          <span class="kpi-value">{{ pct(rejectRows.find((r) => r.condition === 'H')?.false_kill_rate ?? 0) }}</span>
          <span class="kpi-note">规则 + 分类器可部分抵消</span>
        </div>
        <div class="kpi">
          <span class="kpi-label">人工参考样本</span>
          <span class="kpi-value">{{ reference }}</span>
          <span class="kpi-note">误杀判定的标注基数</span>
        </div>
      </div>

      <div class="card" style="margin-top: 16px">
        <div class="card-head">
          <div class="grow">
            <div class="card-title">按条件的误杀率</div>
            <div class="card-sub">红虚线为 20% 参考线；误杀指 LLM 把合法动作判为不合法</div>
          </div>
        </div>
        <div ref="chartEl" style="height: 260px" />
      </div>

      <div class="grid cols-2" style="margin-top: 16px; align-items: start">
        <div class="card">
          <div class="card-head"><span class="card-title">成本与调用</span></div>
          <div class="card-body">
            <table class="tbl">
              <thead><tr><th>条件</th><th class="n">场景</th><th class="n">驳回</th><th class="n">误杀</th><th class="n">误杀率</th></tr></thead>
              <tbody>
                <tr v-for="row in rejectRows" :key="row.condition">
                  <td class="mono">{{ row.condition }}</td>
                  <td class="n">{{ row.scenarios }}</td>
                  <td class="n">{{ row.llm_vetoes }}</td>
                  <td class="n">{{ row.false_kills }}</td>
                  <td class="n">{{ num(row.false_kill_rate, 3) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        <div class="card">
          <div class="card-head">
            <div class="grow">
              <div class="card-title">口径说明</div>
              <div class="card-sub">来自 <code>results/llm_reject_rate.json</code> 与成本表</div>
            </div>
          </div>
          <div class="card-body small muted" style="line-height: 1.9">
            <p>· mock 模式使用确定性模拟器，主表数字均在此口径下；真实模式需 <code>LLM_API_KEY</code>。</p>
            <p>· 低置信但合法的动作进入"可疑缓冲区"，排在候选末尾，不直接丢弃。</p>
            <p>· 解释器只允许改写措辞，<code>steps / citations / signals / sufficiency</code> 一律取自本地轨迹。</p>
            <p>· 越权输出（出现 Compliant / Non-Compliant / Not-Applicable）整段作废并回退模板，原文留档审计。</p>
          </div>
        </div>
      </div>

      <div class="card" style="margin-top: 16px">
        <div class="card-head">
          <div class="grow"><div class="card-title">原始报告</div></div>
          <div class="seg">
            <button type="button" :class="{ on: tab === 'cost' }" @click="tab = 'cost'">成本表</button>
            <button type="button" :class="{ on: tab === 'real' }" @click="tab = 'real'">真实 vs mock</button>
            <button type="button" :class="{ on: tab === 'a2' }" @click="tab = 'a2'">A2 对照</button>
            <button type="button" :class="{ on: tab === 'smoke' }" @click="tab = 'smoke'">冒烟</button>
          </div>
        </div>
        <div class="card-body">
          <div class="md" v-html="renderMarkdown(md)" />
        </div>
      </div>
    </template>

    <div v-else class="sk" style="height: 320px" />
  </div>
</template>
