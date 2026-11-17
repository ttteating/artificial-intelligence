<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts'
import { endpoints } from '@/api/endpoints'
import type { Condition, SummaryRow, TableOut } from '@/api/types'
import { num, pct } from '@/utils/format'
import { conditionMeta, labelCN } from '@/utils/meta'

const split = ref<'dev' | 'test'>('dev')
const summary = ref<Record<string, SummaryRow> | null>(null)
const budgetTable = ref<TableOut | null>(null)
const error = ref('')
const loading = ref(false)

const conditions: Condition[] = ['B0', 'B1', 'A0', 'A1', 'A2', 'M0', 'M1', 'H']
const h = computed(() => summary.value?.H ?? null)
const rows = computed(() => conditions.map((c) => ({ condition: c, ...(summary.value?.[c] ?? {}) as SummaryRow })))

const f1El = ref<HTMLDivElement | null>(null)
const budgetEl = ref<HTMLDivElement | null>(null)
let chart1: echarts.ECharts | null = null
let chart2: echarts.ECharts | null = null

const confusion = computed(() => h.value?.confusion ?? null)
const confusionLabels: Array<'Compliant' | 'Non-Compliant' | 'Not-Applicable'> =
  ['Compliant', 'Non-Compliant', 'Not-Applicable']

function maxOf(matrix: Record<string, Record<string, number>> | null): number {
  if (!matrix) return 1
  let m = 1
  Object.values(matrix).forEach((row) => Object.values(row).forEach((v) => { m = Math.max(m, v) }))
  return m
}
const maxConfusion = computed(() => maxOf(confusion.value))

function renderCharts(): void {
  if (f1El.value) {
    chart1 ??= echarts.init(f1El.value)
    chart1.setOption({
      grid: { left: 42, right: 48, top: 28, bottom: 30 },
      tooltip: { trigger: 'axis' },
      legend: { right: 0, top: 0, itemWidth: 10, itemHeight: 10, textStyle: { fontSize: 11, color: '#71717a' } },
      xAxis: {
        type: 'category', data: conditions,
        axisLine: { lineStyle: { color: '#eaeaec' } }, axisTick: { show: false },
        axisLabel: { fontSize: 11, color: '#71717a' },
      },
      yAxis: [
        { type: 'value', name: 'macro-F1', min: 0.55, max: 0.7, nameTextStyle: { fontSize: 10, color: '#a1a1aa' }, splitLine: { lineStyle: { color: '#f4f4f5' } }, axisLabel: { fontSize: 10, color: '#a1a1aa' } },
        { type: 'value', name: '扩展节点', nameTextStyle: { fontSize: 10, color: '#a1a1aa' }, splitLine: { show: false }, axisLabel: { fontSize: 10, color: '#a1a1aa' } },
      ],
      series: [
        {
          name: 'macro-F1', type: 'bar', barWidth: 18,
          itemStyle: { color: '#d4d4d8', borderRadius: [3, 3, 0, 0] },
          data: conditions.map((c) => summary.value?.[c]?.macro_f1 ?? 0),
          markLine: { silent: true, symbol: 'none', lineStyle: { color: '#5b5bd6', type: 'dashed' }, data: [{ yAxis: 0.6 }], label: { show: false } },
        },
        {
          name: '扩展节点', type: 'line', yAxisIndex: 1, smooth: true, symbolSize: 5,
          lineStyle: { color: '#5b5bd6', width: 1.6 }, itemStyle: { color: '#5b5bd6' },
          data: conditions.map((c) => summary.value?.[c]?.avg_expanded ?? 0),
        },
      ],
    })
  }

  if (budgetEl.value && budgetTable.value) {
    const data = (budgetTable.value.data as Array<Record<string, unknown>>) ?? []
    const key = Object.keys(data[0] ?? {}).find((k) => k.includes('budget')) ?? 'budget'
    const budgets = Array.from(new Set(data.map((r) => Number(r[key]))).values()).sort((a, b) => a - b)
    const pick = ['B0', 'A1', 'H'] as const
    chart2 ??= echarts.init(budgetEl.value)
    chart2.setOption({
      grid: { left: 42, right: 20, top: 28, bottom: 30 },
      tooltip: { trigger: 'axis' },
      legend: { right: 0, top: 0, itemWidth: 10, itemHeight: 10, textStyle: { fontSize: 11, color: '#71717a' } },
      xAxis: {
        type: 'category', data: budgets, name: '扩展预算',
        nameTextStyle: { fontSize: 10, color: '#a1a1aa' },
        axisLine: { lineStyle: { color: '#eaeaec' } }, axisTick: { show: false },
        axisLabel: { fontSize: 10, color: '#71717a' },
      },
      yAxis: { type: 'value', min: 0.5, max: 0.7, splitLine: { lineStyle: { color: '#f4f4f5' } }, axisLabel: { fontSize: 10, color: '#a1a1aa' } },
      series: pick.map((c, i) => ({
        name: c,
        type: 'line',
        smooth: true,
        symbolSize: 4,
        lineStyle: { width: c === 'H' ? 2 : 1.2, color: ['#a1a1aa', '#71717a', '#5b5bd6'][i] },
        itemStyle: { color: ['#a1a1aa', '#71717a', '#5b5bd6'][i] },
        data: budgets.map((b) => {
          const row = data.find((r) => Number(r[key]) === b && r.condition === c)
          return row ? Number(row.macro_f1) : null
        }),
      })),
    })
  }
}

async function load(): Promise<void> {
  loading.value = true
  error.value = ''
  try {
    const [sum, budget] = await Promise.all([
      endpoints.summary(split.value),
      budgetTable.value ? Promise.resolve(budgetTable.value) : endpoints.table('budget_sweep'),
    ])
    summary.value = sum
    budgetTable.value = budget
    await nextTick()
    renderCharts()
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}

function onResize(): void {
  chart1?.resize()
  chart2?.resize()
}

onMounted(async () => {
  await load()
  window.addEventListener('resize', onResize)
})
watch(split, load)
onBeforeUnmount(() => {
  window.removeEventListener('resize', onResize)
  chart1?.dispose()
  chart2?.dispose()
})
</script>

<template>
  <div class="content">
    <div class="page-head">
      <div class="grow">
        <div class="page-kicker">评测</div>
        <h1 class="page-title">八条件主表与收益边界</h1>
        <p class="page-sub">
          所有数字来自 <code>results/</code> 既有产物，与 <code>tools.check_report_numbers</code> 校验的是同一份。
          test 只在最终配置确定后运行一次。
        </p>
      </div>
      <div class="seg">
        <button type="button" :class="{ on: split === 'dev' }" @click="split = 'dev'">dev 600</button>
        <button type="button" :class="{ on: split === 'test' }" @click="split = 'test'">test 1,100</button>
      </div>
    </div>

    <div v-if="error" class="banner bad" style="margin-bottom: 12px">{{ error }}</div>
    <div v-if="loading && !summary" class="sk" style="height: 360px" />

    <template v-else-if="h">
      <div class="grid cols-4">
        <div class="kpi"><span class="kpi-label">macro-F1</span><span class="kpi-value">{{ num(h.macro_f1, 3) }}</span><span class="kpi-note">B0 0.641</span></div>
        <div class="kpi"><span class="kpi-label">accuracy</span><span class="kpi-value">{{ pct(h.accuracy) }}</span><span class="kpi-note">{{ h.n }} 条场景</span></div>
        <div class="kpi"><span class="kpi-label">平均扩展节点</span><span class="kpi-value">{{ num(h.avg_expanded, 2) }}</span><span class="kpi-note">M0 需 9.34</span></div>
        <div class="kpi"><span class="kpi-label">冗余路径占比</span><span class="kpi-value">{{ pct(h.redundant_ratio) }}</span><span class="kpi-note">越低越好</span></div>
      </div>

      <div class="grid cols-2" style="margin-top: 16px">
        <div class="card">
          <div class="card-head">
            <div class="grow">
              <div class="card-title">F1 与搜索成本</div>
              <div class="card-sub">柱：macro-F1；线：平均扩展节点</div>
            </div>
          </div>
          <div ref="f1El" style="height: 260px" />
        </div>
        <div class="card">
          <div class="card-head">
            <div class="grow">
              <div class="card-title">预算扫描</div>
              <div class="card-sub">扩展预算 vs macro-F1（B0 / A1 / H）</div>
            </div>
          </div>
          <div ref="budgetEl" style="height: 260px" />
        </div>
      </div>

      <div class="section-title">主表 <span class="muted small">H 行高亮</span></div>
      <div class="table-wrap">
        <table class="tbl">
          <thead>
            <tr>
              <th>条件</th><th class="n">macro-F1</th><th class="n">accuracy</th><th class="n">扩展节点</th>
              <th class="n">推理步数</th><th class="n">冗余占比</th><th class="n">LLM 调用</th><th class="n">规则剪枝</th>
              <th class="n">目标达成率</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in rows" :key="row.condition" :class="{ 'is-highlight': row.condition === 'H' }">
              <td>
                <span class="mono strong">{{ row.condition }}</span>
                <span class="muted" style="margin-left: 6px; font-size: 11px">{{ conditionMeta[row.condition]?.name }}</span>
              </td>
              <td class="n">{{ num(row.macro_f1, 4) }}</td>
              <td class="n">{{ num(row.accuracy, 4) }}</td>
              <td class="n">{{ num(row.avg_expanded, 2) }}</td>
              <td class="n">{{ num(row.avg_steps, 2) }}</td>
              <td class="n">{{ num(row.redundant_ratio, 3) }}</td>
              <td class="n">{{ num(row.avg_llm_calls, 1) }}</td>
              <td class="n">{{ num(row.avg_pruned, 1) }}</td>
              <td class="n">{{ num(row.goal_rate, 2) }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="grid cols-2" style="margin-top: 16px">
        <div class="card">
          <div class="card-head">
            <div class="grow">
              <div class="card-title">混淆矩阵 · H</div>
              <div class="card-sub">行：标注；列：预测</div>
            </div>
          </div>
          <div class="card-body">
            <table class="tbl" v-if="confusion">
              <thead>
                <tr><th></th><th v-for="l in confusionLabels" :key="l">{{ labelCN[l] }}</th></tr>
              </thead>
              <tbody>
                <tr v-for="row in confusionLabels" :key="row">
                  <th style="position: static; background: transparent; padding: 8px 10px">{{ labelCN[row] }}</th>
                  <td
                    v-for="col in confusionLabels" :key="col" class="n"
                    :style="{
                      background: `rgba(91,91,214,${(confusion[row][col] / maxConfusion) * 0.22})`,
                      fontWeight: row === col ? 500 : 400,
                    }"
                  >{{ confusion[row][col] }}</td>
                </tr>
              </tbody>
            </table>
            <p class="tiny faint" style="margin-top: 8px">
              Non-Compliant 召回 {{ pct(h['recall_Non-Compliant']) }}，主要被误判为 Compliant；
              这是当前失败分析的重点（见 <code>results/failure_cases.md</code>）。
            </p>
          </div>
        </div>

        <div class="card">
          <div class="card-head">
            <div class="grow">
              <div class="card-title">分类召回</div>
              <div class="card-sub">H 条件下三类各自的召回</div>
            </div>
          </div>
          <div class="card-body stack" style="gap: 14px">
            <div v-for="item in [
              { label: 'Compliant', value: h.recall_Compliant, cls: 'ok' },
              { label: 'Non-Compliant', value: h['recall_Non-Compliant'], cls: 'bad' },
              { label: 'Not-Applicable', value: h['recall_Not-Applicable'], cls: '' },
            ]" :key="item.label">
              <div class="row between small">
                <span class="muted">{{ labelCN[item.label] }}</span>
                <span class="num">{{ pct(item.value) }}</span>
              </div>
              <div class="bar" :class="item.cls"><i :style="{ width: pct(item.value) }" /></div>
            </div>
            <div class="divider" />
            <p class="tiny faint" style="line-height: 1.7">
              4+ 跳场景的 F1 只有约 0.30：证据链更长时，启发式收益与判断稳定性都会下降。
              这不是隐藏项，结果页会对高不确定场景单独标注。
            </p>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>
