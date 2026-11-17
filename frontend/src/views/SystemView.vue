<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { endpoints } from '@/api/endpoints'
import type { ConditionInfo, Overview } from '@/api/types'
import { num } from '@/utils/format'

const overview = ref<Overview | null>(null)
const conditions = ref<ConditionInfo[]>([])
const error = ref('')

const ga = computed(() => (overview.value?.ga_summary ?? {}) as Record<string, unknown>)
const best = computed(() => (ga.value.best_params ?? {}) as Record<string, number>)
const lambda = computed(() => overview.value?.lambda_best ?? null)
const graphStats = computed(() => (overview.value?.graph_stats ?? {}) as Record<string, unknown>)

const endpointsList = [
  ['GET', '/api/health', '就绪状态、场景数、图版本、LLM 模式'],
  ['GET', '/api/conditions', '八个条件的装配说明'],
  ['GET', '/api/overview', '总览页一次性取齐的聚合数据'],
  ['GET', '/api/summary/{split}', 'dev / test 主表指标'],
  ['GET', '/api/table/{name}', '21 张既有产物表'],
  ['GET', '/api/scenarios', '场景库筛选与分页'],
  ['POST', '/api/scenarios', '登记客户自己的合同并建图'],
  ['GET', '/api/scenario/{sid}', '场景原文、条款与八条件结果'],
  ['GET', '/api/graph/{sid}', 'facet 级条款图'],
  ['POST', '/api/search', '现场跑一次搜索（核心）'],
  ['POST', '/api/explain', '只生成证据链解释'],
  ['GET', '/api/trees', '三棵样例搜索树'],
  ['GET', '/api/llm/report', '误杀率、成本、真实 vs mock'],
]

onMounted(async () => {
  try {
    overview.value = await endpoints.overview()
    conditions.value = overview.value.conditions
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
})
</script>

<template>
  <div class="content">
    <div class="page-head">
      <div class="grow">
        <div class="page-kicker">系统</div>
        <h1 class="page-title">装配、接口与复现口径</h1>
        <p class="page-sub">前端只调用实验层，不重写任何搜索、启发、规则或融合逻辑；页面上的指标与主表产物同源。</p>
      </div>
      <a class="btn" href="/docs" target="_blank" rel="noreferrer">打开 OpenAPI 文档</a>
    </div>

    <div v-if="error" class="banner bad">{{ error }}</div>

    <template v-else-if="overview">
      <div class="grid cols-4">
        <div class="kpi">
          <span class="kpi-label">服务状态</span>
          <span class="kpi-value" style="font-size: 16px">{{ overview.health.ready ? '就绪' : overview.health.status }}</span>
          <span class="kpi-note">uptime {{ num(overview.health.uptime_sec, 0) }}s</span>
        </div>
        <div class="kpi">
          <span class="kpi-label">场景总数</span>
          <span class="kpi-value">{{ overview.health.n_scenarios.toLocaleString() }}</span>
          <span class="kpi-note">dev {{ overview.counts.dev ?? '—' }} / test {{ overview.counts.test ?? '—' }}</span>
        </div>
        <div class="kpi">
          <span class="kpi-label">图版本</span>
          <span class="kpi-value" style="font-size: 16px">{{ overview.health.graph_version }}</span>
          <span class="kpi-note">facet 级 · 7 类边</span>
        </div>
        <div class="kpi">
          <span class="kpi-label">LLM 模式</span>
          <span class="kpi-value" style="font-size: 16px">{{ overview.health.llm_mode || '—' }}</span>
          <span class="kpi-note">主表数字均为 mock 口径</span>
        </div>
      </div>

      <div class="section-title">条件装配</div>
      <div class="table-wrap">
        <table class="tbl">
          <thead>
            <tr><th>条件</th><th>内核</th><th>规则</th><th>启发函数</th><th>LLM</th><th>分类器</th><th>说明</th></tr>
          </thead>
          <tbody>
            <tr v-for="c in conditions" :key="c.id" :class="{ 'is-highlight': c.id === 'H' }">
              <td class="mono strong">{{ c.id }}</td>
              <td>{{ c.kernel }}</td>
              <td>{{ c.rules ? '硬约束' : '—' }}</td>
              <td>{{ c.h === 'zero' ? 'h=0' : c.h === 'hand' ? '手工 h' : 'GA 优化 h' }}</td>
              <td>{{ c.llm === 'off' ? '—' : c.llm }}</td>
              <td>{{ c.clf ? 'LightGBM' : '—' }}</td>
              <td class="muted" style="font-size: 12px">{{ c.description }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="grid cols-2" style="margin-top: 16px">
        <div class="card">
          <div class="card-head"><div class="card-title">GA 最优参数</div></div>
          <div class="card-body">
            <div class="grid cols-2" style="gap: 6px">
              <div v-for="(value, key) in best" :key="key" class="row between small">
                <span class="muted mono">{{ key }}</span><span class="num">{{ num(value, 4) }}</span>
              </div>
            </div>
            <div class="divider" />
            <div class="stack" style="gap: 4px">
              <div class="row between small"><span class="muted">dev macro-F1</span><span>{{ num(ga.dev_macro_f1, 4) }}</span></div>
              <div class="row between small"><span class="muted">heldout macro-F1</span><span>{{ num(ga.heldout_macro_f1, 4) }}</span></div>
              <div class="row between small"><span class="muted">适应度评估次数</span><span>{{ num(ga.n_evaluations, 0) }}</span></div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-head"><div class="card-title">融合权重（dev 网格最优）</div></div>
          <div class="card-body">
            <div v-if="lambda" class="stack" style="gap: 8px">
              <div v-for="(value, key) in lambda" :key="key" class="row between small">
                <span class="muted">λ · {{ key }}</span><span class="num">{{ num(value, 2) }}</span>
              </div>
            </div>
            <div v-else class="faint small">未找到 lambda_grid.json</div>
            <div class="divider" />
            <div class="stack" style="gap: 4px">
              <div v-for="(value, key) in graphStats" :key="key" class="row between small">
                <span class="muted mono">{{ key }}</span><span class="num">{{ typeof value === 'object' ? '—' : value }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="section-title">接口</div>
      <div class="table-wrap">
        <table class="tbl">
          <thead><tr><th style="width: 70px">方法</th><th style="width: 280px">路径</th><th>用途</th></tr></thead>
          <tbody>
            <tr v-for="[method, path, desc] in endpointsList" :key="path">
              <td><span class="tag mono" :class="method === 'POST' ? 'accent' : 'line'">{{ method }}</span></td>
              <td class="mono">{{ path }}</td>
              <td class="muted">{{ desc }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="section-title">复现纪律</div>
      <div class="card">
        <div class="card-body small muted" style="line-height: 1.9">
          <p>· <code>test</code> 只在最终配置确定后跑一次；GA 适应度只用 dev，过拟合监控用 held-out 100 条。</p>
          <p>· 前端展示的每个数字都能在 <code>results/</code> 找到产物，<code>python -m tools.check_report_numbers</code> 会逐项比对。</p>
          <p>· 自定义合同默认只在内存登记；<code>persist=true</code> 才写入 <code>data/custom/</code>。</p>
          <p>· 旧版静态证据册仍保留在 <a href="/web/" target="_blank" rel="noreferrer">/web/</a>，用于对照。</p>
        </div>
      </div>
    </template>

    <div v-else class="sk" style="height: 320px" />
  </div>
</template>
