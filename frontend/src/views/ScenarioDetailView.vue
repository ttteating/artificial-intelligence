<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { endpoints } from '@/api/endpoints'
import type { GraphOut, ScenarioOut } from '@/api/types'
import ClausePanel from '@/components/review/ClausePanel.vue'
import { num, pct } from '@/utils/format'
import { labelCN, labelClass } from '@/utils/meta'

const route = useRoute()
const router = useRouter()
const scenario = ref<ScenarioOut | null>(null)
const graph = ref<GraphOut | null>(null)
const error = ref('')
const loading = ref(false)
const sid = computed(() => String(route.params.sid ?? ''))

async function load(): Promise<void> {
  loading.value = true
  error.value = ''
  try {
    scenario.value = await endpoints.scenario(sid.value)
    scenario.value.runs.sort((a, b) => a.condition.localeCompare(b.condition))
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    scenario.value = null
  } finally {
    loading.value = false
  }
  try {
    graph.value = await endpoints.graph(sid.value)
  } catch {
    graph.value = null
  }
}

watch(sid, load, { immediate: true })
onMounted(() => { /* 首次由 watch immediate 触发 */ })

function review(): void {
  router.push({ path: `/review/${sid.value}`, query: { condition: 'H', budget: '16', explain: '1' } })
}
</script>

<template>
  <div class="content">
    <div class="page-head">
      <div class="grow">
        <div class="page-kicker">场景详情 · {{ sid }}</div>
        <h1 class="page-title">场景与八条件对照</h1>
        <p class="page-sub" v-if="scenario">{{ scenario.text }}</p>
      </div>
      <button class="btn primary" type="button" :disabled="!scenario" @click="review">用 H 审查</button>
    </div>

    <div v-if="error" class="banner bad">{{ error }}</div>
    <div v-else-if="loading" class="sk" style="height: 320px" />

    <div v-else-if="scenario" class="split-2">
      <div class="stack" style="gap: 14px">
        <div class="card">
          <div class="card-head">
            <span class="card-title">场景原文</span>
            <span class="grow" />
            <span class="tag" :class="labelClass(scenario.gold)">{{ labelCN[scenario.gold] ?? scenario.gold }}</span>
            <span class="tag mono">{{ scenario.split }}</span>
          </div>
          <div class="card-body" style="font-size: 14px; line-height: 1.85">{{ scenario.text }}</div>
        </div>

        <div class="card">
          <div class="card-head">
            <span class="card-title">合同条款</span>
            <span class="grow" />
            <span class="tiny faint">{{ scenario.n_clauses }} 条 · {{ graph?.n_nodes ?? '—' }} 个 facet / {{ graph?.n_edges ?? '—' }} 条边</span>
          </div>
          <div class="card-body">
            <ClausePanel
              :clauses="scenario.clauses" :graph="graph" :covered="[]" :relevant="[]" :selected="null"
              @select="() => undefined"
            />
          </div>
        </div>
      </div>

      <aside class="stack" style="position: sticky; top: calc(var(--topbar) + 20px)">
        <div class="panel">
          <div class="panel-head"><span class="panel-title">八条件判定</span></div>
          <table class="tbl" style="font-size: 12px">
            <thead><tr><th>条件</th><th>判定</th><th class="n">扩展</th><th class="n">步数</th><th class="n">conf</th></tr></thead>
            <tbody>
              <tr v-for="run in scenario.runs" :key="run.condition">
                <td class="mono">{{ run.condition }}</td>
                <td>
                  <span class="tag" :class="labelClass(run.answer)">{{ labelCN[run.answer] ?? run.answer }}</span>
                  <span class="tiny" :style="{ color: run.correct ? 'var(--ok)' : 'var(--bad)' }">
                    {{ run.correct ? '✓' : '✗' }}
                  </span>
                </td>
                <td class="n">{{ run.expanded }}</td>
                <td class="n">{{ run.steps }}</td>
                <td class="n">{{ num(run.conf, 2) }}</td>
              </tr>
              <tr v-if="!scenario.runs.length"><td colspan="5" class="faint">该场景没有预计算的条件结果（自定义合同需现场跑）</td></tr>
            </tbody>
          </table>
        </div>

        <div class="panel">
          <div class="panel-head"><span class="panel-title">主表口径</span></div>
          <div class="stack" style="gap: 6px">
            <div class="row between small"><span class="muted">场景标注</span><span>{{ scenario.gold }}</span></div>
            <div class="row between small"><span class="muted">oracle 跳数</span><span>{{ scenario.oracle_hops ?? '—' }}</span></div>
            <div class="row between small"><span class="muted">H 是否正确</span>
              <span>{{ scenario.runs.find((r) => r.condition === 'H')?.correct ? '是' : '否' }}</span>
            </div>
          </div>
          <div class="divider" />
          <p class="tiny faint" style="line-height: 1.7">
            八条件结果直接读自 <code>results/rows_dev_all.csv</code>，与主表同一份产物。
          </p>
        </div>
      </aside>
    </div>
  </div>
</template>
