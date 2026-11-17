<script setup lang="ts">
import { computed } from 'vue'
import type { SearchResponse } from '@/api/types'
import { num, pct } from '@/utils/format'
import { ruleHits } from '@/utils/review'
import { ruleMeta } from '@/utils/meta'

const props = defineProps<{ result: SearchResponse; budget: number }>()

const suf = computed(() => props.result.explanation?.sufficiency ?? null)
const signals = computed(() => props.result.explanation?.signals ?? {})
const hits = computed(() => ruleHits(props.result))
const lastRow = computed(() => (props.result.trace ?? [])[props.result.trace!.length - 1] ?? null)

const reward = computed(() => Number(signals.value.llm_reward ?? lastRow.value?.llm_reward ?? 0))
const clf = computed(() => {
  const v = signals.value.clf_score ?? lastRow.value?.clf_score
  return typeof v === 'number' ? v : null
})
const risk = computed(() => {
  const v = (signals.value as Record<string, unknown>).risk
  return typeof v === 'number' ? v : null
})

const coverage = computed(() => {
  const s = suf.value
  if (!s) return []
  return s.relevant_clauses.map((c) => ({ clause: c, covered: s.covered_clauses.includes(c) }))
})
</script>

<template>
  <div class="panel">
    <div class="panel-head">
      <span class="panel-title">判定依据</span>
      <span class="tiny faint">{{ result.condition }} · 预算 {{ budget }}</span>
    </div>

    <div v-if="suf" class="stack" style="gap: 2px">
      <div class="metric-row">
        <span class="metric-label">条款覆盖</span>
        <span class="metric-value">
          <span :style="{ color: suf.coverage_ok ? 'var(--ok)' : 'var(--bad)' }">
            {{ suf.coverage_ok ? '达标' : '未达标' }}
          </span>
          <span class="faint"> {{ suf.covered_clauses.length }}/{{ suf.relevant_clauses.length }}</span>
        </span>
      </div>
      <div class="metric-row">
        <span class="metric-label">例外处理</span>
        <span class="metric-value">
          <span :style="{ color: suf.exception_ok ? 'var(--ok)' : 'var(--bad)' }">
            {{ suf.exception_required ? (suf.exception_ok ? '已解开' : '未解开') : '无需解开' }}
          </span>
        </span>
      </div>
      <div class="metric-row">
        <span class="metric-label">置信度 / 门槛</span>
        <span class="metric-value">{{ num(suf.confidence, 2) }} / {{ num(suf.tau_conf, 2) }}</span>
      </div>
      <div class="bar" :class="suf.confidence_ok ? 'ok' : 'bad'">
        <i :style="{ width: `${Math.min(100, (suf.confidence / Math.max(0.01, suf.tau_conf)) * 100)}%` }" />
      </div>
    </div>

    <div class="divider" />

    <div class="panel-title" style="margin-bottom: 6px">融合信号</div>
    <div class="metric-row">
      <span class="metric-label" title="LLM 对整条路径的语义相关度打分，不输出标签">LLM reward</span>
      <span class="metric-value">{{ num(reward, 2) }}</span>
    </div>
    <div class="bar accent"><i :style="{ width: `${Math.min(100, reward * 100)}%` }" /></div>
    <div class="metric-row" style="margin-top: 6px">
      <span class="metric-label" title="路径价值分类器判为正确路径的概率">分类器 P(正确)</span>
      <span class="metric-value">{{ clf === null ? '—' : num(clf, 2) }}</span>
    </div>
    <div class="bar"><i :style="{ width: `${Math.min(100, (clf ?? 0) * 100)}%` }" /></div>
    <div class="metric-row" style="margin-top: 6px">
      <span class="metric-label" title="规则引擎给出的分支风险，越高越可能无效">规则风险</span>
      <span class="metric-value">{{ risk === null ? '—' : num(risk, 2) }}</span>
    </div>
    <div class="bar bad"><i :style="{ width: `${Math.min(100, (risk ?? 0) * 100)}%` }" /></div>

    <div class="divider" />

    <div class="panel-title" style="margin-bottom: 6px">规则命中</div>
    <div v-if="hits.length" class="row wrap">
      <span v-for="[rule, count] in hits" :key="rule" class="tag warn" :title="ruleMeta[rule] ?? rule">
        {{ rule }} × {{ count }} <span class="faint">{{ ruleMeta[rule] }}</span>
      </span>
    </div>
    <div v-else class="tiny faint">本次未命中硬约束</div>

    <div class="divider" />

    <div class="panel-title" style="margin-bottom: 6px">证据覆盖</div>
    <div class="row wrap">
      <span
        v-for="item in coverage" :key="item.clause"
        class="tag mono" :class="item.covered ? 'ok' : 'bad'"
        :title="item.covered ? '已审查' : '强相关但未覆盖'"
      >{{ item.clause }}</span>
    </div>
    <div v-if="!coverage.length" class="tiny faint">解释未返回覆盖清单（关闭了解释）</div>

    <div class="divider" />

    <div class="row between small">
      <span class="muted">LLM 调用 / 驳回</span>
      <span class="num">{{ result.llm_calls }} / {{ result.llm_rejected }}</span>
    </div>
    <div class="row between small" style="margin-top: 4px">
      <span class="muted">规则剪枝</span>
      <span class="num">{{ result.pruned_by_rule }}</span>
    </div>
    <div class="row between small" style="margin-top: 4px">
      <span class="muted">目标测试</span>
      <span>{{ result.reached_goal ? '满足' : '未满足' }}</span>
    </div>
    <div class="row between small" style="margin-top: 4px">
      <span class="muted">置信度</span>
      <span>{{ pct(result.confidence) }}</span>
    </div>
  </div>
</template>
