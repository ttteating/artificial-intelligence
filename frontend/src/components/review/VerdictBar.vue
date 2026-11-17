<script setup lang="ts">
import { computed } from 'vue'
import type { SearchResponse } from '@/api/types'
import ConfidenceRing from '@/components/ui/ConfidenceRing.vue'
import { labelCN, labelClass } from '@/utils/meta'
import { ms, num, pct } from '@/utils/format'

const props = defineProps<{ result: SearchResponse; budget: number }>()

const cls = computed(() => labelClass(props.result.answer))
const suf = computed(() => props.result.explanation?.sufficiency ?? null)
const checks = computed(() => {
  const s = suf.value
  if (!s) return []
  return [
    { label: '条款覆盖', ok: s.coverage_ok, detail: `${s.covered_clauses.length}/${s.relevant_clauses.length}` },
    {
      label: s.exception_required ? '例外已解' : '无需例外',
      ok: s.exception_ok,
      detail: s.exception_required ? (s.exception_ok ? '已解开' : '未解开') : '场景无违约线索',
    },
    { label: '置信达标', ok: s.confidence_ok, detail: `${num(s.confidence, 2)} / ${num(s.tau_conf, 2)}` },
  ]
})
const budgetPct = computed(() => Math.min(100, (props.result.expanded_nodes / Math.max(1, props.budget)) * 100))
const goldMismatch = computed(() => {
  const gold = props.result.meta.gold
  return gold && gold !== 'Unlabeled' && gold !== props.result.answer
})
</script>

<template>
  <div class="verdict">
    <div class="verdict-main">
      <ConfidenceRing :value="result.confidence" :label="result.answer" />
      <div>
        <div class="row" style="gap: 10px">
          <span class="verdict-label" :class="cls">{{ labelCN[result.answer] ?? result.answer }}</span>
          <span v-if="!result.reached_goal" class="tag warn">未达目标测试</span>
          <span v-if="goldMismatch" class="tag line" title="数据集标注与模型输出不同，这不代表系统故障">与标注不一致</span>
        </div>
        <div class="verdict-sub">
          终局置信度 {{ pct(result.confidence) }} · {{ result.condition }} · {{ result.llm_mode === 'off' ? '无 LLM' : `LLM ${result.llm_mode}` }}
        </div>
      </div>
    </div>

    <div class="checks" v-if="checks.length">
      <span
        v-for="c in checks" :key="c.label" class="check"
        :class="c.ok ? 'ok' : 'fail'" :title="c.detail"
      >
        <span class="dot" />{{ c.label }}<span class="faint" style="margin-left: 2px">{{ c.detail }}</span>
      </span>
    </div>

    <div class="verdict-meta">
      <div class="vm">
        <span class="vm-label">扩展预算</span>
        <span class="vm-value"><b>{{ result.expanded_nodes }}</b> / {{ budget }}</span>
        <div class="budget"><i :style="{ width: `${budgetPct}%` }" /></div>
      </div>
      <div class="vm">
        <span class="vm-label">证据链</span>
        <span class="vm-value"><b>{{ result.steps }}</b> 步</span>
      </div>
      <div class="vm">
        <span class="vm-label">耗时</span>
        <span class="vm-value">{{ ms(result.elapsed_ms) }}</span>
      </div>
      <div class="vm">
        <span class="vm-label">LLM 调用</span>
        <span class="vm-value">{{ result.llm_calls }}</span>
      </div>
      <div class="vm">
        <span class="vm-label">规则剪枝</span>
        <span class="vm-value">{{ result.pruned_by_rule }}</span>
      </div>
    </div>
  </div>
</template>
