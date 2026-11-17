<script setup lang="ts">
import { computed } from 'vue'
import type { Explanation } from '@/api/types'
import { useToast } from '@/utils/toast'

const props = defineProps<{ explanation: Explanation | null }>()
const emit = defineEmits<{ (e: 'cite', clause: string): void }>()
const { toast } = useToast()

function esc(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

const html = computed(() => {
  const text = props.explanation?.text ?? ''
  return text.split(/\n{2,}/).filter(Boolean).map((p) =>
    `<p>${esc(p).replace(/(evidence_\d+)/g, '<span class="cite" data-clause="$1">$1</span>')}</p>`,
  ).join('')
})

const modeLabel = computed(() => {
  const mode = props.explanation?.mode ?? ''
  if (mode === 'real') return '真实 LLM 解释'
  if (mode === 'real->template') return '真实调用失败 · 已回退模板'
  return '模板解释（离线可复现）'
})

function onClick(event: MouseEvent): void {
  const target = event.target as HTMLElement
  const clause = target?.dataset?.clause
  if (clause) emit('cite', clause)
}

async function copy(): Promise<void> {
  const text = props.explanation?.text ?? ''
  try {
    await navigator.clipboard.writeText(text)
    toast('已复制解释全文')
  } catch {
    toast('复制失败，请手动选择')
  }
}
</script>

<template>
  <div v-if="explanation" class="card">
    <div class="card-head">
      <div class="grow">
        <div class="card-title">推理说明</div>
        <div class="card-sub">{{ modeLabel }} · 仅描述证据链，不输出合规标签</div>
      </div>
      <span class="tag" :class="explanation.mode === 'real' ? 'accent' : 'line'">{{ explanation.mode }}</span>
      <button class="btn ghost sm no-print" type="button" @click="copy">复制</button>
    </div>
    <div class="card-body">
      <div v-if="explanation.violation" class="banner warn" style="margin-bottom: 12px">
        模型输出中出现合规标签词，已整段作废并回退模板解释；原文保留在审计字段中。
      </div>
      <div class="explain" v-html="html" @click="onClick" />
      <template v-if="explanation.citations?.length">
        <div class="divider" />
        <div class="row between" style="margin-bottom: 6px">
          <span class="panel-title">引用条款</span>
          <span class="tiny faint">{{ explanation.citations.length }} 条</span>
        </div>
        <div class="row wrap">
          <span
            v-for="c in explanation.citations" :key="c.clause"
            class="tag mono line" style="cursor: pointer; max-width: 100%"
            :title="c.excerpt" @click="emit('cite', c.clause)"
          >{{ c.clause }}</span>
        </div>
      </template>
    </div>
  </div>
  <div v-else class="card">
    <div class="empty">
      <div class="empty-title">未生成解释</div>
      <div class="empty-text">本次运行关闭了解释开关。重新运行时勾选"生成推理说明"即可看到证据链说明。</div>
    </div>
  </div>
</template>
