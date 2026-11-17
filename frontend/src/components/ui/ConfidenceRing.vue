<script setup lang="ts">
import { computed } from 'vue'
import { labelClass } from '@/utils/meta'

const props = withDefaults(defineProps<{
  value: number
  label?: string | null
  size?: number
}>(), { size: 62, label: null })

const r = 26
const c = 2 * Math.PI * r
const ratio = computed(() => Math.max(0, Math.min(1, props.value || 0)))
const dash = computed(() => `${(c * ratio.value).toFixed(1)} ${c.toFixed(1)}`)
const color = computed(() => {
  const cls = labelClass(props.label)
  return cls === 'ok' ? 'var(--ok)' : cls === 'bad' ? 'var(--bad)' : 'var(--na)'
})
</script>

<template>
  <div class="ring" :style="{ width: `${size}px`, height: `${size}px`, flexBasis: `${size}px` }">
    <svg :width="size" :height="size" viewBox="0 0 62 62" aria-hidden="true">
      <circle cx="31" cy="31" :r="r" fill="none" stroke="var(--surface-3)" stroke-width="3" />
      <circle
        cx="31" cy="31" :r="r" fill="none" :stroke="color" stroke-width="3"
        :stroke-dasharray="dash" stroke-linecap="round"
        style="transform: rotate(-90deg); transform-origin: 31px 31px; transition: stroke-dasharray 500ms ease"
      />
    </svg>
    <div class="ring-text">{{ Math.round(ratio * 100) }}<span class="tiny faint">%</span></div>
  </div>
</template>
