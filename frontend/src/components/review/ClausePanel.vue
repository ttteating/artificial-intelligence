<script setup lang="ts">
import { computed } from 'vue'
import type { GraphOut } from '@/api/types'
import RoleTag from '@/components/ui/RoleTag.vue'

const props = defineProps<{
  clauses: Record<string, string>
  graph: GraphOut | null
  covered: string[]
  relevant: string[]
  selected: string | null
}>()
const emit = defineEmits<{ (e: 'select', clause: string): void }>()

const facets = computed(() => {
  const map: Record<string, number> = {}
  for (const n of props.graph?.nodes ?? []) map[n.clause] = (map[n.clause] ?? 0) + 1
  return map
})
const roles = computed(() => {
  const out: Record<string, string[]> = {}
  for (const n of props.graph?.nodes ?? []) {
    (out[n.clause] ??= []).push(n.role)
  }
  return out
})
</script>

<template>
  <div>
    <div
      v-for="(text, clause) in clauses" :key="clause"
      class="clause-card"
      :style="selected === clause ? { borderColor: 'var(--accent)', background: 'var(--accent-50)' } : undefined"
      @click="emit('select', clause)"
    >
      <div class="clause-head">
        <span class="mono strong" style="font-size: 12px">{{ clause }}</span>
        <span v-if="relevant.includes(String(clause))" class="tag accent">强相关</span>
        <span v-if="covered.includes(String(clause))" class="tag ok">已审查</span>
        <span class="tiny faint" style="margin-left: auto">{{ facets[String(clause)] ?? 0 }} 个语义单元</span>
      </div>
      <p class="clause-text">{{ text }}</p>
      <div class="row wrap" style="margin-top: 8px">
        <RoleTag
          v-for="(role, idx) in Array.from(new Set(roles[String(clause)] ?? []))"
          :key="idx" :role="role" short
        />
      </div>
    </div>
  </div>
</template>
