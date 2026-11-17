<script setup lang="ts">
import { computed } from 'vue'
import { edgeMeta } from '@/utils/meta'

const props = defineProps<{ type: string; compact?: boolean }>()
const meta = computed(() => edgeMeta[props.type] ?? { cn: props.type, color: 'var(--ink-3)' })
</script>

<template>
  <span
    class="tag mono"
    :title="`${type} · ${meta.cn}`"
    :style="{ color: meta.color, background: 'var(--surface-2)', borderColor: 'var(--line)' }"
  >
    <span :style="{ width: '6px', height: '6px', borderRadius: '50%', background: meta.color }" />
    {{ compact ? type : meta.cn }}
    <span v-if="!compact" class="faint" style="font-size: 9.5px">{{ type }}</span>
  </span>
</template>
