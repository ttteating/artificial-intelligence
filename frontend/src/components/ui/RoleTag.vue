<script setup lang="ts">
import { computed } from 'vue'
import { roleMeta } from '@/utils/meta'

const props = defineProps<{ role?: string | null; short?: boolean }>()
const meta = computed(() => (props.role ? roleMeta[props.role] : null))
</script>

<template>
  <span
    v-if="role"
    class="tag"
    :style="meta ? { color: meta.color, background: 'var(--surface-2)', borderColor: 'var(--line)' } : undefined"
  >
    {{ meta?.cn ?? role }}
    <span v-if="!short" class="mono faint" style="font-size: 9.5px">{{ role }}</span>
  </span>
</template>
