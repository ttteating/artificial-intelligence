<script setup lang="ts">
import { CircleSlash } from 'lucide-vue-next'

defineProps<{ title?: string; text?: string }>()
</script>

<template>
  <div class="empty">
    <div class="empty-icon"><CircleSlash :size="16" /></div>
    <div class="empty-title">{{ title ?? '暂无内容' }}</div>
    <div class="empty-text">{{ text ?? '当前筛选条件下没有可展示的数据。' }}</div>
    <slot />
  </div>
</template>
