<script setup lang="ts">
import type { ChainStep } from '@/utils/review'
import EdgeChip from '@/components/ui/EdgeChip.vue'
import RoleTag from '@/components/ui/RoleTag.vue'
import { num, shortNid } from '@/utils/format'

defineProps<{ steps: ChainStep[]; active: number | null }>()
const emit = defineEmits<{ (e: 'select', index: number): void }>()
</script>

<template>
  <div class="chain">
    <button
      v-for="step in steps" :key="step.i" type="button"
      class="step" :class="{ active: active === step.i }"
      @click="emit('select', step.i)"
    >
      <span class="step-idx">{{ step.i }}</span>
      <div class="step-body">
        <div class="step-edge">
          <EdgeChip :type="step.edge" />
          <RoleTag :role="step.role" short />
        </div>
        <div class="step-flow">
          {{ shortNid(step.from) }} <span class="faint">→</span> {{ shortNid(step.to) }}
          <span v-if="step.clause" class="faint">· {{ step.clause }}</span>
        </div>
        <p v-if="step.excerpt" class="step-excerpt clamp-3" :title="step.excerpt">{{ step.excerpt }}</p>
        <div class="step-meta">
          <span v-if="step.g !== null">g <b>{{ num(step.g, 2) }}</b></span>
          <span v-if="step.h !== null">h <b>{{ num(step.h, 2) }}</b></span>
          <span v-if="step.f !== null">f <b>{{ num(step.f, 2) }}</b></span>
          <span v-if="step.llm_reward !== null">reward <b>{{ num(step.llm_reward, 2) }}</b></span>
          <span v-if="step.clf_score !== null">clf <b>{{ num(step.clf_score, 2) }}</b></span>
          <span v-for="rule in step.rule_mask" :key="rule" class="tag warn" style="height: 18px; padding: 0 5px">{{ rule }}</span>
        </div>
      </div>
    </button>
  </div>
</template>
