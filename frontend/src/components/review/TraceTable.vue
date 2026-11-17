<script setup lang="ts">
import { computed } from 'vue'
import type { TraceRow } from '@/api/types'
import { num } from '@/utils/format'

const props = defineProps<{ rows: TraceRow[] | null }>()

const isStateLevel = computed(() => (props.rows ?? []).every((r) => !r.action))
const rows = computed(() => (props.rows ?? []).slice(0, 120))
</script>

<template>
  <div>
    <div v-if="isStateLevel" class="banner info" style="margin-bottom: 10px">
      当前内核（BFS / A*）记录的是<strong>状态级轨迹</strong>：每行是一个被评估的状态，不含具体动作。
      动作序列请看左侧证据链，"哪条边走到哪"以证据链与搜索树为准。
    </div>
    <div class="table-wrap" style="max-height: 520px">
      <table class="tbl">
        <thead>
          <tr>
            <th>#</th><th class="n">depth</th><th>条款集合</th><th>label</th>
            <th class="n">conf</th><th class="n">g</th><th class="n">h</th><th class="n">f</th>
            <th class="n">reward</th><th class="n">clf</th><th>rule_mask</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, i) in rows" :key="i">
            <td class="mono">{{ i + 1 }}</td>
            <td class="n">{{ row.depth }}</td>
            <td class="mono" style="font-size: 11px">{{ row.clauses.join(' ') || '∅' }}</td>
            <td><span class="tag" :class="row.label === 'Compliant' ? 'ok' : row.label === 'Non-Compliant' ? 'bad' : 'na'">{{ row.label }}</span></td>
            <td class="n">{{ num(row.conf, 2) }}</td>
            <td class="n">{{ num(row.g, 2) }}</td>
            <td class="n">{{ num(row.h, 2) }}</td>
            <td class="n">{{ num(row.f, 2) }}</td>
            <td class="n">{{ row.llm_reward === undefined ? '—' : num(row.llm_reward, 2) }}</td>
            <td class="n">{{ row.clf_score === undefined ? '—' : num(row.clf_score, 2) }}</td>
            <td>
              <span v-for="rule in row.rule_mask ?? []" :key="rule" class="tag warn" style="height: 18px; padding: 0 5px">{{ rule }}</span>
              <span v-if="!(row.rule_mask ?? []).length" class="faint">—</span>
            </td>
          </tr>
          <tr v-if="!rows.length"><td colspan="11" class="faint">没有轨迹数据</td></tr>
        </tbody>
      </table>
    </div>
    <p v-if="(props.rows?.length ?? 0) > 120" class="tiny faint" style="margin-top: 8px">
      仅展示前 120 行，共 {{ props.rows?.length }} 行。
    </p>
  </div>
</template>
