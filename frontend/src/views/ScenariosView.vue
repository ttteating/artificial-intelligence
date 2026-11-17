<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { endpoints } from '@/api/endpoints'
import type { Condition, ScenarioListItem } from '@/api/types'
import { num } from '@/utils/format'
import { labelCN, labelClass } from '@/utils/meta'

const router = useRouter()
const items = ref<ScenarioListItem[]>([])
const total = ref(0)
const loading = ref(false)
const error = ref('')

const split = ref<'dev' | 'test'>('dev')
const gold = ref('')
const nClauses = ref('')
const hops = ref('')
const q = ref('')
const page = ref(0)
const limit = 20
const conditions: Condition[] = ['B0', 'B1', 'A0', 'A1', 'A2', 'M0', 'M1', 'H']

const pages = computed(() => Math.max(1, Math.ceil(total.value / limit)))
const hopRange = computed(() => {
  if (hops.value === '1-2') return { min_hops: 1, max_hops: 2 }
  if (hops.value === '3') return { min_hops: 3, max_hops: 3 }
  if (hops.value === '4+') return { min_hops: 4, max_hops: 8 }
  return {}
})

let debounce: number | undefined
async function load(): Promise<void> {
  loading.value = true
  error.value = ''
  try {
    const res = await endpoints.scenarios({
      split: split.value,
      q: q.value || undefined,
      n_clauses: nClauses.value ? Number(nClauses.value) : undefined,
      ...hopRange.value,
      limit,
      offset: page.value * limit,
    })
    items.value = res.items
    total.value = res.total
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    items.value = []
  } finally {
    loading.value = false
  }
}

function reload(): void {
  page.value = 0
  window.clearTimeout(debounce)
  debounce = window.setTimeout(load, 220)
}

watch([split, nClauses, hops], reload)
watch(q, reload)
watch(page, load)
onMounted(load)

function open(sid: string): void {
  router.push({ path: `/review/${sid}`, query: { condition: 'H', budget: '16', explain: '1' } })
}
function dotClass(item: ScenarioListItem, c: Condition): string {
  const run = item.runs[c]
  if (!run) return 'na'
  return run.correct ? 'ok' : 'bad'
}
function dotTitle(item: ScenarioListItem, c: Condition): string {
  const run = item.runs[c]
  if (!run) return `${c}: 无结果`
  return `${c}: ${labelCN[run.answer] ?? run.answer} · 置信 ${run.conf.toFixed(2)} · 扩展 ${run.expanded}`
}
</script>

<template>
  <div class="content">
    <div class="page-head">
      <div class="grow">
        <div class="page-kicker">场景库</div>
        <h1 class="page-title">4,700 个合规场景</h1>
        <p class="page-sub">
          ACE 数据集按标签、条款数与 oracle 跳数筛选。每行的八个点对应八个条件的判定结果，点击行直接进入审查结果。
        </p>
      </div>
      <button class="btn primary" type="button" @click="router.push('/review/new')">新建审查</button>
    </div>

    <div class="card" style="margin-bottom: 14px">
      <div class="card-body row wrap" style="gap: 10px">
        <div class="seg">
          <button type="button" :class="{ on: split === 'dev' }" @click="split = 'dev'">dev 600</button>
          <button type="button" :class="{ on: split === 'test' }" @click="split = 'test'">test 1,100</button>
        </div>
        <input v-model="q" class="input" style="width: 200px" placeholder="按 sid 搜索" />
        <select v-model="nClauses" class="select" style="width: 130px">
          <option value="">全部条款数</option>
          <option v-for="n in [1, 2, 3, 4]" :key="n" :value="String(n)">{{ n }} 条条款</option>
        </select>
        <select v-model="hops" class="select" style="width: 130px">
          <option value="">全部跳数</option>
          <option value="1-2">1–2 跳</option>
          <option value="3">3 跳</option>
          <option value="4+">4+ 跳</option>
        </select>
        <span class="grow" />
        <span class="tiny faint">{{ loading ? '加载中…' : `共 ${total.toLocaleString()} 条` }}</span>
      </div>
    </div>

    <div v-if="error" class="banner bad" style="margin-bottom: 12px">{{ error }}</div>

    <div class="table-wrap">
      <table class="tbl">
        <thead>
          <tr>
            <th style="width: 110px">sid</th>
            <th style="width: 90px">标注</th>
            <th class="n" style="width: 70px">条款</th>
            <th class="n" style="width: 70px">跳数</th>
            <th style="width: 190px">八条件结果</th>
            <th style="width: 110px">H 置信度</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="item in items" :key="item.sid" class="clickable" @click="open(item.sid)">
            <td class="mono">{{ item.sid }}</td>
            <td><span class="tag" :class="labelClass(item.gold)">{{ labelCN[item.gold] ?? item.gold }}</span></td>
            <td class="n">{{ item.n_clauses }}</td>
            <td class="n">{{ item.oracle_hops ?? '—' }}</td>
            <td>
              <span class="dot-grid" style="gap: 5px">
                <i
                  v-for="c in conditions" :key="c" :class="dotClass(item, c)"
                  :title="dotTitle(item, c)" style="width: 8px; height: 8px"
                />
              </span>
            </td>
            <td class="n">{{ item.runs.H ? num(item.runs.H.conf, 2) : '—' }}</td>
            <td class="n"><span class="tiny faint">审查 →</span></td>
          </tr>
          <tr v-if="!items.length">
            <td colspan="7" class="faint">{{ loading ? '加载中…' : '没有符合条件的场景' }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="row between" style="margin-top: 12px">
      <span class="tiny faint">第 {{ page + 1 }} / {{ pages }} 页</span>
      <div class="row">
        <button class="btn sm" type="button" :disabled="page === 0" @click="page--">上一页</button>
        <button class="btn sm" type="button" :disabled="page + 1 >= pages" @click="page++">下一页</button>
      </div>
    </div>
  </div>
</template>
