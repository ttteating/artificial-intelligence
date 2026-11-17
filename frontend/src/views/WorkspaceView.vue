<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ArrowRight, FilePlus2, Library, Sparkles } from 'lucide-vue-next'
import { endpoints } from '@/api/endpoints'
import type { Overview } from '@/api/types'
import { useSystemStore } from '@/stores/system'
import { useReviewStore } from '@/stores/review'
import { ago, num, pct } from '@/utils/format'
import { labelCN, labelClass } from '@/utils/meta'

const router = useRouter()
const system = useSystemStore()
const store = useReviewStore()
const overview = ref<Overview | null>(null)
const error = ref('')

const h = computed(() => overview.value?.summary_dev?.H ?? null)
const treeSamples = computed(() => {
  const trees = overview.value?.trees ?? {}
  const meta: Record<string, { title: string; note: string }> = {
    success: { title: '成功：例外被解开', note: '沿 EXCEPTIONS 边走到例外句，证据充分后终止' },
    astar_misled: { title: 'A* 被启发式误导', note: '高 h 的分支先被展开，预算耗尽仍未覆盖关键条款' },
    llm_misjudge: { title: 'LLM 评估失误', note: 'LLM 驳回了一个合法动作，搜索改走冗余分支' },
  }
  return Object.entries(trees).map(([key, value]) => ({
    key, ...value, title: meta[key]?.title ?? key, note: meta[key]?.note ?? '',
  }))
})

onMounted(async () => {
  system.start()
  try {
    overview.value = await endpoints.overview()
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
})

function openRecent(item: { sid: string; condition: string }): void {
  router.push({ path: `/review/${item.sid}`, query: { condition: item.condition, budget: '16', explain: '1' } })
}
</script>

<template>
  <div class="content">
    <div class="page-head">
      <div class="grow">
        <div class="page-kicker">ClauseScope</div>
        <h1 class="page-title">合同合规审查工作台</h1>
        <p class="page-sub">
          给出结论，更给出为什么。一次审查会同时产出三分类判定、逐跳证据链、条款图定位与搜索过程，
          所有数字都可以点开追溯。
        </p>
      </div>
      <button class="btn primary" type="button" @click="router.push('/review/new')">
        <FilePlus2 /> 新建审查
      </button>
    </div>

    <div v-if="error" class="banner bad" style="margin-bottom: 16px">
      无法读取总览数据：{{ error }}。请确认推理服务已启动。
    </div>

    <div class="grid cols-4" style="margin-bottom: 8px">
      <div class="kpi">
        <span class="kpi-label">H · dev macro-F1</span>
        <span class="kpi-value">{{ h ? num(h.macro_f1, 3) : '—' }}</span>
        <span class="kpi-note">{{ h ? `accuracy ${pct(h.accuracy)}` : '等待指标' }}</span>
      </div>
      <div class="kpi">
        <span class="kpi-label">平均扩展节点</span>
        <span class="kpi-value">{{ h ? num(h.avg_expanded, 2) : '—' }}</span>
        <span class="kpi-note">预算 16 · 越低越省</span>
      </div>
      <div class="kpi">
        <span class="kpi-label">冗余路径占比</span>
        <span class="kpi-value">{{ h ? pct(h.redundant_ratio) : '—' }}</span>
        <span class="kpi-note">B0 为 50.1%，H 降到 36.5%</span>
      </div>
      <div class="kpi">
        <span class="kpi-label">场景规模</span>
        <span class="kpi-value">{{ system.health?.n_scenarios?.toLocaleString() ?? '—' }}</span>
        <span class="kpi-note">{{ system.health?.graph_version || '—' }} · LLM {{ system.health?.llm_mode || '—' }}</span>
      </div>
    </div>

    <div class="split-2" style="margin-top: 18px">
      <div class="stack" style="gap: 18px">
        <section class="card">
          <div class="card-head">
            <span class="card-title">开始一次审查</span>
            <span class="grow" />
            <span class="tiny faint">三条入口，同一套推理链路</span>
          </div>
          <div class="card-body grid cols-3" style="gap: 12px">
            <button class="condition-card" type="button" style="min-height: 128px" @click="router.push('/review/new')">
              <FilePlus2 :size="18" />
              <strong style="font-size: 13px">粘贴客户合同</strong>
              <span class="condition-desc">场景 + 1–10 条条款，现场建图并推理。默认不落盘。</span>
            </button>
            <button class="condition-card" type="button" style="min-height: 128px" @click="router.push('/scenarios')">
              <Library :size="18" />
              <strong style="font-size: 13px">浏览数据集场景</strong>
              <span class="condition-desc">4,700 个 ACE 场景，按标签、条款数、跳数筛选。</span>
            </button>
            <button class="condition-card" type="button" style="min-height: 128px" @click="router.push('/benchmark')">
              <Sparkles :size="18" />
              <strong style="font-size: 13px">查看评测口径</strong>
              <span class="condition-desc">八条件主表、分层结果、预算与幻觉扫描。</span>
            </button>
          </div>
        </section>

        <section class="card">
          <div class="card-head">
            <span class="card-title">最近审查</span>
            <span class="grow" />
            <span class="tiny faint">仅保存在本机浏览器</span>
          </div>
          <div v-if="store.recent.length" class="card-body" style="padding: 4px 0">
            <button
              v-for="item in store.recent" :key="`${item.sid}-${item.condition}-${item.at}`"
              class="row clickable" type="button"
              style="width: 100%; text-align: left; padding: 11px 16px; border: 0; background: transparent; border-bottom: 1px solid var(--line); cursor: pointer"
              @click="openRecent(item)"
            >
              <span class="tag mono">{{ item.condition }}</span>
              <span class="grow" style="min-width: 0">
                <span class="row" style="gap: 8px">
                  <strong style="font-size: 12.5px">{{ item.title || item.sid }}</strong>
                  <span class="tag" :class="labelClass(item.answer)">{{ labelCN[item.answer] ?? item.answer }}</span>
                  <span class="tiny faint">{{ pct(item.conf) }}</span>
                </span>
                <span class="tiny faint">{{ item.sid }} · {{ ago(item.at) }} · {{ item.elapsed }}ms</span>
              </span>
              <ArrowRight :size="14" class="faint" />
            </button>
          </div>
          <div v-else class="empty" style="padding: 32px">
            <div class="empty-title">还没有审查记录</div>
            <div class="empty-text">跑完第一次审查后，这里会保留最近 10 条的入口。</div>
          </div>
        </section>
      </div>

      <aside class="stack" style="gap: 14px">
        <div class="panel">
          <div class="panel-head"><span class="panel-title">示例案件</span></div>
          <div class="stack" style="gap: 8px">
            <button
              v-for="sample in treeSamples" :key="sample.key" type="button" class="condition-card"
              style="text-align: left" @click="openRecent(sample)"
            >
              <div class="row between">
                <span class="strong" style="font-size: 12.5px">{{ sample.title }}</span>
                <span class="tag mono">{{ sample.condition }}</span>
              </div>
              <span class="condition-desc" style="min-height: 0">{{ sample.note }}</span>
              <span class="tiny faint mono">{{ sample.sid }} · {{ sample.nodes }} 节点</span>
            </button>
          </div>
        </div>

        <div class="panel">
          <div class="panel-head"><span class="panel-title">服务</span></div>
          <div class="stack" style="gap: 6px">
            <div class="row between small"><span class="muted">状态</span>
              <span>{{ system.error ? '未连接' : system.ready ? '就绪' : '加载中' }}</span>
            </div>
            <div class="row between small"><span class="muted">场景</span><span>{{ system.health?.n_scenarios?.toLocaleString() ?? '—' }}</span></div>
            <div class="row between small"><span class="muted">图版本</span><span class="mono">{{ system.health?.graph_version || '—' }}</span></div>
            <div class="row between small"><span class="muted">LLM</span><span class="mono">{{ system.health?.llm_mode || '—' }}</span></div>
          </div>
          <div class="divider" />
          <p class="tiny faint" style="line-height: 1.7">
            本工具为辅助审查，不构成法律意见；LLM 不输出合规标签。
          </p>
        </div>
      </aside>
    </div>
  </div>
</template>
