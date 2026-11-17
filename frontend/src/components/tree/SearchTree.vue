<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import { hierarchy, tree, type HierarchyPointNode } from 'd3-hierarchy'
import type { TreeNode } from '@/api/types'
import { flagMeta, roleMeta, statusMeta } from '@/utils/meta'

interface Datum { node: TreeNode; children: Datum[] }
interface Placed { tid: number; node: TreeNode; x: number; y: number }

const props = defineProps<{
  nodes: TreeNode[] | null
  selectedTid: number | null
}>()
const emit = defineEmits<{ (e: 'select', tid: number): void }>()

const playing = ref(false)
const cursor = ref(Number.POSITIVE_INFINITY)
let timer: number | undefined

const maxTid = computed(() => Math.max(0, ...(props.nodes ?? []).map((n) => n.tid)))

const layout = computed(() => {
  const list = props.nodes ?? []
  if (!list.length) return { placed: [] as Placed[], links: [] as Array<{ sx: number; sy: number; tx: number; ty: number; on: boolean }>, width: 320, height: 240 }

  const byId = new Map(list.map((n) => [n.tid, n]))
  const kids = new Map<number, TreeNode[]>()
  for (const n of list) {
    if (n.parent === null) continue
    const arr = kids.get(n.parent) ?? []
    arr.push(n)
    kids.set(n.parent, arr)
  }
  const rootNode = list.find((n) => n.parent === null) ?? list[0]
  const build = (n: TreeNode): Datum => ({
    node: n,
    children: (kids.get(n.tid) ?? []).map(build),
  })
  const root = hierarchy<Datum>(build(rootNode))
  tree<Datum>().nodeSize([46, 88])(root)

  const placed: Placed[] = (root.descendants() as unknown as HierarchyPointNode<Datum>[]).map((d) => ({
    tid: d.data.node.tid, node: d.data.node, x: d.x, y: d.y,
  }))
  const minX = Math.min(...placed.map((p) => p.x))
  const maxX = Math.max(...placed.map((p) => p.x))
  const maxY = Math.max(...placed.map((p) => p.y))
  const offset = 44 - minX

  const normalized = placed.map((p) => ({ ...p, x: p.x + offset }))
  const pos = new Map(normalized.map((p) => [p.tid, p]))
  const links = root.links().map((l) => {
    const s = pos.get(l.source.data.node.tid)!
    const t = pos.get(l.target.data.node.tid)!
    return { sx: s.x, sy: s.y, tx: t.x, ty: t.y, on: l.target.data.node.flags.includes('success_path') }
  })
  return { placed: normalized, links, width: Math.max(360, maxX - minX + 200), height: maxY + 96 }
})

function fill(n: TreeNode): string {
  if (n.flags.includes('success_path')) return 'var(--ok)'
  if (n.flags.includes('llm_rejected')) return 'var(--bad)'
  if (n.flags.includes('llm_misjudge')) return 'var(--warn)'
  return statusMeta[n.status]?.color ?? 'var(--ink-4)'
}
function stroke(n: TreeNode): string {
  if (props.selectedTid === n.tid) return 'var(--accent)'
  if (n.flags.includes('success_path')) return 'var(--ok)'
  if (n.flags.includes('llm_misjudge')) return 'var(--warn)'
  return 'var(--line-strong)'
}
function linkPath(l: { sx: number; sy: number; tx: number; ty: number }): string {
  const my = (l.sy + l.ty) / 2
  return `M${l.sx},${l.sy} C${l.sx},${my} ${l.tx},${my} ${l.tx},${l.ty}`
}
function isVisible(tid: number): boolean {
  return tid <= cursor.value
}

function stop(): void {
  playing.value = false
  window.clearInterval(timer)
  cursor.value = Number.POSITIVE_INFINITY
}
function replay(): void {
  window.clearInterval(timer)
  cursor.value = 0
  playing.value = true
  timer = window.setInterval(() => {
    cursor.value += 1
    if (cursor.value >= maxTid.value) {
      window.clearInterval(timer)
      playing.value = false
    }
  }, 420)
}

watch(() => props.nodes, () => { stop() }, { deep: false })
onUnmounted(() => window.clearInterval(timer))
</script>

<template>
  <div>
    <div class="tree-toolbar no-print">
      <button class="btn sm" type="button" @click="playing ? stop() : replay()">
        {{ playing ? '停止回放' : '回放扩展过程' }}
      </button>
      <span class="tiny faint">{{ nodes?.length ?? 0 }} 节点 · 按 tid 顺序</span>
      <span class="grow" />
      <span v-for="(m, key) in statusMeta" :key="key" class="legend-item">
        <span class="status-dot" :style="{ background: m.color }" />{{ m.cn }}
      </span>
    </div>
    <div class="tree-shell">
      <svg
        v-if="layout.placed.length" class="tree-svg"
        :width="layout.width" :height="layout.height" :viewBox="`0 0 ${layout.width} ${layout.height}`"
        role="img" aria-label="搜索树"
      >
        <g>
          <path
            v-for="(l, i) in layout.links" :key="`l-${i}`"
            class="tree-edge" :class="{ on: l.on }" :d="linkPath(l)"
          />
          <g
            v-for="p in layout.placed" :key="p.tid" class="tree-node"
            :transform="`translate(${p.x},${p.y})`"
            :opacity="isVisible(p.tid) ? 1 : 0.18"
            @click="emit('select', p.tid)"
          >
            <circle
              :r="selectedTid === p.tid ? 9 : 7" :fill="fill(p.node)" :stroke="stroke(p.node)"
              :stroke-width="selectedTid === p.tid ? 3 : 1.4"
            />
            <text class="tree-label" x="12" y="3.5">
              {{ roleMeta[p.node.role ?? '']?.cn ?? (p.node.role ?? 'ROOT') }}
              <tspan class="faint"> f{{ p.node.f.toFixed(1) }}</tspan>
            </text>
            <text v-if="p.node.flags.length" class="tree-flag" x="12" y="16">
              {{ p.node.flags.map((f) => flagMeta[f]?.cn ?? f).join(' · ') }}
            </text>
            <title>
              #{{ p.tid }} · {{ p.node.status }} · g {{ p.node.g }} / h {{ p.node.h }} / f {{ p.node.f }}
              <template v-if="p.node.conf !== null"> · conf {{ p.node.conf }}</template>
            </title>
          </g>
        </g>
      </svg>
      <div v-else class="empty">
        <div class="empty-title">没有搜索树</div>
        <div class="empty-text">本次运行未返回树结构，可能关闭了 include_tree。</div>
      </div>
    </div>
    <p class="tiny faint" style="margin-top: 8px">
      节点颜色表示状态（已扩展 / 目标 / 被剪），描边表示标记（成功路径 / LLM 驳回 / LLM 误判）；数字为 f 值。
    </p>
  </div>
</template>
