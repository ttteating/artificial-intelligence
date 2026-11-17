<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { VueFlow, useVueFlow, type Edge, type Node } from '@vue-flow/core'
import { Background } from '@vue-flow/background'
import { Controls } from '@vue-flow/controls'
import { MiniMap } from '@vue-flow/minimap'
import type { Action, GraphOut } from '@/api/types'
import { edgeMeta, roleMeta } from '@/utils/meta'

const props = defineProps<{
  graph: GraphOut | null
  path: Action[]
  selectedNid: string | null
}>()
const emit = defineEmits<{ (e: 'select', nid: string): void }>()

const nodes = ref<Node[]>([])
const edges = ref<Edge[]>([])
const onlyPath = ref(false)
const { fitView } = useVueFlow()

const pathEdgeKeys = computed(() => new Set(
  (props.path ?? []).filter((a) => a.src !== 'ROOT').map((a) => `${a.src}|${a.edge_type}|${a.dst}`),
))
const pathNodes = computed(() => new Set((props.path ?? []).map((a) => a.dst)))

function build(): void {
  const g = props.graph
  if (!g) { nodes.value = []; edges.value = []; return }

  const clauses = Array.from(new Set(g.nodes.map((n) => n.clause)))
  const colOf = new Map(clauses.map((c, i) => [c, i]))
  const perClause = new Map<string, number>()
  const next: Node[] = []

  for (const clause of clauses) {
    next.push({
      id: `clause-${clause}`,
      type: 'clause-label',
      position: { x: 12 + (colOf.get(clause) ?? 0) * 268, y: 6 },
      data: { label: clause },
      draggable: false,
      selectable: false,
    })
  }

  for (const n of g.nodes) {
    const col = colOf.get(n.clause) ?? 0
    const row = perClause.get(n.clause) ?? 0
    perClause.set(n.clause, row + 1)
    const onPath = pathNodes.value.has(n.nid)
    next.push({
      id: n.nid,
      type: 'facet',
      position: { x: 12 + col * 268, y: 42 + row * 96 },
      data: { node: n, onPath },
      draggable: true,
    })
  }
  nodes.value = next

  const nextEdges: unknown[] = g.edges.map((e, i) => {
    const meta = edgeMeta[e.etype] ?? { cn: e.etype, color: 'var(--ink-4)' }
    const onPath = pathEdgeKeys.value.has(`${e.src}|${e.etype}|${e.dst}`)
    return {
      id: `e-${i}-${e.src}-${e.dst}`,
      source: e.src,
      target: e.dst,
      type: 'smoothstep',
      label: onPath ? meta.cn : undefined,
      labelStyle: { fill: meta.color, fontSize: '10px' },
      labelBgStyle: { fill: '#ffffff', fillOpacity: 0.9 },
      animated: onPath,
      style: {
        stroke: meta.color,
        strokeWidth: onPath ? 2 : 1.1,
        strokeDasharray: onPath ? undefined : meta.dash,
        opacity: onPath ? 1 : 0.5,
      },
      markerEnd: 'url(#cs-arrow)',
    }
  })
  edges.value = nextEdges as Edge[]

  window.setTimeout(() => { try { fitView({ padding: 0.18, duration: 300 }) } catch { /* 容器未就绪 */ } }, 60)
}

watch(() => props.graph, build, { immediate: true })
watch([() => props.path, () => props.selectedNid], build, { deep: true })
watch(onlyPath, (on) => {
  if (!on) { build(); return }
  nodes.value = nodes.value.map((n) => {
    if (n.type === 'clause-label') return n
    const onPath = pathNodes.value.has(n.id)
    return { ...n, hidden: !onPath }
  })
})

function nodeClass(data: { node: { nid: string }; onPath: boolean }): Record<string, boolean> {
  return {
    active: props.selectedNid === data.node.nid,
    'on-path': data.onPath,
    dim: (props.path ?? []).length > 0 && !data.onPath,
  }
}
</script>

<template>
  <div>
    <div class="row between" style="margin-bottom: 8px">
      <div class="row wrap" style="gap: 10px">
        <span class="tiny faint">facet 级条款图 · {{ graph?.n_nodes ?? 0 }} 节点 / {{ graph?.n_edges ?? 0 }} 边</span>
        <span class="tiny faint">{{ graph?.graph_version }}</span>
      </div>
      <label class="switch no-print">
        <input v-model="onlyPath" type="checkbox" /> 仅路径节点
      </label>
    </div>

    <div class="graph-shell">
      <svg style="position: absolute; width: 0; height: 0" aria-hidden="true">
        <defs>
          <marker id="cs-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M0 0L10 5L0 10z" fill="context-stroke" />
          </marker>
        </defs>
      </svg>
      <VueFlow
        :nodes="nodes" :edges="edges" :min-zoom="0.35" :max-zoom="1.5"
        :default-edge-options="{ type: 'smoothstep' }"
        fit-view-on-init
        :nodes-draggable="true"
        @node-click="(e) => e.node.type === 'facet' && emit('select', e.node.id)"
      >
        <template #node-facet="{ data }">
          <div class="node-facet" :class="nodeClass(data)">
            <div class="node-top">
              <span class="node-role">{{ roleMeta[data.node.role]?.cn ?? data.node.role }}</span>
              <span class="mono faint" style="font-size: 9px; margin-left: auto">{{ data.node.nid.split('#')[1] }}</span>
            </div>
            <div class="node-text">{{ data.node.text }}</div>
            <div class="node-foot">
              <span>{{ data.node.nid }}</span>
              <span v-if="data.onPath" style="color: var(--accent)">PATH</span>
            </div>
          </div>
        </template>
        <template #node-clause-label="{ data }">
          <div class="clause-label">{{ data.label }}</div>
        </template>
        <Background :gap="16" :size="1" pattern-color="#eaeaec" />
        <Controls :show-interactive="false" />
        <MiniMap pannable zoomable :height="90" :width="140" node-color="#e4e4e7" mask-color="rgba(255,255,255,.72)" />
      </VueFlow>
      <div class="graph-legend">
        <span v-for="(m, key) in edgeMeta" :key="key" class="legend-item">
          <span class="legend-line" :style="{ borderTopColor: m.color, borderTopStyle: m.dash ? 'dashed' : 'solid' }" />
          {{ m.cn }}
        </span>
      </div>
    </div>
  </div>
</template>
