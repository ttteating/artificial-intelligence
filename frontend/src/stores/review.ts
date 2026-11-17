import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { endpoints } from '@/api/endpoints'
import type { Condition, GraphOut, Label, ScenarioOut, SearchResponse } from '@/api/types'

export type Selection =
  | { kind: 'step'; index: number }
  | { kind: 'node'; nid: string }
  | { kind: 'tree'; tid: number }
  | { kind: 'clause'; clause: string }
  | null

export interface RecentItem {
  sid: string
  condition: Condition
  answer: Label
  conf: number
  at: string
  elapsed: number
  title?: string
}

const RECENT_KEY = 'clausescope.recent.v1'
const TITLE_KEY = 'clausescope.titles.v1'

const scenarioCache = new Map<string, ScenarioOut>()
const graphCache = new Map<string, GraphOut>()
const searchCache = new Map<string, SearchResponse>()

function readRecent(): RecentItem[] {
  try {
    const raw = localStorage.getItem(RECENT_KEY)
    const list = raw ? (JSON.parse(raw) as RecentItem[]) : []
    return Array.isArray(list) ? list : []
  } catch {
    return []
  }
}

function readTitles(): Record<string, string> {
  try {
    const raw = localStorage.getItem(TITLE_KEY)
    return raw ? (JSON.parse(raw) as Record<string, string>) : {}
  } catch {
    return {}
  }
}

export const useReviewStore = defineStore('review', () => {
  const result = ref<SearchResponse | null>(null)
  const scenario = ref<ScenarioOut | null>(null)
  const graph = ref<GraphOut | null>(null)
  const loading = ref(false)
  const loadingScenario = ref(false)
  const error = ref('')
  const stage = ref('')
  const selection = ref<Selection>(null)
  const recent = ref<RecentItem[]>(readRecent())
  const titles = ref<Record<string, string>>(readTitles())
  const elapsed = ref(0)

  let controller: AbortController | null = null
  let stageTimers: number[] = []
  let tickTimer: number | undefined

  const answer = computed(() => result.value?.answer ?? null)
  const consistent = computed(() => {
    const gold = result.value?.meta.gold
    if (!gold || gold === 'Unlabeled') return null
    return gold === result.value?.answer
  })

  function startClock(): void {
    const t0 = performance.now()
    elapsed.value = 0
    window.clearInterval(tickTimer)
    tickTimer = window.setInterval(() => { elapsed.value = performance.now() - t0 }, 200)
  }

  function stopClock(): void {
    window.clearInterval(tickTimer)
    tickTimer = undefined
  }

  function setStages(): void {
    stageTimers.forEach((t) => window.clearTimeout(t))
    stageTimers = []
    const stages: Array<[number, string]> = [
      [0, '校验场景与条款图'],
      [450, '规则掩码与启发评估'],
      [1_100, '在条款图上扩展'],
      [2_600, '终局三分类评估'],
      [3_600, '生成证据链解释'],
    ]
    stageTimers = stages.map(([delay, text]) =>
      window.setTimeout(() => { stage.value = text }, delay))
  }

  function clearStages(): void {
    stageTimers.forEach((t) => window.clearTimeout(t))
    stageTimers = []
    stage.value = ''
  }

  async function loadScenario(sid: string): Promise<void> {
    if (scenario.value?.sid === sid) return
    const cached = scenarioCache.get(sid)
    if (cached) { scenario.value = cached; return }
    loadingScenario.value = true
    try {
      scenario.value = await endpoints.scenario(sid)
      scenarioCache.set(sid, scenario.value)
    } catch (err) {
      error.value = err instanceof Error ? err.message : String(err)
    } finally {
      loadingScenario.value = false
    }
  }

  async function loadGraph(sid: string): Promise<void> {
    if (graph.value?.sid === sid) return
    const cached = graphCache.get(sid)
    if (cached) { graph.value = cached; return }
    try {
      graph.value = await endpoints.graph(sid)
      graphCache.set(sid, graph.value)
    } catch {
      graph.value = null
    }
  }

  async function run(sid: string, condition: Condition, budget: number,
                     explain: boolean, explainMode: 'template' | 'real' | 'auto'): Promise<void> {
    const key = `${sid}|${condition}|${budget}|${explain ? 1 : 0}|${explainMode}`
    controller?.abort()
    controller = new AbortController()
    error.value = ''
    selection.value = null
    const cached = searchCache.get(key)
    if (cached) {
      result.value = cached
      void loadScenario(sid)
      void loadGraph(sid)
      return
    }
    result.value = null
    loading.value = true
    startClock()
    setStages()
    try {
      const [res] = await Promise.all([
        endpoints.search({
          sid,
          condition,
          explain,
          explain_mode: explainMode,
          max_expansions: budget,
          include_trace: true,
          include_tree: true,
        }, controller.signal),
        loadScenario(sid),
        loadGraph(sid),
      ])
      result.value = res
      searchCache.set(key, res)
      pushRecent(res, titles.value[sid])
    } catch (err) {
      if (err instanceof Error && err.message.includes('取消')) {
        // 用户主动取消：保留现状，不当作错误
      } else {
        error.value = err instanceof Error ? err.message : String(err)
      }
    } finally {
      loading.value = false
      stopClock()
      clearStages()
    }
  }

  function cancel(): void {
    controller?.abort()
    controller = null
    loading.value = false
    stopClock()
    clearStages()
  }

  function select(next: Selection): void {
    selection.value = next
  }

  function rememberTitle(sid: string, title: string): void {
    titles.value = { ...titles.value, [sid]: title }
    try { localStorage.setItem(TITLE_KEY, JSON.stringify(titles.value)) } catch { /* 忽略配额错误 */ }
  }

  function pushRecent(res: SearchResponse, title?: string): void {
    const item: RecentItem = {
      sid: res.sid,
      condition: res.condition,
      answer: res.answer,
      conf: res.confidence,
      at: new Date().toISOString(),
      elapsed: Math.round(res.elapsed_ms),
      title,
    }
    recent.value = [item, ...recent.value.filter((r) => !(r.sid === item.sid && r.condition === item.condition))].slice(0, 10)
    try { localStorage.setItem(RECENT_KEY, JSON.stringify(recent.value)) } catch { /* 忽略 */ }
  }

  function reset(): void {
    cancel()
    result.value = null
    scenario.value = null
    graph.value = null
    error.value = ''
    selection.value = null
  }

  return {
    result, scenario, graph, loading, loadingScenario, error, stage, selection,
    recent, titles, elapsed, answer, consistent,
    loadScenario, loadGraph, run, cancel, select, rememberTitle, reset,
  }
})
