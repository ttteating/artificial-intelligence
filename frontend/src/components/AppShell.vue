<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSystemStore } from '@/stores/system'
import {
  BarChart3, BookOpen, LayoutDashboard, Library, Plus, Settings2, Sparkles,
} from 'lucide-vue-next'

const route = useRoute()
const router = useRouter()
const system = useSystemStore()

const nav = [
  { section: '工作', items: [
    { to: '/workspace', label: '工作台', icon: LayoutDashboard },
    { to: '/review/new', label: '审查台', icon: BookOpen },
  ] },
  { section: '数据', items: [
    { to: '/scenarios', label: '场景库', icon: Library },
    { to: '/benchmark', label: '评测', icon: BarChart3 },
  ] },
  { section: '系统', items: [
    { to: '/llm', label: 'LLM 治理', icon: Sparkles },
    { to: '/system', label: '系统', icon: Settings2 },
  ] },
]

const statusText = computed(() => {
  if (system.error) return '服务未连接'
  if (!system.health) return '连接中'
  if (!system.health.ready) return '加载语料中'
  return `${system.health.n_scenarios.toLocaleString()} 场景 · ${system.health.graph_version}`
})

const dotClass = computed(() => {
  if (system.error) return 'error'
  if (system.health?.ready) return 'ok'
  return 'loading'
})

const title = computed(() => {
  if (route.name === 'review') return `审查 ${String(route.params.sid ?? '')}`
  return (route.meta.title as string) ?? 'ClauseScope'
})

const active = (to: string) => {
  if (to === '/review/new') return route.path.startsWith('/review')
  return route.path === to || route.path.startsWith(`${to}/`)
}

onMounted(() => system.start())
</script>

<template>
  <div class="shell">
    <aside class="rail">
      <div class="rail-brand">
        <div class="rail-mark">CS</div>
        <div class="rail-name">ClauseScope</div>
        <div class="rail-ver">v1.0</div>
      </div>

      <nav class="rail-nav" aria-label="主导航">
        <template v-for="group in nav" :key="group.section">
          <div class="rail-section">{{ group.section }}</div>
          <router-link
            v-for="item in group.items" :key="item.to" :to="item.to"
            class="nav-item" :class="{ active: active(item.to) }" :title="item.label"
          >
            <component :is="item.icon" />
            <span>{{ item.label }}</span>
          </router-link>
        </template>
      </nav>

      <div class="rail-foot">
        <div class="row" style="gap: 6px">
          <span class="status-dot" :class="dotClass" />
          <span>{{ statusText }}</span>
        </div>
        <div style="margin-top: 6px">辅助审查工具，不构成法律意见</div>
      </div>
    </aside>

    <div class="main">
      <header class="topbar">
        <div class="topbar-title">{{ title }}</div>
        <div class="topbar-crumb" v-if="route.name === 'review' && route.query.condition">
          / {{ route.query.condition }} · 预算 {{ route.query.budget ?? 16 }}
        </div>
        <div class="topbar-actions">
          <span v-if="system.health?.llm_mode" class="tag mono" :class="{ warn: system.health.llm_mode === 'mock' }">
            LLM {{ system.health.llm_mode }}
          </span>
          <span class="tag line mono">{{ system.health?.graph_version || '—' }}</span>
          <a class="btn ghost sm" href="/docs" target="_blank" rel="noreferrer">接口文档</a>
          <button
            v-if="route.path !== '/review/new'"
            class="btn primary sm" type="button" @click="router.push('/review/new')"
          >
            <Plus /> 新建审查
          </button>
        </div>
      </header>
      <slot />
    </div>
  </div>
</template>
