import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { endpoints } from '@/api/endpoints'
import type { Health } from '@/api/types'

export const useSystemStore = defineStore('system', () => {
  const health = ref<Health | null>(null)
  const error = ref('')
  const checking = ref(false)
  const started = ref(false)
  let timer: number | undefined

  const ready = computed(() => Boolean(health.value?.ready))
  const status = computed(() => (error.value ? 'error' : (health.value?.status ?? 'loading')))

  async function refresh(): Promise<void> {
    if (checking.value) return
    checking.value = true
    try {
      health.value = await endpoints.health()
      error.value = ''
    } catch (err) {
      error.value = err instanceof Error ? err.message : String(err)
    } finally {
      checking.value = false
      schedule()
    }
  }

  function schedule(): void {
    window.clearTimeout(timer)
    timer = window.setTimeout(() => void refresh(), ready.value ? 15_000 : 3_000)
  }

  function start(): void {
    if (started.value) return
    started.value = true
    void refresh()
  }

  function stop(): void {
    window.clearTimeout(timer)
    started.value = false
  }

  return { health, error, checking, ready, status, refresh, start, stop }
})
