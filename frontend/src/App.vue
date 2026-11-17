<script setup lang="ts">
import AppShell from '@/components/AppShell.vue'
import { useToast } from '@/utils/toast'

const { message } = useToast()
</script>

<template>
  <AppShell>
    <router-view v-slot="{ Component }">
      <transition name="fade" mode="out-in">
        <component :is="Component" />
      </transition>
    </router-view>
  </AppShell>
  <div class="toast" :class="{ on: !!message }" role="status" aria-live="polite">{{ message }}</div>
</template>

<style>
.fade-enter-active, .fade-leave-active { transition: opacity 120ms ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>
