import { ref } from 'vue'

const message = ref('')
let timer: number | undefined

export function useToast() {
  function toast(text: string, duration = 2200): void {
    message.value = text
    window.clearTimeout(timer)
    timer = window.setTimeout(() => { message.value = '' }, duration)
  }
  return { message, toast }
}
