import { onBeforeUnmount, onMounted, ref } from 'vue'

const MOBILE_MQ = '(max-width: 640px)'

/** 640px 及以下为手机布局，641px 及以上为桌面/专业软件布局 */
export function useLayoutMode() {
  const isMobile = ref(
    typeof window !== 'undefined' ? window.matchMedia(MOBILE_MQ).matches : false,
  )

  let mq: MediaQueryList | null = null

  function sync() {
    isMobile.value = mq?.matches ?? false
    document.body.classList.toggle('layout-mobile-app', isMobile.value)
    document.body.classList.toggle('layout-desktop-app', !isMobile.value)
  }

  onMounted(() => {
    mq = window.matchMedia(MOBILE_MQ)
    sync()
    mq.addEventListener('change', sync)
  })

  onBeforeUnmount(() => {
    mq?.removeEventListener('change', sync)
    document.body.classList.remove('layout-mobile-app', 'layout-desktop-app')
  })

  return { isMobile }
}
