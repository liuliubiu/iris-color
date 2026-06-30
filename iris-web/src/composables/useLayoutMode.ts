import { onBeforeUnmount, onMounted, ref } from 'vue'

const MOBILE_MQ = '(max-width: 640px)'

function detectMobile(): boolean {
  const w = window as Window & { irisDesktop?: { isDesktop: boolean } }
  if (typeof window !== 'undefined' && w.irisDesktop?.isDesktop) {
    return false
  }
  if (typeof window === 'undefined') return false
  return window.matchMedia(MOBILE_MQ).matches
}

/** 640px 及以下为手机布局；Electron 桌面壳强制使用桌面布局 */
export function useLayoutMode() {
  const isMobile = ref(detectMobile())

  let mq: MediaQueryList | null = null

  function sync() {
    isMobile.value = detectMobile()
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
