<script setup lang="ts">
import CapturePage from './views/CapturePage.vue'
</script>

<template>
  <CapturePage />
</template>

<style>
* {
  box-sizing: border-box;
}

html {
  -webkit-text-size-adjust: 100%;
}

body {
  margin: 0;
  min-width: 320px;
  font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', system-ui, sans-serif;
  background: #eef5f8;
  -webkit-font-smoothing: antialiased;
  /* 防止移动端横向滚动与下拉橡皮筋穿透 */
  overflow-x: hidden;
  overscroll-behavior-y: contain;
}

button,
input {
  font: inherit;
}

/* 移动端去除点击高亮，统一触控反馈 */
@media (hover: none) and (pointer: coarse) {
  * {
    -webkit-tap-highlight-color: transparent;
  }
}
</style>
