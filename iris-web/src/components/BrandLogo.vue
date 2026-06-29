<script setup lang="ts">
import { ref } from 'vue'
import { BRAND, type BrandLogoVariant } from '../config/brand'

defineProps<{
  variant?: BrandLogoVariant
}>()

const showFallback = ref(false)

function onError() {
  showFallback.value = true
}
</script>

<template>
  <img
    v-if="!showFallback"
    :src="BRAND.logoUrl"
    :alt="BRAND.shortName"
    :class="['brand-logo', variant ? `brand-logo--${variant}` : 'brand-logo--inline']"
    @error="onError"
  />
  <span
    v-else
    :class="['brand-logo-fallback', variant ? `brand-logo-fallback--${variant}` : 'brand-logo-fallback--inline']"
    aria-hidden="true"
  >豪</span>
</template>

<style scoped>
.brand-logo,
.brand-logo-fallback {
  flex-shrink: 0;
  object-fit: contain;
}

.brand-logo--desktop,
.brand-logo-fallback--desktop {
  width: 36px;
  height: 36px;
  border-radius: 6px;
  background: #fff;
  padding: 3px;
  box-sizing: border-box;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
}

.brand-logo-fallback--desktop {
  display: grid;
  place-items: center;
  background: linear-gradient(145deg, #3a9fd4, #1876a9);
  color: #fff;
  font-size: 16px;
  font-weight: 800;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.25);
}

.brand-logo--mobile,
.brand-logo-fallback--mobile {
  width: 28px;
  height: 28px;
  border-radius: 6px;
  background: #fff;
  padding: 2px;
  box-sizing: border-box;
}

.brand-logo-fallback--mobile {
  display: grid;
  place-items: center;
  background: linear-gradient(145deg, #3a9fd4, #1876a9);
  color: #fff;
  font-size: 14px;
  font-weight: 800;
}

.brand-logo--splash,
.brand-logo-fallback--splash {
  width: 64px;
  height: 64px;
  margin: 0 auto 16px;
  border-radius: 14px;
}

.brand-logo-fallback--splash {
  display: grid;
  place-items: center;
  background: rgba(255, 255, 255, 0.15);
  color: #fff;
  font-size: 28px;
  font-weight: 800;
}
</style>
