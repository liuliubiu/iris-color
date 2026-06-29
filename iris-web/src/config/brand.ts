/** 品牌资源路径（文件位于 iris-web/public/brand/，由 scripts/sync-brand.ps1 同步） */
export const BRAND = {
  name: '豪赋-虹膜颜色识别',
  shortName: '豪赋',
  logoUrl: '/brand/logo.png',
  faviconIco: '/brand/favicon.ico',
  faviconPng: '/brand/favicon.png',
  appleTouchIcon: '/brand/apple-touch-icon.png',
} as const

export type BrandLogoVariant = 'desktop' | 'mobile' | 'splash'
