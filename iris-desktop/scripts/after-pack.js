/** electron-builder afterPack: embed custom icon into main .exe (works with signAndEditExecutable: false) */

const path = require('path')
const fs = require('fs')
const rcedit = require('rcedit')

module.exports = async function afterPack(context) {
  if (context.electronPlatformName !== 'win32') return

  const iconPath = path.join(context.packager.projectDir, 'build', 'icon.ico')
  if (!fs.existsSync(iconPath)) {
    console.warn('[after-pack] build/icon.ico missing, skip exe icon embed')
    return
  }

  const exeName = `${context.packager.appInfo.productFilename}.exe`
  const exePath = path.join(context.appOutDir, exeName)
  if (!fs.existsSync(exePath)) {
    console.warn(`[after-pack] exe not found: ${exePath}`)
    return
  }

  console.log(`[after-pack] embedding icon into ${exeName}`)
  await rcedit(exePath, { icon: iconPath })
}
