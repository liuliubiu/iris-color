# 品牌图标资源

将您的图标文件放入**本目录**后，在项目根目录执行：

```powershell
.\scripts\sync-brand.ps1
```

然后**刷新浏览器**（`Ctrl + F5`）或重启 `npm run dev`。若 dev 服务在放图标之前就已启动，需先同步再刷新。

脚本会自动复制到网页与桌面软件所需位置。

## 需要准备的文件

| 文件名 | 用途 | 建议规格 |
|--------|------|----------|
| `logo.png` | 网页顶栏 Logo（桌面端 + 手机端） | PNG，透明底，**128×128** 或更大 |
| `favicon.ico` | 浏览器标签页图标 | ICO，含 **16×16** 与 **32×32** |
| `favicon.png` | 浏览器标签页（可选，现代浏览器） | PNG **32×32** |
| `apple-touch-icon.png` | 手机「添加到主屏幕」图标（可选） | PNG **180×180** |
| `app-icon.ico` | Windows 桌面软件 / 安装包 / 任务栏 | ICO，**256×256** 为主 |

## 复制目标

```
brand/logo.png              → iris-web/public/brand/logo.png
brand/favicon.ico           → iris-web/public/brand/favicon.ico
brand/favicon.png           → iris-web/public/brand/favicon.png（若存在）
brand/apple-touch-icon.png  → iris-web/public/brand/apple-touch-icon.png（若存在）
brand/app-icon.ico          → iris-desktop/build/icon.ico
brand/logo.png              → iris-desktop/build/logo.png（启动页用）
```

## 开发时

```powershell
# 1. 图标放入 brand/ 后同步
.\scripts\sync-brand.ps1

# 2. 网页开发（刷新即可看到新图标）
cd iris-web
npm run dev

# 3. 桌面软件需重新打包
cd iris-desktop
.\scripts\build-all.ps1 -SkipRuntimePrep
```

若未放置 `logo.png`，顶栏会显示文字「豪」作为占位；未放置 `app-icon.ico` 时 Electron 使用默认图标。
