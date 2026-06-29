const { contextBridge } = require('electron')

contextBridge.exposeInMainWorld('irisDesktop', {
  isDesktop: true,
})
