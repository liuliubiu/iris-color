# Shared encoding setup for PowerShell scripts (UTF-8 console + UTF-8 child process output)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
try {
    chcp 65001 | Out-Null
} catch {
    # ignore if chcp unavailable
}
