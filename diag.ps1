# qwen_shutdown_focus.ps1
# 目的: 2026/01/01 17:39 と 17:47 の直前直後ログを「関係ありそうなものだけ」抽出してZIP化

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = Join-Path $env:TEMP "qwen_shutdown_focus_$ts"
New-Item -ItemType Directory -Path $outDir | Out-Null

# すでに分かっている再起動時刻（Kernel-Power 41の時刻）から前後を広めに確保
$start = Get-Date "2026-01-01 17:34:00"
$end   = Get-Date "2026-01-01 17:52:00"

# 重点プロバイダ（ここに痕跡が出ることが多い）
$providers = @(
  "Microsoft-Windows-Kernel-Power",
  "Microsoft-Windows-WHEA-Logger",
  "Microsoft-Windows-Display",
  "nvlddmkm",
  "Microsoft-Windows-DxgKrnl",
  "Microsoft-Windows-Kernel-Boot",
  "Microsoft-Windows-Kernel-General",
  "Microsoft-Windows-Eventlog",
  "Microsoft-Windows-WER-SystemErrorReporting"
)

# System / Application を時間範囲で取得し、プロバイダで絞る
$logs = @("System","Application")
foreach ($log in $logs) {
  $ev = Get-WinEvent -FilterHashtable @{LogName=$log; StartTime=$start; EndTime=$end} |
        Where-Object { $providers -contains $_.ProviderName } |
        Select-Object TimeCreated, LogName, Id, LevelDisplayName, ProviderName, Message
  $path = Join-Path $outDir "$log.filtered.txt"
  $ev | Format-List | Out-String | Out-File $path -Encoding UTF8
}

# WER / DxgKrnl の Operational があれば追加（環境によって無い/無効あり）
$extraLogs = @(
  "Microsoft-Windows-WER-SystemErrorReporting/Operational",
  "Microsoft-Windows-DxgKrnl/Operational"
)

foreach ($elog in $extraLogs) {
  try {
    $ev2 = Get-WinEvent -FilterHashtable @{LogName=$elog; StartTime=$start; EndTime=$end} |
           Select-Object TimeCreated, LogName, Id, LevelDisplayName, ProviderName, Message
    $p2 = Join-Path $outDir (($elog -replace '[\\/:]','_') + ".txt")
    $ev2 | Format-List | Out-String | Out-File $p2 -Encoding UTF8
  } catch { }
}

# GPUドライバ/電源関連のスナップショット
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  & nvidia-smi -q | Out-File (Join-Path $outDir "nvidia_smi_q.txt") -Encoding UTF8
}

# ZIP化
$zipPath = Join-Path $env:USERPROFILE "Desktop\qwen_shutdown_focus_$ts.zip"
Compress-Archive -Path "$outDir\*" -DestinationPath $zipPath -Force
"Saved: $zipPath"
