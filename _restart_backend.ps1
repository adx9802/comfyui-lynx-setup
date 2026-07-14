$ErrorActionPreference = "SilentlyContinue"
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "ComfyUI\\main.py" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep -Seconds 3
$out = 'D:\LocalAI\LocalAI\comfy_manual_out.log'
$err = 'D:\LocalAI\LocalAI\comfy_manual_err.log'
Remove-Item $out, $err -ErrorAction SilentlyContinue
$args = @('-s','ComfyUI\main.py','--enable-manager',
  '--extra-model-paths-config','"C:\Users\aho\AppData\Roaming\Comfy Desktop\shared_model_paths.yaml"',
  '--input-directory','"C:\Users\aho\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input"',
  '--output-directory','"C:\temp"')
$p = Start-Process -FilePath 'D:\LocalAI\LocalAI\ComfyUI\.venv\Scripts\python.exe' -ArgumentList $args `
  -WorkingDirectory 'D:\LocalAI\LocalAI' -RedirectStandardOutput $out -RedirectStandardError $err `
  -WindowStyle Hidden -PassThru
Write-Output "launched backend PID $($p.Id)"
$up = $false
for ($i = 0; $i -lt 36; $i++) {
  Start-Sleep -Seconds 5
  try { $r = Invoke-RestMethod 'http://127.0.0.1:8188/system_stats' -TimeoutSec 4; $up = $true;
    Write-Output "Backend UP after ~$([int](($i+1)*5))s | VRAM free $([math]::Round($r.devices[0].vram_free/1GB,2)) GB"; break } catch {}
}
if (-not $up) { Write-Output "still down"; Get-Content $err -Tail 20 }
