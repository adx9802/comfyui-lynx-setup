param(
    [string]$ComfyRoot = $env:LYNX_COMFY_ROOT,
    [string]$Python = $env:LYNX_PYTHON,
    [string]$ModelsRoot = $env:LYNX_MODELS_ROOT,
    [string]$LogFile = $env:LYNX_LOG_FILE,
    [string]$WorkflowDst = $env:LYNX_WORKFLOW_DST,
    [switch]$SkipBuildTools,
    [switch]$SkipModels
)

$ErrorActionPreference = "Continue"

if (-not $ComfyRoot) { $ComfyRoot = "D:\LocalAI\LocalAI\ComfyUI" }
if (-not $Python) { $Python = Join-Path $ComfyRoot ".venv\Scripts\python.exe" }
if (-not $ModelsRoot) { $ModelsRoot = Join-Path $env:LOCALAPPDATA "Comfy-Desktop\ComfyUI-Shared\models" }
if (-not $LogFile) { $LogFile = Join-Path $env:USERPROFILE "lynx_setup_log.txt" }
if (-not $WorkflowDst) { $WorkflowDst = Join-Path ([Environment]::GetFolderPath("Desktop")) "lynx_workflow.json" }

$CustomNodes = Join-Path $ComfyRoot "custom_nodes"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts $msg"
    Add-Content -Path $LogFile -Value $line
    Write-Output $line
}

if (Test-Path $LogFile) { Remove-Item $LogFile -Force }
Log "=== Lynx ComfyUI Setup ==="

if (-not (Test-Path $ComfyRoot)) { Log "ERROR: ComfyUI not found at $ComfyRoot"; exit 1 }
if (-not (Test-Path $Python)) { Log "ERROR: Python not found at $Python"; exit 1 }

Log "ComfyUI: $ComfyRoot"
Log "Python: $Python"
Log "Models: $ModelsRoot"

function Test-CppCompiler {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $false }
    $inst = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    return [bool]$inst
}

function Import-VsDevEnv {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return }
    $inst = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if (-not $inst) { return }
    $devShell = Join-Path $inst "Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
    if (Test-Path $devShell) {
        try {
            Import-Module $devShell -ErrorAction Stop
            Enter-VsDevShell -VsInstallPath $inst -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64" | Out-Null
            Log "VS Dev environment loaded from $inst"
        } catch {
            Log "WARN: Could not load VS Dev environment: $_"
        }
    }
}

if (-not $SkipBuildTools) {
    if (Test-CppCompiler) {
        Log "C++ Build Tools already installed."
    } else {
        Log "C++ Build Tools not found. Installing Visual Studio 2022 Build Tools (C++ workload)..."
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if ($winget) {
            Log "Installing via winget (this may take several minutes)..."
            winget install --id Microsoft.VisualStudio.2022.BuildTools -e --accept-source-agreements --accept-package-agreements --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" 2>&1 | ForEach-Object { Log $_ }
        } else {
            Log "winget not available. Downloading vs_BuildTools.exe directly..."
            $bootstrapper = Join-Path $env:TEMP "vs_BuildTools.exe"
            try {
                Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vs_BuildTools.exe" -OutFile $bootstrapper -UseBasicParsing
                Log "Running installer (quiet)..."
                $p = Start-Process -FilePath $bootstrapper -ArgumentList "--quiet","--wait","--norestart","--add","Microsoft.VisualStudio.Workload.VCTools","--includeRecommended" -Wait -PassThru
                Log "Installer exit code: $($p.ExitCode)"
            } catch {
                Log "ERROR: Failed to download/install Build Tools: $_"
                Log "Manual install: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
            }
        }
        if (Test-CppCompiler) {
            Log "C++ Build Tools installed successfully."
        } else {
            Log "WARN: C++ Build Tools still not detected. A reboot may be required."
        }
    }
    Import-VsDevEnv
} else {
    Log "Skipping Build Tools install (-SkipBuildTools)."
}

$dirs = @(
    "$ModelsRoot\diffusion_models\WanVideo\fp8_scaled_kj\T2V",
    "$ModelsRoot\diffusion_models\WanVideo\lynx",
    "$ModelsRoot\text_encoders",
    "$ModelsRoot\vae\wanvideo",
    "$ModelsRoot\loras\WanVideo\Lightx2v",
    "$ModelsRoot\loras\WanVid\funreward",
    $CustomNodes
)
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Path $d -Force | Out-Null
    Log "Dir OK: $d"
}

function Ensure-GitClone($url, $name) {
    $dest = Join-Path $CustomNodes $name
    if (Test-Path (Join-Path $dest ".git")) {
        Log "Updating $name..."
        Push-Location $dest
        git pull --ff-only 2>&1 | ForEach-Object { Log $_ }
        Pop-Location
    } else {
        Log "Cloning $name..."
        git clone $url $dest 2>&1 | ForEach-Object { Log $_ }
    }
}

Log "Installing huggingface_hub..."
& $Python -m pip install --upgrade pip huggingface_hub 2>&1 | ForEach-Object { Log $_ }

Ensure-GitClone "https://github.com/kijai/ComfyUI-WanVideoWrapper.git" "ComfyUI-WanVideoWrapper"
Ensure-GitClone "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git" "ComfyUI-VideoHelperSuite"
Ensure-GitClone "https://github.com/kijai/ComfyUI-KJNodes.git" "ComfyUI-KJNodes"

$reqFiles = @(
    (Join-Path $CustomNodes "ComfyUI-WanVideoWrapper\requirements.txt"),
    (Join-Path $CustomNodes "ComfyUI-VideoHelperSuite\requirements.txt"),
    (Join-Path $CustomNodes "ComfyUI-KJNodes\requirements.txt")
)
foreach ($rf in $reqFiles) {
    if (Test-Path $rf) {
        Log "pip install -r $rf (prefer-binary)"
        & $Python -m pip install --prefer-binary --only-binary=numpy,scipy,opencv-python,opencv-python-headless -r $rf 2>&1 | ForEach-Object { Log $_ }
    }
}

Log "Installing face encoding deps (insightface + onnxruntime-gpu)..."
Log "NOTE: skipping facenet-pytorch (pins numpy 1.26 and breaks Python 3.13)."
& $Python -m pip install onnxruntime-gpu 2>&1 | ForEach-Object { Log $_ }
& $Python -m pip install insightface 2>&1 | ForEach-Object { Log $_ }
if ($LASTEXITCODE -ne 0) {
    Log "WARN: insightface install failed. Lynx still works with manual face crop."
}

if (-not $SkipModels) {
    $pyPath = Join-Path $ScriptDir "download_models.py"
    if (-not (Test-Path $pyPath)) { Log "ERROR: download_models.py not found at $pyPath"; exit 1 }
    Log "Starting model downloads (~35GB)... (live progress bar shown in console)"
    & $Python $pyPath --models-root $ModelsRoot
    Log "Model download step finished (exit code $LASTEXITCODE)"
} else {
    Log "Skipping model downloads (-SkipModels)."
}

$workflowSrc = Join-Path $CustomNodes "ComfyUI-WanVideoWrapper\example_workflows\wanvideo_2_1_14B_T2V_14B_lynx_example_01.json"
if (Test-Path $workflowSrc) {
    Copy-Item $workflowSrc $WorkflowDst -Force
    Log "Workflow copied to $WorkflowDst"
}

Log "=== SETUP COMPLETE ==="
Log "Restart Comfy Desktop, then load the Lynx workflow JSON in ComfyUI."
