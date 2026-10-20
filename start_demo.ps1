param([switch]$NoBrowser, [string]$PythonExe, [string]$NodeExe)
$ErrorActionPreference='Stop'
$demoRoot=[IO.Path]::GetFullPath($PSScriptRoot)
$projectRoot=$demoRoot
$runtimeRoot=Join-Path $demoRoot '.runtime'
if (-not $PythonExe) { if (Test-Path -LiteralPath 'D:\Python\python.exe') {$PythonExe='D:\Python\python.exe'} else {$PythonExe=(Get-Command python -ErrorAction Stop).Source} }
if (-not $NodeExe) {$NodeExe=(Get-Command node -ErrorAction Stop).Source}
$expectedHash='f54e50f3c0af5585b8ce894d7eaa421ddf58306c0339dd2ef00912e454c5b1ee'
foreach ($requiredPath in @($pythonExe,$nodeExe,(Join-Path $projectRoot 'backend\main.py'),(Join-Path $projectRoot 'frontend\node_modules\vite\bin\vite.js'))) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {throw "演示依赖不存在：$requiredPath"}
}
[IO.Directory]::CreateDirectory($runtimeRoot) | Out-Null
$listeners=Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object {$_.LocalPort -in @(8000,5173)}
if ($listeners) {
    try {
        $existingHealth=Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 2
        $expectedWeight=Join-Path $projectRoot 'backend\weights\yolo11s_full_25d_epoch090.pt'
        $actualWeight=[IO.Path]::GetFullPath($existingHealth.config.yolo_weights).Replace('/','\')
        if ($actualWeight -ne $expectedWeight -or $existingHealth.config.yolo_sha256 -ne $expectedHash) {throw '当前端口对应其他项目'}
        if (-not ($listeners | Where-Object {$_.LocalPort -eq 5173})) {throw '前端未启动'}
        Write-Output '最终版本演示已在运行。'
        if (-not $NoBrowser) {Start-Process 'http://127.0.0.1:5173/'}
        exit 0
    } catch {throw '8000或5173端口已被其他服务占用，请先关闭原服务。'}
}
$env:PYTHONIOENCODING='utf-8'
$backend=Start-Process -FilePath $pythonExe -ArgumentList @('-B','main.py') -WorkingDirectory (Join-Path $projectRoot 'backend') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimeRoot 'backend.stdout.log') -RedirectStandardError (Join-Path $runtimeRoot 'backend.stderr.log') -PassThru
$frontend=Start-Process -FilePath $nodeExe -ArgumentList @('node_modules/vite/bin/vite.js','--host','127.0.0.1') -WorkingDirectory (Join-Path $projectRoot 'frontend') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimeRoot 'frontend.stdout.log') -RedirectStandardError (Join-Path $runtimeRoot 'frontend.stderr.log') -PassThru
@{project=$projectRoot;backend_pid=$backend.Id;backend_exe=$pythonExe;frontend_exe=$nodeExe;backend_started=$backend.StartTime.ToUniversalTime().ToString('o');frontend_pid=$frontend.Id;frontend_started=$frontend.StartTime.ToUniversalTime().ToString('o');url='http://127.0.0.1:5173/'} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimeRoot 'running_web.json') -Encoding utf8
$health=$null
for ($attempt=0;$attempt -lt 20;$attempt++) {
    try {$health=Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 1; break} catch {Start-Sleep -Seconds 1}
}
if ($null -eq $health -or $health.status -ne 'ok' -or $health.config.yolo_sha256 -ne $expectedHash) {throw "后端未就绪，请查看 $runtimeRoot 中的日志。"}
if (-not $NoBrowser) {Start-Process 'http://127.0.0.1:5173/'}
Write-Output ('演示已就绪：http://127.0.0.1:5173/；当前权重为第'+$health.config.selected_epoch+'轮。')
