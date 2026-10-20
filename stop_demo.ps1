$ErrorActionPreference='Stop'
$statePath=Join-Path $PSScriptRoot '.runtime\running_web.json'
if (-not (Test-Path -LiteralPath $statePath)) {Write-Output '没有本启动器的运行记录。'; exit 0}
$state=Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
foreach ($serviceName in @('frontend','backend')) {
    $servicePid=[int]$state.($serviceName+'_pid')
    $serviceProcess=Get-Process -Id $servicePid -ErrorAction SilentlyContinue
    if ($null -eq $serviceProcess) {continue}
    $recordedStart=[datetime]$state.($serviceName+'_started')
    $actualStart=$serviceProcess.StartTime.ToUniversalTime()
    if ([math]::Abs(($actualStart-$recordedStart.ToUniversalTime()).TotalSeconds) -gt 1) {throw '进程编号已被其他程序复用，未停止该进程。'}
    if (($serviceName -eq 'backend' -and $serviceProcess.Path -ne $state.backend_exe) -or ($serviceName -eq 'frontend' -and $serviceProcess.Path -ne $state.frontend_exe)) {throw '进程路径不匹配，未停止该进程。'}
    Stop-Process -Id $servicePid
}
Write-Output '本启动器开启的演示服务已停止。'
