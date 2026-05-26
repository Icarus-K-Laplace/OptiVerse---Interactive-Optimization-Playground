# OptiVerse 安装和运行脚本

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  OptiVerse - 交互式数值优化演武场" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$pythonPath = "C:\Users\lenovo\Miniforge3\envs\dl-env\python.exe"

# 第一步：检查 Python
Write-Host "[1/3] 检查 Python 环境..." -ForegroundColor Yellow
if (-not (Test-Path $pythonPath)) {
    Write-Host "错误：找不到 Python 可执行文件" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Python 找到了！" -ForegroundColor Green
& $pythonPath --version
Write-Host ""

# 第二步：安装依赖
Write-Host "[2/3] 安装依赖..." -ForegroundColor Yellow
& $pythonPath -m pip install --upgrade pip
& $pythonPath -m pip install -r requirements.txt
Write-Host "✓ 依赖安装完成！" -ForegroundColor Green
Write-Host ""

# 第三步：运行应用
Write-Host "[3/3] 启动 OptiVerse 应用..." -ForegroundColor Yellow
Write-Host "应用将在浏览器中自动打开..." -ForegroundColor Gray
Write-Host ""
& $pythonPath -m streamlit run app.py
