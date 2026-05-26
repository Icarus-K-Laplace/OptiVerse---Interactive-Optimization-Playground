@echo off
echo ========================================
echo   OptiVerse - 安装和运行脚本
echo ========================================
echo.

echo [步骤 1/3] 检查 Python...
"C:\Users\lenovo\Miniforge3\envs\dl-env\python.exe" --version
if errorlevel 1 (
    echo 错误：无法找到 Python
    pause
    exit /b 1
)
echo.

echo [步骤 2/3] 安装依赖...
echo 正在安装 streamlit, matplotlib, plotly...
"C:\Users\lenovo\Miniforge3\envs\dl-env\python.exe" -m pip install streamlit matplotlib plotly
if errorlevel 1 (
    echo 错误：安装依赖失败
    pause
    exit /b 1
)
echo.

echo [步骤 3/3] 启动应用...
echo 应用将在浏览器中自动打开
echo 按 Ctrl+C 可以停止应用
echo.
"C:\Users\lenovo\Miniforge3\envs\dl-env\python.exe" -m streamlit run app.py

pause
