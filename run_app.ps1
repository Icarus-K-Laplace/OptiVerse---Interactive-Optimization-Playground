# 使用 dl-env 环境运行 OptiVerse 应用
$pythonPath = "C:\Users\lenovo\Miniforge3\envs\dl-env\python.exe"

# 检查是否已安装依赖
Write-Host "检查并安装依赖..."
& $pythonPath -m pip install -r requirements.txt

# 运行应用
Write-Host "启动 OptiVerse 应用..."
& $pythonPath -m streamlit run app.py
