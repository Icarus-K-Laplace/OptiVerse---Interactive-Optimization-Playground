@echo off
REM 使用 dl-env 环境运行 OptiVerse 应用
"C:\Users\lenovo\Miniforge3\envs\dl-env\python.exe" -m pip install -r requirements.txt
"C:\Users\lenovo\Miniforge3\envs\dl-env\python.exe" -m streamlit run app.py
pause
