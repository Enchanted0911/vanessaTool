@echo off
chcp 65001 >nul
echo ============================================
echo   批量加水印工具 - Windows EXE 打包脚本
echo ============================================
echo.

:: 检查 Python 是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.11+
    pause
    exit /b 1
)

:: 安装依赖
echo [1/3] 安装依赖包...
pip install pillow numpy pyinstaller --quiet
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络或手动执行 pip install pillow numpy pyinstaller
    pause
    exit /b 1
)

:: 执行打包
echo [2/3] 开始打包（首次打包较慢，请耐心等待）...
pyinstaller build_win.spec --clean
if errorlevel 1 (
    echo [错误] 打包失败，请查看上方错误信息
    pause
    exit /b 1
)

:: 提示完成
echo.
echo [3/3] 打包完成！
echo.
echo EXE 文件位于：dist\批量加水印工具.exe
echo.
pause

