@echo off
cd /d "%~dp0"
if exist "FDFormer\train.py" cd /d "%~dp0FDFormer"
if not exist "train.py" (
    echo [Error] Please put this bat file inside your FDFormer folder and double-click it!
    pause
    exit /b 1
)
echo [1/4] Initializing Git repository...
git init -b main
git config --local user.name "iuytrerty3-tech"
git config --local user.email "iuytrerty3-tech@users.noreply.github.com"
echo [2/4] Staging and committing files...
git add -A
git commit -m "Initial commit: FDFormer code and supplementary materials"
echo [3/4] Setting remote origin...
git remote remove origin >nul 2>&1
git remote add origin https://github.com/iuytrerty3-tech/FDFormer.git
git branch -M main
echo [4/4] Pushing to GitHub (a browser sign-in window may pop up)...
git push -u origin main --force
echo.
echo Done! Check https://github.com/iuytrerty3-tech/FDFormer
pause