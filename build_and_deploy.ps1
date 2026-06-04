# 1. 初期設定と仮想環境の有効化
Set-Location -Path $PSScriptRoot
$ErrorActionPreference = "Continue"

$venvActivate = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "--- Activating virtual environment ---" -ForegroundColor Cyan
    . $venvActivate
} else {
    Write-Host "Error: .venv not found!" -ForegroundColor Red
    exit 1
}

# 2. 古いビルド成果物のクリーンアップ
$buildDir = Join-Path $PSScriptRoot "build"
$distBuildDir = Join-Path $PSScriptRoot "dist_prep" # 作業用フォルダ

Write-Host "--- Cleaning up old folders ---" -ForegroundColor Cyan
Remove-Item -Path $buildDir, $distBuildDir, ".\dist", ".\_internal" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path ".\run_app.exe" -Force -ErrorAction SilentlyContinue

# 3. Cython extensions のビルド
Write-Host "--- Building Cython extensions ---" -ForegroundColor Cyan
py setup.py build_ext

# 4. 作業ディレクトリ (dist_prep) の作成と集約
# build/lib.win-amd64-cpython-312 を探す
$buildLib = Get-ChildItem -Path "build" -Filter "lib.*" -Directory | Select-Object -First 1
if (-not $buildLib) {
    Write-Error "Cython build folder not found!"
    exit 1
}

Write-Host "--- Preparing dist_prep folder ---" -ForegroundColor Cyan
New-Item -Path $distBuildDir -ItemType Directory -Force | Out-Null
Copy-Item -Path "$($buildLib.FullName)\*" -Destination $distBuildDir -Recurse -Force

# 5. 不要な C ファイル削除
Write-Host "--- Finalizing package structure ---" -ForegroundColor Cyan
$cTargetDirs = @("anomalib_app")
foreach ($dir in $cTargetDirs) {
    if (Test-Path $dir) {
        Get-ChildItem -Path $dir -Filter "*.c" -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue
    }
}

# 6. 実行に必要なスクリプト群を dist_prep にコピー
$filesToCopy = @(
    "run_app.py",
    "run_app.spec",
    "anomalib_app/streamlit_app.py",
    "LICENSE",
    "config.json"
)

foreach ($f in $filesToCopy) {
    if (Test-Path $f) {

        if ($f -like "anomalib_app/*") {
            # サブフォルダ用
            $destDir = Join-Path $distBuildDir "anomalib_app"

            if (!(Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir | Out-Null
            }

            Copy-Item -Path $f -Destination $destDir -Force
        }
        else {
            # 通常コピー
            Copy-Item -Path $f -Destination $distBuildDir -Force
        }

    } else {
        Write-Host "Warning: $f not found, skipping." -ForegroundColor Yellow
    }
}

# Read-Host "Press Enter to continue"

# 7. PyInstaller の実行
Write-Host "--- Executing PyInstaller ---" -ForegroundColor Cyan
Set-Location -Path $distBuildDir
pyinstaller run_app.spec --noconfirm

# Read-Host "Press Enter to continue"

# 8. 成果物の移動と後片付け
Write-Host "--- Moving results to root ---" -ForegroundColor Cyan
$pyinstallerOut = Join-Path "dist" "run_app"
if (Test-Path $pyinstallerOut) {
    # dist_prepに戻って移動
    $rootPath = $distBuildDir
    $execPath = Join-Path $rootPath "dist"
    Move-Item -Path "$pyinstallerOut\*" -Destination $rootPath -Force
    Set-Location -Path $rootPath

    # Read-Host "Press Enter to continue"

    # streamlit_app.pyのコピー
    $srcAppPath = Join-Path $distBuildDir "anomalib_app\streamlit_app.py"
    $dstAppPath = Join-Path $distBuildDir "_internal\anomalib_app\"
    Copy-Item -Path $srcAppPath -Destination $dstAppPath -Force

    # Read-Host "Press Enter to continue"
    
    # 作業用フォルダの削除
    $distBuildDirBuild = Join-Path $distBuildDir "build"
    Remove-Item -Path $distBuildDirBuild -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path $buildDir -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path $pyinstallerOut -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path $execPath -Recurse -Force -ErrorAction SilentlyContinue
    $distLibPath = Join-Path $distBuildDir "anomalib_app"
    Remove-Item -Path $distLibPath -Recurse -Force -ErrorAction SilentlyContinue
    $distRunAppPath = Join-Path $distBuildDir "run_app.py"
    Remove-Item -Path $distRunAppPath -Force -ErrorAction SilentlyContinue
    $distRunSpecPath = Join-Path $distBuildDir "run_app.spec"
    Remove-Item -Path $distRunSpecPath -Force -ErrorAction SilentlyContinue

    # Read-Host "Press Enter to continue"
} else {
    Set-Location -Path $PSScriptRoot
    Write-Error "PyInstaller failed to generate output."
    exit 1
}

# Read-Host "Press Enter to continue"

# # 9. ショートカットの作成
# if (Test-Path ".\run_app.exe") {
#     $shortcutPath = Join-Path $distBuildDir "run_app.lnk"
#     if (Test-Path $shortcutPath) {
#         Remove-Item -Path $shortcutPath -Force -ErrorAction SilentlyContinue
#     }
#     $WScriptShell = New-Object -ComObject WScript.Shell
#     $shortcut = $WScriptShell.CreateShortcut($shortcutPath)
#     $shortcut.TargetPath = Join-Path $distBuildDir "run_app.exe"
#     $shortcut.WindowStyle = 7 # 最小化で起動
#     $shortcut.IconLocation = "C:\Windows\System32\shell32.dll,251"
#     $shortcut.WorkingDirectory = $distBuildDir
#     $shortcut.Save()
# } else {
#     Write-Error "run_app.exe not found."
# }

# # 10. アプリの実行
# Start-Process -FilePath $shortcutPath