@echo off
REM Assemble submission.zip from the split parts, then verify it.
REM Usage: put this next to submission.part.* and run:  reassemble.cmd
setlocal

cd /d "%~dp0"

set "PARTS=submission.part.00 submission.part.01 submission.part.02 submission.part.03 submission.part.04"
set "EXPECTED=91a1c214475734e752f5c7d5641d0c1fec5a588ea2eba5a94fb36cd6a250b08f"

for %%F in (%PARTS%) do (
  if not exist "%%F" (
    echo Missing %%F
    exit /b 1
  )
)

echo Joining parts...
copy /b %PARTS% submission.zip > nul
if errorlevel 1 (
  echo Join failed
  exit /b 1
)
dir submission.zip

echo.
echo Verifying SHA256...
for /f "tokens=1" %%H in ('certutil -hashfile submission.zip SHA256 ^| findstr /r /v "hash CertUtil"') do set "ACTUAL=%%H"

if /I "%ACTUAL%"=="%EXPECTED%" (
  echo OK - the archive is intact. Upload submission.zip.
) else (
  echo MISMATCH
  echo expected %EXPECTED%
  echo actual   %ACTUAL%
  echo A mismatch means the download was corrupted. Do not upload it.
  exit /b 1
)
endlocal