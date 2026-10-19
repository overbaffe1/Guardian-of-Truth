# Reassemble submission.zip from the split parts, then verify it.
# PowerShell 5.1+ :  powershell -ExecutionPolicy Bypass -File reassemble.ps1
#
# Note: `cat` in PowerShell is an alias for Get-Content, which reads files
# instead of joining them. The parts are streamed here so that an 8.6 GB
# archive never has to fit in memory.

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$Expected = '91a1c214475734e752f5c7d5641d0c1fec5a588ea2eba5a94fb36cd6a250b08f'
$Target = Join-Path $PSScriptRoot 'submission.zip'

# Sort by name so 00,01,02... stay in order; a plain glob would be fine too
# because the names are zero-padded, but be explicit about it.
$parts = @(Get-ChildItem -Path $PSScriptRoot -Filter 'submission.part.*' -File | Sort-Object Name)
if ($parts.Count -eq 0) {
    Write-Error 'No submission.part.* files found next to this script.'
    exit 1
}

Write-Host "Found $($parts.Count) part(s):"
$parts | ForEach-Object { Write-Host ("  {0}  {1:N2} GiB" -f $_.Name, ($_.Length / 1GB)) }

Write-Host 'Joining...'
$out = [System.IO.File]::Create($Target)
try {
    foreach ($part in $parts) {
        $in = [System.IO.File]::OpenRead($part.FullName)
        try {
            $in.CopyTo($out)
        }
        finally {
            $in.Dispose()
        }
        Write-Host ("  + {0}" -f $part.Name)
    }
}
finally {
    $out.Dispose()
}

$size = (Get-Item $Target).Length
Write-Host ("Joined: {0} ({1:N2} GiB)" -f $Target, ($size / 1GB))

Write-Host 'Verifying SHA256...'
$actual = (Get-FileHash -Path $Target -Algorithm SHA256).Hash.ToLower()
if ($actual -eq $Expected.ToLower()) {
    Write-Host 'OK - the archive is intact. Upload submission.zip.' -ForegroundColor Green
}
else {
    Write-Host 'MISMATCH' -ForegroundColor Red
    Write-Host "expected $Expected"
    Write-Host "actual   $actual"
    Write-Host 'A mismatch means the download was corrupted. Do not upload it.'
    exit 1
}