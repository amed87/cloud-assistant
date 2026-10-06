[CmdletBinding()]
param(
    [string]$Version = 'demo',
    [switch]$IncludeKpi
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$RepositoryParent = Join-Path $RepositoryRoot '\..\'
$PackageName = "FAQ-Bot-$Version"
$TemporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) "$PackageName-$PID"
$StageDirectory = Join-Path $TemporaryRoot $PackageName
$DistributionDirectory = Join-Path $RepositoryParent 'dist'
$ArchivePath = Join-Path $DistributionDirectory "$PackageName.zip"

# ============================================
# KPI-DATEI DEFINITION (neu!)
# ============================================
$KpiMetricsSourceDir = Join-Path $RepositoryParent 'eval\output\'
$KpiMetricsSource = (Get-ChildItem -Path $KpiMetricsSourceDir -Filter "kpis_*.json" | Sort-Object -Property LastWriteTime -Descending | Select-Object -First 1).FullName
$KpiMetricsDest = Join-Path $StageDirectory 'app\web\dashboard\metrics.json'

Write-Host "RepositoryRoot: $RepositoryRoot"
Write-Host "RepositoryParent: $RepositoryParent"
Write-Host "KPI-Quelle: $KpiMetricsSource"
Write-Host

if (Test-Path $TemporaryRoot) {
    throw "Temporary staging path already exists: $TemporaryRoot"
}
if (Test-Path $ArchivePath) {
    throw "Package already exists: $ArchivePath. Remove it or choose another -Version."
}

New-Item -ItemType Directory -Path $StageDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $DistributionDirectory -Force | Out-Null

try {
    $ApplicationDirectory = Join-Path $RepositoryRoot 'app'
    $RuntimeExtensions = @('.py', '.html', '.css', '.js')
    Get-ChildItem -LiteralPath $ApplicationDirectory -Recurse -File |
        Where-Object {
            $RelativePath = $_.FullName.Substring($ApplicationDirectory.Length)
            $_.Extension.ToLowerInvariant() -in $RuntimeExtensions -and
            $RelativePath -notmatch '[\\/](tests|__pycache__)([\\/]|$)'
        } |
        ForEach-Object {
            $RelativePath = $_.FullName.Substring($ApplicationDirectory.Length).TrimStart('\', '/')
            $Destination = Join-Path $StageDirectory (Join-Path 'app' $RelativePath)
            New-Item -ItemType Directory -Path (Split-Path $Destination -Parent) -Force | Out-Null
            Copy-Item -LiteralPath $_.FullName -Destination $Destination
        }

    # ============================================
    # KPI-COPY
    # ============================================
    if ($IncludeKpi) {
        if (Test-Path -LiteralPath $KpiMetricsSource) {
            $KpiDashboardDir = Split-Path -Path $KpiMetricsDest -Parent
            New-Item -ItemType Directory -Path $KpiDashboardDir -Force | Out-Null
            Copy-Item -LiteralPath $KpiMetricsSource -Destination $KpiMetricsDest
            Write-Host "KPI-Daten kopiert: $KpiMetricsSource -> $KpiMetricsDest"
        }
        else {
            Write-Warning "kpi_latest.json nicht gefunden bei $KpiMetricsSource"
            Write-Warning "Dashboard wird ohne KPI-Daten ausgeliefert."
            Write-Host "Tipp: Fuehre die Evaluation aus oder verwende -IncludeKpi:`$false"
        }
    }

    $RootFiles = @(
        'requirements.txt',
        'content_config.yaml',
        '.env.example'
    )
    foreach ($RelativePath in $RootFiles) {
        $Source = Join-Path $RepositoryRoot $RelativePath
        if (-not (Test-Path $Source)) {
            throw "Required release input is missing: $RelativePath"
        }
        Copy-Item -LiteralPath $Source -Destination (Join-Path $StageDirectory $RelativePath)
    }

    $PackageData = Join-Path $StageDirectory 'data'
    New-Item -ItemType Directory -Path $PackageData -Force | Out-Null

    $EnvironmentFile = Join-Path $RepositoryRoot '.env'
    if (-not (Test-Path $EnvironmentFile)) {
        throw 'A local .env with FAQ_INPUT_FILE is required to build the demo package.'
    }

    $FaqInputLines = @(Get-Content -LiteralPath $EnvironmentFile | Where-Object { $_ -match '^\s*FAQ_INPUT_FILE\s*=' })
    if ($FaqInputLines.Count -ne 1) {
        throw 'Set exactly one FAQ_INPUT_FILE value in .env before building the demo package.'
    }

    $FaqSourceValue = ($FaqInputLines[0] -replace '^\s*FAQ_INPUT_FILE\s*=\s*', '').Trim()
    if (($FaqSourceValue.StartsWith('"') -and $FaqSourceValue.EndsWith('"')) -or
        ($FaqSourceValue.StartsWith("'") -and $FaqSourceValue.EndsWith("'"))) {
        $FaqSourceValue = $FaqSourceValue.Substring(1, $FaqSourceValue.Length - 2)
    }
    $FaqSourcePath = if ([System.IO.Path]::IsPathRooted($FaqSourceValue)) {
        $FaqSourceValue
    } else {
        Join-Path $RepositoryRoot $FaqSourceValue
    }
    if (-not (Test-Path -LiteralPath $FaqSourcePath -PathType Leaf)) {
        throw "FAQ_INPUT_FILE does not exist: $FaqSourcePath"
    }

    $csvContent = Get-Content -LiteralPath $FaqSourcePath -Encoding UTF8 -Raw
    $FaqRows = @(ConvertFrom-Csv -InputObject $csvContent -Delimiter ';')
    if ($FaqRows.Count -eq 0) {
        throw "FAQ_INPUT_FILE contains no FAQ rows: $FaqSourcePath"
    }

    $FaqColumns = @($FaqRows[0].PSObject.Properties.Name)
    $RequiredFaqColumns = @('ID', 'Frage', 'Antwort', 'Themen', 'Fragetyp', 'Schlüsselwörter')
    $MissingFaqColumns = @($RequiredFaqColumns | Where-Object { $_ -notin $FaqColumns })
    if ($MissingFaqColumns.Count -gt 0) {
        throw "FAQ_INPUT_FILE is missing required columns: $($MissingFaqColumns -join ', ')"
    }

    $CredentialRows = @($FaqRows | Where-Object { $_.Frage -match '(?i)(w[-\s]?lan|wifi).*passwort' })
    $DemoFaqRows = @($FaqRows | Where-Object { $_.Frage -notmatch '(?i)(w[-\s]?lan|wifi).*passwort' })
    if ($CredentialRows.Count -gt 0) {
        Write-Host "Excluded $($CredentialRows.Count) FAQ row(s) asking for the actual WLAN password."
    }
    if ($DemoFaqRows.Count -eq 0) {
        throw 'No FAQ rows remain after excluding credential-related entries.'
    }

    $DemoFaqRows |
        Select-Object $RequiredFaqColumns |
        Export-Csv -LiteralPath (Join-Path $PackageData 'faq.csv') -Delimiter ';' -NoTypeInformation -Encoding UTF8

    Copy-Item -LiteralPath (Join-Path $RepositoryRoot 'deployment\demo\setup.ps1') -Destination (Join-Path $StageDirectory 'setup.ps1')
    Copy-Item -LiteralPath (Join-Path $RepositoryRoot 'deployment\demo\start.ps1') -Destination (Join-Path $StageDirectory 'start.ps1')
    Copy-Item -LiteralPath (Join-Path $RepositoryRoot 'deployment\demo\README.md') -Destination (Join-Path $StageDirectory 'README.md')

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $StageDirectory,
        $ArchivePath,
        [System.IO.Compression.CompressionLevel]::Optimal,
        $false
    )

    Write-Host "Demo package created: $ArchivePath"
}
finally {
    Remove-Item -LiteralPath $TemporaryRoot -Recurse -Force
}