param(
    [string]$DocxPath = "",
    [string]$PdfPath = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not $DocxPath) {
    $DocxPath = Join-Path $root "reports\Informe_Tecnico_Arboles_Ensambles.docx"
}
if (-not $PdfPath) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($DocxPath) + ".pdf"
    $PdfPath = Join-Path (Split-Path -Parent $DocxPath) $name
}

$docx = (Resolve-Path -LiteralPath $DocxPath).Path
$pdfDir = Split-Path -Parent $PdfPath
if (-not (Test-Path -LiteralPath $pdfDir)) {
    New-Item -ItemType Directory -Path $pdfDir -Force | Out-Null
}
$pdf = [System.IO.Path]::GetFullPath($PdfPath)

$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0

    $doc = $word.Documents.Open($docx, $false, $false)

    # Update PAGE/NUMPAGES and the TOC before saving and exporting.
    $doc.Fields.Update() | Out-Null
    for ($i = 1; $i -le $doc.TablesOfContents.Count; $i++) {
        $doc.TablesOfContents.Item($i).Update()
    }
    if ($doc.TablesOfContents.Count -gt 0) {
        $tocRange = $doc.TablesOfContents.Item(1).Range
        $tocRange.Font.Size = 9.5
        $tocRange.ParagraphFormat.SpaceBefore = 0
        $tocRange.ParagraphFormat.SpaceAfter = 0
        $tocRange.ParagraphFormat.LineSpacingRule = 0
    }
    $doc.Repaginate()
    $doc.Save()

    $doc.ExportAsFixedFormat($pdf, 17, $false)
    $pages = $doc.ComputeStatistics(2)

    Write-Output "DOCX: $docx"
    Write-Output "PDF:  $pdf"
    Write-Output "Pages: $pages"
}
finally {
    if ($null -ne $doc) {
        $doc.Close($false)
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc) | Out-Null
    }
    if ($null -ne $word) {
        $word.Quit()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
