<#
.SYNOPSIS
  Deploy CodeFix AI (web app + ADK/CrewAI agents + A2A + MCP) to Google Cloud Run.

.DESCRIPTION
  Builds the Dockerfile with Cloud Build (no local Docker needed), stores GEMINI_API_KEY from
  .env in Secret Manager (never in the image or in plain env vars), deploys the service and then
  sets CODEFIX_PUBLIC_URL so the A2A agent card advertises the real URL.

  Prerequisites (once):  gcloud auth login ; a project with billing enabled.

.EXAMPLE
  ./deploy/deploy_cloudrun.ps1 -Project my-gcp-project            # private (needs a Google login to open)
  ./deploy/deploy_cloudrun.ps1 -Project my-gcp-project -Public    # anyone with the URL can use it
#>
param(
    [Parameter(Mandatory = $true)][string]$Project,
    [string]$Region = "asia-south1",
    [string]$Service = "codefix-ai",
    [switch]$Public
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Run([string[]]$cmdArgs) {
    & gcloud @cmdArgs
    if ($LASTEXITCODE -ne 0) { throw "gcloud $($cmdArgs -join ' ') failed" }
}

Run @("config", "set", "project", $Project)
Run @("services", "enable", "run.googleapis.com", "cloudbuild.googleapis.com", "artifactregistry.googleapis.com",
      "secretmanager.googleapis.com")

# --- GEMINI_API_KEY -> Secret Manager (read from .env, never printed) -------------------------
$line = Get-Content .env | Where-Object { $_ -match '^GEMINI_API_KEY=(.+)$' } | Select-Object -First 1
if (-not $line) { throw "GEMINI_API_KEY is not set in .env" }
$tmp = New-TemporaryFile
try {
    [IO.File]::WriteAllText($tmp, ($line -replace '^GEMINI_API_KEY=', '').Trim())
    & gcloud secrets describe gemini-api-key *> $null
    if ($LASTEXITCODE -ne 0) {
        Run @("secrets", "create", "gemini-api-key", "--replication-policy=automatic", "--data-file=$tmp")
    } else {
        Run @("secrets", "versions", "add", "gemini-api-key", "--data-file=$tmp")
    }
} finally { Remove-Item $tmp -Force }

$number = (& gcloud projects describe $Project --format "value(projectNumber)").Trim()
$runtimeSa = "$number-compute@developer.gserviceaccount.com"
Run @("secrets", "add-iam-policy-binding", "gemini-api-key", "--member=serviceAccount:$runtimeSa",
      "--role=roles/secretmanager.secretAccessor", "--quiet")

# --- build + deploy -------------------------------------------------------------------------------
$auth = if ($Public) { "--allow-unauthenticated" } else { "--no-allow-unauthenticated" }
Run @("run", "deploy", $Service, "--source", ".", "--region", $Region, $auth,
      "--memory", "4Gi", "--cpu", "2", "--timeout", "3600", "--concurrency", "4", "--max-instances", "2",
      "--set-secrets", "GEMINI_API_KEY=gemini-api-key:latest", "--quiet")

$url = (& gcloud run services describe $Service --region $Region --format "value(status.url)").Trim()
Run @("run", "services", "update", $Service, "--region", $Region, "--update-env-vars", "CODEFIX_PUBLIC_URL=$url",
      "--quiet")

Write-Host ""
Write-Host "Deployed: $url"
Write-Host "  web app        $url/"
Write-Host "  A2A agent card $url/a2a/crew/.well-known/agent-card.json"
if (-not $Public) {
    Write-Host "  (private service: open it through  gcloud run services proxy $Service --region $Region  -> http://localhost:8080)"
}
