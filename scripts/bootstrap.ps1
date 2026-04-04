param(
  [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment..."
& $PythonExe -m venv .venv

Write-Host "Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host "Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Running unit tests..."
python -m pytest -q

Write-Host "Bootstrap complete."
Write-Host "Next: place raw data files under data/raw and run scripts in README."
