# Debug script for inventory system
$ErrorActionPreference = "Stop"

# Activate virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

Write-Host "Activating virtual environment..."
.\.venv\Scripts\Activate.ps1

# Install/update requirements
Write-Host "Installing/updating requirements..."
pip install -r requirements.txt
pip install -e .

# Create necessary directories
Write-Host "Creating necessary directories..."
New-Item -ItemType Directory -Force -Path "output", "models", "logs", "data" | Out-Null

# Run the CLI with debug logging
Write-Host "Starting inventory system in debug mode..."
Write-Host "Press 'r' to run, 'q' to quit, or any other key to continue"
Write-Host ""

while ($true) {
    $choice = Read-Host "Enter your choice"
    
    if ($choice -eq 'q') {
        Write-Host "Exiting..."
        break
    }
    elseif ($choice -eq 'r') {
        try {
            Write-Host "Running inventory system..."
            python -m inventory_system.cli run --data-path "data/input.xlsx" --verbose
        }
        catch {
            Write-Host "Error: $_"
        }
    }
    else {
        Write-Host "Continuing..."
    }
} 