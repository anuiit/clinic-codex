# Installation Guide

Welcome! You don't need to be a programmer to set up the Clinic Codex annotation tool. This guide will help you get everything running on your computer.

## Prerequisites

Before we start, you'll need three tools installed on your computer. They are standard pieces of software used by many applications:

1. **Git**: Used to manage the project files.
2. **Python (3.10 or newer)**: The engine that runs our AI models.
3. **Node.js (18 or newer)**: The engine that runs the web interface.

## Step-by-Step Installation

### For Mac and Linux Users

1. **Open your Terminal**: You can find this in your Applications folder or by searching for "Terminal".
2. **Navigate to the project folder**: Use the `cd` command to enter the directory where you downloaded this project.
3. **Run the installer**: Type the following and press Enter:
   ```bash
   bash scripts/install.sh
   ```
4. **Download AI models**: These are large files needed for the tool to work. Run:
   ```bash
   bash scripts/download-weights.sh
   ```

### For Windows Users (Best-effort)

1. **Open PowerShell**: Search for "PowerShell" in your Start menu.
2. **Navigate to the project folder**.
3. **Run the installer**:
   ```powershell
   .\scripts\install.ps1
   ```
4. **Download AI models**:
   ```bash
   bash scripts/download-weights.sh
   ```

## Manual Launch (Primary Path — WSL2 / Linux)

This section is for researchers who want to understand the installation step-by-step or need to debug their environment. These commands perform the same actions as the automated scripts but give you more control.

### 1. Set Up the Python Environment
We use a virtual environment to keep the project's tools separate from your system.
```bash
python3.11 -m venv backend/.venv
```

### 2. Install Core Utilities
These are the basic tools needed for data processing and progress tracking.
```bash
backend/.venv/bin/pip install --no-cache-dir --prefer-binary numpy pillow pyyaml scipy tqdm
```

### 3. Install Web Framework
This installs the software that lets the backend talk to your web browser.
```bash
backend/.venv/bin/pip install --no-cache-dir --prefer-binary flask flask-cors
```

### 4. Install AI Engine (CPU Version)
This installs the machine learning engine. We use the CPU-only version to keep the download small and avoid issues with specialized graphics hardware.
```bash
backend/.venv/bin/pip install --no-cache-dir --prefer-binary --extra-index-url https://download.pytorch.org/whl/cpu "torch>=2.1,<2.6" "torchvision>=0.16,<0.21"
```

### 5. Install Image Analysis Tools
These are the specific AI models used for identifying glyphs in historical documents.
```bash
backend/.venv/bin/pip install --no-cache-dir --prefer-binary "segment-anything==1.0" "mobile-sam==1.0" "albumentations>=1.4,<2.0" "timm>=0.9"
```

### 6. Set Up the Web Interface
This installs the files needed for the visual part of the tool.
```bash
cd frontend && npm install && cd ..
```

### 7. Start the Backend Server
This starts the AI engine on port 7117. The `&` at the end lets it run while you type the next command.
```bash
PORT=7117 backend/.venv/bin/python backend/examples/flask_api.py &
```

### 8. Start the Web Interface
This starts the visual interface on port 7118.
```bash
cd frontend && PORT=7118 npm run dev
```

### 9. Verify and Open
To make sure the backend is working, you can check it with this command:
```bash
curl http://localhost:7117/classes
```
Then, open your web browser and go to:
`http://localhost:7118`

## How to Run the Tool

Once installation is finished, you can start the application whenever you want to use it:

1. In your Terminal or PowerShell, run:
   ```bash
   bash scripts/run-dev.sh
   ```
2. Wait for the message saying the servers have started.
3. Open your web browser (like Chrome or Firefox) and go to:
   `http://localhost:7118`

## How to Stop

To stop the tool, go back to your Terminal or PowerShell window and press **Ctrl + C** on your keyboard. This will safely shut down the servers.

## Troubleshooting

- **Python or Node not found**: Ensure you've installed them from their official websites and restarted your Terminal.
- **Port already in use**: This usually means the tool is already running in another window. Close that window or stop the process.
- **Weights download fails**: Check your internet connection. These files are several hundred megabytes in size.
