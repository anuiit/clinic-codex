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
