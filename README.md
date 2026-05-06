# Clinic Codex

Clinic Codex is an annotation tool for Nahuatl glyphs (codices). It uses machine learning to segment and classify individual elements within complex historical glyphs, helping researchers identify archetypes and analyze historical manuscripts.

---

## For Researchers (Non-Technical)

### What it does
- **Analyze codices**: Upload images of historical Nahuatl manuscripts.
- **Auto-segmentation**: Automatically identify individual glyph elements using AI.
- **Similarity search**: Find historical archetypes similar to a selected glyph element.
- **Collaborative annotation**: Mark and identify elements to build a shared dataset.

### How to install
Please refer to the [INSTALL.md](INSTALL.md) file for step-by-step instructions on setting up the tool on your computer.

### How to run
Once installed, you can start both the backend and frontend with a single command:
```bash
bash scripts/run-dev.sh
```
This will launch the application in your web browser.

### Screenshots
*Placeholders for screenshots showing the annotation interface and similarity results.*

---

## For Developers (Technical)

### Architecture Overview
The project follows a decoupled client-server architecture:
- **Backend**: Flask API serving ML models (DINOv2-ViT-S/14). Runs on port `7117`.
- **Frontend**: React application built with Vite and Tailwind CSS. Runs on port `7118`.

### Environment Variables
Configure the following in your `.env` file (see `.env.example` if available):
- `PORT=7117`: Backend port.
- `HOST=0.0.0.0`: Server host.
- `CORS_ORIGINS`: Allowed origins for cross-domain requests.
- `MODEL_DIR`: Path to the machine learning model weights.
- `VITE_API_BASE_URL=http://localhost:7117`: Frontend setting to point to the backend API.

### Project Structure
- `backend/`: Python Flask server and ML pipeline (`backend/examples/flask_api.py`).
- `frontend/`: React + Vite frontend source code.
- `scripts/`: Production and utility scripts (e.g., `run-dev.sh`).
- `dev-scripts/`: Development-only automation and helper scripts.
- `_legacy/`: Archived previous implementations.

### Development Setup
1. Follow [INSTALL.md](INSTALL.md) to set up the Python and Node environments.
2. Launch the development stack:
   ```bash
   bash scripts/run-dev.sh
   ```

### Contributing
Please ensure you follow the established project structure and keep the `_legacy/` directory clean by archiving deprecated modules there.
