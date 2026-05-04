# Codex Glyph Analyzer

The Codex Glyph Analyzer is an ML-powered system for the segmentation and classification of Nahuatl and Codex glyphs. It identifies individual elements within complex glyphs and classifies them against a known set of historical archetypes.

## Project Structure

- **`codex-frontend/`**: A React application built with Vite and TypeScript. It provides a web interface for uploading images, viewing segmentation results, and annotating glyph elements.
- **`frontend_integration_fix/frontend_integration/`**: The backend and AI integration package. It includes the Python-based ML models (classification and segmentation), a Flask API for integration, and sample data.

## Prerequisites

- **Node.js**: 18.0.0 or higher
- **Python**: 3.8 or higher
- **Model weights**: Required `.pt` artifacts must be present in `frontend_integration_fix/frontend_integration/`.

## Getting Started

### Backend Setup

The backend serves the ML models via a REST API.

1.  Navigate to the integration directory:
    ```bash
    cd frontend_integration_fix/frontend_integration
    ```
2.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Flask API:
    ```bash
    python examples/flask_api.py
    ```

The backend starts at `http://localhost:5000`.

### Frontend Setup

The frontend provides the user interface for interacting with the backend.

1.  Navigate to the frontend directory:
    ```bash
    cd codex-frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```

The frontend application will be available at the URL shown in your terminal (typically `http://localhost:5173`).

## Smoke Checks

### Backend Smoke Check

To verify the backend is running and can process images, run the following command while the API is active:

```bash
curl -s -X POST -F "image=@frontend_integration_fix/frontend_integration/data/glyphs_sample/atl-glyph/026r_a_07-2.jpg" http://localhost:5000/segment
```

A successful response returns a JSON object containing `num_elements`, `image_size`, and an `elements` array.

### Frontend Smoke Check

Run the following commands in the `codex-frontend` directory:

```bash
npm ci && npm run build && npm run lint
```

All commands must exit with code 0 for the smoke check to pass.

## Model Artifacts

This project relies on several pretrained model weights and prototypes:

- **Classification**: `codex_model/weights/prototypes.pt`, `codex_model/weights/projection.pt`, `prototypes/prototypes.pt`
- **Segmentation**: `mobile_sam.pt` (used by the MobileSAM wrapper)

These files are treated as external assets and must be present in the `frontend_integration_fix/frontend_integration/` directory. Specifically, `mobile_sam.pt` may need to be downloaded separately if not already present in the expected location (defaulting to `~/.cache/mobile_sam/mobile_sam.pt`).

## Additional Documentation

- [Frontend README](codex-frontend/README.md)
- [Backend/Integration README](frontend_integration_fix/frontend_integration/README.md)
