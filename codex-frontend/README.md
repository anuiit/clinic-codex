# Codex Glyph Analyzer Frontend

The Codex Glyph Analyzer is a web interface for uploading glyph images, viewing segmentation results, and performing classification/annotation of segmented elements.

## Features

- **Upload**: Upload glyph images for analysis.
- **Dashboard**: View history of processed glyphs.
- **Detail View**: Inspect segmentation results for a specific glyph.
- **Annotation**: Classify and annotate segmented elements.

## Prerequisites

- **Node.js**: Version 18 or higher is required.
- **Backend API**: A compatible backend must be running. By default, the app expects the API at `http://localhost:5000`.
  - A demo backend is available at `frontend_integration_fix/frontend_integration/examples/flask_api.py`.

## Installation

Install the project dependencies using npm:

```bash
npm install
```

## Environment Variables

The frontend uses the following environment variable to communicate with the backend:

- `VITE_API_BASE_URL`: The base URL of the backend API.
  - Default: `http://localhost:5000`
  - Ensure this is set correctly if your backend is running on a different port or host.

## Development

Start the development server with live reload:

```bash
npm run dev
```

## Production

Build the application for production:

```bash
npm run build
```

The output will be generated in the `dist` directory.

## Linting

Run ESLint to check for code quality and style issues:

```bash
npm run lint
```
