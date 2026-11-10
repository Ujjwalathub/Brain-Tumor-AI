# BrainTumor AI

A simple AI project that helps look at brain MRI scans and guess whether a tumor may be present, and if so, what kind of tumor it looks like.

This project brings together:
- a smart image model trained on brain scans
- a small backend API that processes the scan
- a modern frontend website for users to upload and test scans
- model explanations that highlight the suspicious area on the image

The system is designed to help with early screening and learning, not to replace a doctor.

## What this project does

The app looks at MRI images and classifies them into four groups:
- Glioma
- Meningioma
- Pituitary tumor
- No tumor

It is meant for research, educational use, and local demonstration. It is helpful for understanding how AI can assist in medical image analysis, especially when a scan is reviewed by a specialist.

## Why this project exists

Brain tumors can be hard to spot early, and doctors often need to review many scans carefully. This project is an example of how AI can help by quickly looking at a scan and giving a first opinion.

The goal is not to make a final diagnosis. The real goal is to:
- make the process faster for learning and research
- help people understand how AI can support medical image analysis
- provide a working demo that can run locally on a computer
- show model results and explainability in a friendly interface

## Project overview

This repository contains a complete local project with three main parts:

1. AI model engine
   - Python backend
   - deep learning model for MRI image classification
   - runs on a GPU when available
   - can also work in a lighter way on CPU

2. API layer
   - FastAPI backend
   - accepts uploaded MRI images
   - returns a class prediction, confidence score, and probability breakdown
   - supports Grad-CAM-style visual explanation

3. Frontend app
   - simple website built with React and Vite
   - lets users upload an MRI image
   - shows prediction results and helpful visuals
   - includes pages for overview, scanner, model performance, and early signs

## What makes it useful

- Reads MRI images in a simple upload flow
- Rejects color images because this pipeline expects grayscale medical scan inputs
- Shows confidence levels for each class
- Highlights important brain regions using heatmap-style visuals
- Includes result pages for explanation and model evaluation
- Keeps everything local for easy testing and development

## Current model status

The project has been trained and evaluated on brain MRI data and reports strong results in testing:

- Accuracy: around 96.19%
- Macro recall: around 96.19%
- Inference speed: very fast on GPU, often around a few milliseconds per image
- Supported classes: glioma, meningioma, pituitary, no tumor

These numbers show the model can perform well in a research setting, but they do not mean the system is ready for real medical diagnosis.

## Important safety note

This project is for learning, research, and demonstration.

It is not a medical device and should not be used as a substitute for:
- a qualified doctor
- a radiologist
- clinical testing
- patient evaluation

If someone has concerning symptoms, they should speak with a medical professional.

## Main features

### Simple image-based test flow
Users can upload a brain MRI image and get a result such as:
- likely glioma
- likely meningioma
- likely pituitary tumor
- no tumor detected

### Visual explanation
The project includes explainability tools that show where the model focused its attention. This helps users understand which area of the image influenced the prediction.

### Frontend experience
The frontend includes pages for:
- project overview
- tumor detector workspace
- model performance report
- early symptoms and patient education

### Local development focus
Everything is designed to run locally on a machine for experimentation and demos.

## Project structure

```text
Brain/
├── backend/                 # AI engine and backend API
│   ├── api/                 # FastAPI routes and schemas
│   ├── configs/             # model setup files
│   ├── data/                # processed data and dataset cache
│   ├── models/              # saved model checkpoints and ONNX exports
│   ├── reports/             # metrics, charts, confusion matrix, etc.
│   ├── src/                 # training, evaluation, explainability code
│   ├── tests/               # backend tests
│   ├── .env.example         # sample environment setup
│   ├── pyproject.toml       # Python project config
│   ├── requirements.txt     # backend dependencies
│   └── ...
├── frontend/                # website and UI
│   ├── src/                 # app pages and components
│   ├── public/              # static files
│   ├── package.json         # frontend dependencies
│   └── ...
├── design.md                # UI and design direction
├── README.md                # this file
├── ...
```

## How the system works in plain language

The project follows a simple pattern:

1. A user uploads a brain MRI image
2. The backend checks that the file is valid
3. The image is cleaned and prepared for the model
4. The model looks at the scan and scores each class
5. The app picks the class with the highest score
6. The system shows the answer and a confidence score
7. If enabled, the app also shows a heatmap to highlight the suspicious area

This is a classic AI workflow: input image -> model analysis -> result -> explanation.

## Requirements

### For the backend
- Python 3.11
- pip or uv
- a GPU is optional but recommended for speed
- access to MRI image files for testing

### For the frontend
- Node.js
- npm
- a browser for local testing

## Quick start

### 1. Open a terminal and go to the project

```bash
cd E:\Brain
```

### 2. Set up the backend

From the project root:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you want to copy the environment template:

```bash
copy .env.example .env
```

### 3. Start the backend API

```bash
cd backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- http://localhost:8000
- API docs: http://localhost:8000/docs

### 4. Set up the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Then open the local frontend page in your browser, usually:
- http://localhost:5173

## Using the app

### Uploading a scan
- Open the frontend
- go to the detector page
- upload a valid MRI image
- wait for the AI result

### What you will see
- predicted class
- confidence percentage
- probability breakdown for each class
- heatmap explanation if available
- a plain-language summary and result card

### Good image quality matters
For best results:
- use a grayscale MRI scan
- use a clean image without extra color filters
- avoid blurry or heavily edited pictures
- confirm the image is readable and not distorted

## Frontend pages

The frontend is more than just a prediction box. It includes:

### Home page
A friendly landing page explaining the project and why it matters.

### Detector page
This is the main image-upload area. It lets users run predictions and view model output.

### Performance page
Shows evaluation results, charts, and model metrics.

### Early signs page
Explains patient education and how symptoms can appear in early stages.

## Model and evaluation summary

The project includes training and evaluation code for several model families such as:
- EfficientNet-B0
- ResNet50
- ConvNeXt-Tiny

The training pipeline supports:
- image preprocessing
- data augmentation
- checkpoint saving
- validation tracking
- confusion matrix generation
- Grad-CAM visual explanations
- ONNX export for faster runtime inference

The project also keeps reports and results in the backend reports folder, including metrics and charts.

## Backend API details

The backend API provides a few main things:
- health check
- image upload prediction endpoint
- explanation endpoint
- model loading and validation

The primary prediction endpoint takes an image and returns:
- predicted label
- confidence value
- per-class probabilities
- processing time

## Common development notes

- This is a local-first project, not a cloud product.
- Model behavior is best when used with proper grayscale MRI scans.
- The frontend can still show demo results if the backend is temporarily offline.
- The backend rejects color images to keep the system aligned with the medical image use case.

## Troubleshooting

### Backend not starting
Check that:
- Python 3.11 is installed
- dependencies were installed correctly
- the virtual environment is activated
- the correct port is free

### Frontend not loading
Check that:
- Node.js is installed
- npm install completed successfully
- the dev server is running in the frontend folder

### Model not returning predictions
Possible causes:
- backend not running
- ONNX model missing
- model path not configured
- image is invalid or in the wrong format

### Color image rejected
This project expects MRI-style grayscale inputs. If a color image is uploaded, the backend will reject it.

## Recommended workflow

If you are exploring the project for the first time, this is the best path:

1. Start the backend API
2. Confirm the backend health endpoint responds
3. Start the frontend
4. Upload a test MRI image
5. Review the prediction and explanation output
6. Check the statistics and reports pages for performance context

## Final note

This project is a strong example of how AI can be used to support medical imaging workflows. It is designed to be clear, useful, and easy to explore locally.

It brings together real machine learning, backend services, frontend design, and model explainability in one workspace.

If you want to learn from it, experiment with it, or extend it, this repository is a great starting point.

## License and project context

This project is built for research, education, and local experimentation. It is not a medical diagnosis tool and should always be reviewed with human expertise.

The codebase includes training, inference, evaluation, deployment, and frontend visualization pieces that together form a complete small AI system.

---

For a more technical view of the project, see the design notes and backend source files in this repository.
