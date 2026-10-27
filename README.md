# Senior Citizen Identification System

## Project Overview

The Senior Citizen Identification System is a computer vision project that detects faces in webcam streams and video files, estimates age and gender using a custom convolutional neural network (CNN), identifies predictions above the age of 60, and records detection events in CSV and Excel files.

## Features

- UTKFace dataset preparation and deduplication.
- Age regression and gender classification using a custom CNN.
- Face detection using OpenCV Haar cascades.
- Multi-face detection in video and webcam streams.
- Senior citizen status based on predicted age greater than 60.
- CSV and Excel detection logs with date, time, predicted age, gender, status, and track ID.
- Training, validation, and held-out test evaluation.

## Technologies

- Python
- PyTorch and Torchvision
- OpenCV
- Pandas
- Scikit-learn
- OpenPyXL

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

Download UTKFace and place the image folders in the project root:

- `crop_part1/`
- `UTKFace/`
- `utkface_aligned_cropped/`

The preparation script expects filenames following the UTKFace format:

`age_gender_race_datetime.jpg`

Gender labels: `0 = Male`, `1 = Female`.

## Prepare and Train

```bash
python -m src.prepare_dataset
python -m src.train_model
```

The prepared dataset splits are saved in `dataset/`. The trained model is saved as `models/age_gender_model.pth`.

## Run Webcam Detection

```bash
python main.py
```

Select option 3, or run:

```bash
python -m src.detect_video --source 0
```

Press **Q** to close the webcam window.

## Run Video Detection

```bash
python -m src.detect_video --source "path/to/video.mp4"
```

## Output Files

- `outputs/visits.csv`
- `outputs/visits.xlsx`
- `models/training_history.json`

The training history contains validation results and held-out test metrics, including age mean absolute error and gender accuracy.

## Important Limitations

Age and gender are estimates from facial images. Predictions may be inaccurate due to lighting, camera angle, image quality, and dataset bias. A predicted age above 60 does not verify a person's actual age.

The simple face tracker does not reliably identify unique individuals across long periods, occlusions, or camera changes. The logs should therefore be interpreted as detection events, not confirmed unique visitor counts.

For real deployments, obtain appropriate authorization, provide notice, minimize retained data, and evaluate performance across different populations and conditions. Do not use predictions as the sole basis for decisions affecting individuals.

## Future Improvements

- Replace Haar cascades with a modern face detector.
- Add a GUI using Streamlit.
- Improve tracking and visitor-counting accuracy.
- Evaluate senior/non-senior classification separately from age regression.
- Add confusion matrices, error analysis, and model confidence calibration.
