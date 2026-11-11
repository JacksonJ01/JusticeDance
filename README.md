# Justice Dance

A humorous, dependency-free Just Dance-style browser prototype. Players upload a reference dance video, optionally enable a live camera overlay, and receive a local score. The repository also includes an analysis layer aligned to the AIHuMo pose/feature conventions.

## Run locally

Open `index.html` in a browser. Camera access works on localhost or HTTPS.

## Project files

- `index.html` — upload flow, dance room, scoreboard, and instructions
- `styles.css` — responsive game UI
- `script.js` — local video preview, camera feed, timer, and prototype scoring
- `analysis/justice_analysis.py` — normalized landmarks, angles, distances, velocity, acceleration, Parquet output, and player scoring
- `analysis/README.md` — AIHuMo alignment notes
- `app/justice_dance.py` — PyQt6 desktop stage with large reference video and live camera picture-in-picture
- `.gitignore` — common local files excluded from Git

## Analysis scope

The browser prototype uses a playful placeholder score. The Python analysis layer is ready for the real extraction pipeline: MediaPipe pose landmarks are centered at the shoulder midpoint, scaled by shoulder width, enriched with angles/distances/velocity/acceleration, stored as one Parquet dataset per player, and compared against the preprocessed reference stream. Install `requirements-analysis.txt` to use it.

## PyQt6 desktop app

Install `requirements-pyqt6.txt`, ensure a MediaPipe Pose Landmarker `.task` model is available, and run:

```text
python app/justice_dance.py
```

Choose the downloaded reference video and model, start the camera, then start and stop the dance. The app saves `data/justice_sessions/player_1/` or `player_2/` landmark and feature Parquet files. The reference occupies the main frame and the live camera is composited in the lower-left picture-in-picture region.
