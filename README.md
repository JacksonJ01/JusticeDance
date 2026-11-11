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
- `.gitignore` — common local files excluded from Git

## Analysis scope

The browser prototype uses a playful placeholder score. The Python analysis layer is ready for the real extraction pipeline: MediaPipe pose landmarks are centered at the shoulder midpoint, scaled by shoulder width, enriched with angles/distances/velocity/acceleration, stored as one Parquet dataset per player, and compared against the preprocessed reference stream. Install `requirements-analysis.txt` to use it.
