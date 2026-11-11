# Justice Dance analysis layer

This module mirrors the AIHuMo conventions found on the computer:

- 33 MediaPipe pose landmarks per person
- normalized landmark coordinates centered at the shoulder midpoint
- shoulder-width scaling so body size and camera distance are comparable
- world coordinates when available
- joint angles, normalized distances, velocity, and acceleration
- one Parquet feature table per player
- matching score based on normalized landmark, angle, and movement-speed error

The browser UI is intentionally lightweight. A production session should run
MediaPipe extraction for the uploaded reference and the live camera stream,
call `build_features` for each person, and send the resulting feature frames to
`score`. The reference and each player remain separate datasets.
