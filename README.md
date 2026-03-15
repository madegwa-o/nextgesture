# GestureBoard (Next.js)

A Next.js rebuild of GestureBoard using browser-side **MediaPipe Hands** for hand tracking and **OpenCV.js** for motion analysis.

## Features

- Real-time hand tracking (up to 2 hands) with MediaPipe.
- Gesture-controlled drawing, finalize, pan, zoom, clear, and undo.
- Canvas-based whiteboard with world/screen transform logic.
- Save PNG snapshots of the drawing board.
- HUD with live gesture state, hand count, polygon count, zoom, and OpenCV motion score.

## Controls

- **Point**: draw.
- **Peace**: finalize polygon.
- **Open hand**: pan.
- **Two open/pinch hands**: zoom.
- **Fist**: clear all.
- **Rock-on**: undo last polygon.

## Run

```bash
npm install
npm run dev
```

Open `http://localhost:3000` and allow camera permission.

## Tech notes

- MediaPipe scripts are loaded from jsDelivr at runtime.
- OpenCV.js is loaded from docs.opencv.org and used to compute frame-difference motion score.
- Everything runs client-side in the browser.
