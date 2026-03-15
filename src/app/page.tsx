"use client";

import { useEffect, useRef, useState, useCallback } from "react";

type Point = { x: number; y: number };
type Gesture = "unknown" | "point" | "peace" | "open" | "fist" | "pinch" | "rockon";

type HandState = {
  landmarks: Point[];
  gesture: Gesture;
  confirmed: Gesture;
  wrist: Point;
  indexTip: Point;
  handedness: "Left" | "Right";
};

type Polygon = {
  id: number;
  color: string;
  points: Point[];
};

type CVRuntime = {
  matFromImageData: (imageData: ImageData) => CVMat;
  Mat: new () => CVMat;
  absdiff: (src1: CVMat, src2: CVMat, dst: CVMat) => void;
  cvtColor: (src: CVMat, dst: CVMat, code: number, dstCn?: number) => void;
  mean: (src: CVMat) => number[];
  COLOR_RGBA2GRAY: number;
};

type CVMat = {
  clone: () => CVMat;
  delete: () => void;
};

type MediaPipeHandedness = {
  label: "Left" | "Right";
  score: number;
};

declare global {
  interface Window {
    Hands: new (config: { locateFile: (file: string) => string }) => {
      setOptions: (options: Record<string, number>) => void;
      onResults: (
        cb: (results: {
          multiHandLandmarks?: Point[][];
          multiHandedness?: MediaPipeHandedness[];
        }) => void
      ) => void;
      send: (data: { image: HTMLVideoElement }) => Promise<void>;
    };
    Camera: new (
      video: HTMLVideoElement,
      options: { onFrame: () => Promise<void>; width: number; height: number }
    ) => { start: () => Promise<void>; stop: () => void };
    cv: CVRuntime;
  }
}

const colors = ["#ff8a00", "#55d6ff", "#89ff65", "#d899ff", "#ffe066"];

const loadScript = (src: string) =>
  new Promise<void>((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) {
      resolve();
      return;
    }
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.body.appendChild(script);
  });

const dist = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y);

// FIX 3: Smooth new drawing points to reduce jitter
const smoothPoint = (draft: Point[], newPt: Point): Point => {
  if (draft.length === 0) return newPt;
  const last = draft[draft.length - 1];
  return { x: last.x * 0.5 + newPt.x * 0.5, y: last.y * 0.5 + newPt.y * 0.5 };
};

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [status, setStatus] = useState("Loading MediaPipe + OpenCV...");
  const [gestureLabel, setGestureLabel] = useState("unknown");
  const [handsCount, setHandsCount] = useState(0);
  const [polygonCount, setPolygonCount] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [motionScore, setMotionScore] = useState(0);

  const handsRef = useRef<HandState[]>([]);
  const holdRef = useRef<
    Map<number, { gesture: Gesture; hold: number; confirmed: Gesture }>
  >(new Map());
  const polygonsRef = useRef<Polygon[]>([]);
  const draftRef = useRef<Point[]>([]);
  const undoRef = useRef<Polygon[][]>([]);
  const modeRef = useRef<"passive" | "drawing" | "panning" | "zooming">("passive");
  const transformRef = useRef({ scale: 1, tx: 0, ty: 0 });
  const actionLockRef = useRef({ erase: false, undo: false });
  const prevPinchDistRef = useRef<number | null>(null);
  const prevPanAnchorRef = useRef<Point | null>(null);
  const idRef = useRef(1);
  const lastGrayRef = useRef<CVMat | null>(null);
  // FIX 5: Throttle OpenCV motion scoring
  const lastMotionTimeRef = useRef(0);
  // FIX 8: Track RAF id for visibility-change pause/resume
  const rafRef = useRef(0);
  const renderingRef = useRef(false);

  // FIX 7: HiDPI-aware canvas sizing
  const updateCanvasSize = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const w = window.innerWidth;
    const h = window.innerHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    const ctx = canvas.getContext("2d");
    if (ctx) ctx.scale(dpr, dpr);
  }, []);

  useEffect(() => {
    let camera: { start: () => Promise<void>; stop: () => void } | null = null;

    // FIX 1 & 2: Hand classification with handedness awareness and finger thresholds
    const classify = (lm: Point[], handedness: "Left" | "Right"): Gesture => {
      // Add a small y-threshold to reduce false positives from near-horizontal fingers
      const fingerUp = (tip: number, pip: number) =>
        lm[tip].y < lm[pip].y - 0.02;

      const indexUp = fingerUp(8, 6);
      const middleUp = fingerUp(12, 10);
      const ringUp = fingerUp(16, 14);
      const pinkyUp = fingerUp(20, 18);

      // FIX 1: Thumb direction depends on which hand it is
      const thumbUp =
        handedness === "Left" ? lm[4].x > lm[3].x : lm[4].x < lm[3].x;

      const pinchDist = dist(lm[4], lm[8]);
      if (pinchDist < 0.06) return "pinch";
      if (indexUp && !middleUp && !ringUp && !pinkyUp) return "point";
      if (indexUp && middleUp && !ringUp && !pinkyUp) return "peace";
      if (indexUp && pinkyUp && !middleUp && !ringUp) return "rockon";
      if (thumbUp && indexUp && middleUp && ringUp && pinkyUp) return "open";
      if (!indexUp && !middleUp && !ringUp && !pinkyUp) return "fist";
      return "unknown";
    };

    const confirmGesture = (idx: number, gesture: Gesture): Gesture => {
      const current = holdRef.current.get(idx) ?? {
        gesture: "unknown" as Gesture,
        hold: 0,
        confirmed: "unknown" as Gesture,
      };
      if (current.gesture === gesture) {
        current.hold = Math.min(10, current.hold + 1);
      } else {
        current.hold = Math.max(0, current.hold - 2);
        if (current.hold === 0) current.gesture = gesture;
      }
      if (current.hold >= 10) current.confirmed = current.gesture;
      holdRef.current.set(idx, current);
      return current.confirmed;
    };

    const toScreen = (p: Point) => {
      const { scale, tx, ty } = transformRef.current;
      return { x: p.x * scale + tx, y: p.y * scale + ty };
    };

    const toWorld = (p: Point) => {
      const { scale, tx, ty } = transformRef.current;
      return { x: (p.x - tx) / scale, y: (p.y - ty) / scale };
    };

    const processGestures = (canvas: HTMLCanvasElement) => {
      const hands = handsRef.current;

      if (!hands.length) {
        // Finalize any in-progress draft when hands disappear
        if (draftRef.current.length >= 3) {
          polygonsRef.current.push({
            id: idRef.current++,
            color: colors[polygonsRef.current.length % colors.length],
            points: [...draftRef.current],
          });
        }
        // FIX 4: Always clear draft (even tiny ones) to avoid stale points
        draftRef.current = [];
        modeRef.current = "passive";
        prevPinchDistRef.current = null;
        prevPanAnchorRef.current = null;
        actionLockRef.current.erase = false;
        actionLockRef.current.undo = false;
        return;
      }

      // FIX 6: Explicit two-hand zoom check — avoid false positives from mixed gestures
      if (hands.length >= 2) {
        const g0 = hands[0].confirmed;
        const g1 = hands[1].confirmed;
        const bothPinching = g0 === "pinch" && g1 === "pinch";
        const pinchAndOpen =
          (g0 === "pinch" && g1 === "open") ||
          (g0 === "open" && g1 === "pinch");

        if (bothPinching || pinchAndOpen) {
          modeRef.current = "zooming";
          const a = hands[0].indexTip;
          const b = hands[1].indexTip;
          const d = dist(a, b);
          if (!prevPinchDistRef.current) {
            prevPinchDistRef.current = d;
          } else {
            const ratio = d / prevPinchDistRef.current;
            transformRef.current.scale = Math.min(
              3,
              Math.max(0.5, transformRef.current.scale * ratio)
            );
            prevPinchDistRef.current = d;
            setZoom(transformRef.current.scale);
          }
          return;
        }
      }

      const hand = hands[0];
      const screenPoint = {
        x: hand.indexTip.x * canvas.width,
        y: hand.indexTip.y * canvas.height,
      };
      const worldPoint = toWorld(screenPoint);

      if (hand.confirmed === "point") {
        modeRef.current = "drawing";
        const draft = draftRef.current;
        const minDist = 8 / transformRef.current.scale;
        if (!draft.length || dist(draft[draft.length - 1], worldPoint) > minDist) {
          // FIX 3: Apply smoothing before pushing the point
          draft.push(smoothPoint(draft, worldPoint));
        }
      } else if (hand.confirmed === "peace") {
        if (draftRef.current.length >= 3) {
          polygonsRef.current.push({
            id: idRef.current++,
            color: colors[polygonsRef.current.length % colors.length],
            points: [...draftRef.current],
          });
          draftRef.current = [];
        }
      } else if (hand.confirmed === "open") {
        modeRef.current = "panning";
        const wrist = {
          x: hand.wrist.x * canvas.width,
          y: hand.wrist.y * canvas.height,
        };
        if (!prevPanAnchorRef.current) {
          prevPanAnchorRef.current = wrist;
        } else {
          transformRef.current.tx += wrist.x - prevPanAnchorRef.current.x;
          transformRef.current.ty += wrist.y - prevPanAnchorRef.current.y;
          prevPanAnchorRef.current = wrist;
        }
      } else if (hand.confirmed === "fist") {
        if (!actionLockRef.current.erase) {
          undoRef.current.push(
            polygonsRef.current.map((p) => ({ ...p, points: [...p.points] }))
          );
          polygonsRef.current = [];
          actionLockRef.current.erase = true;
        }
      } else if (hand.confirmed === "rockon") {
        if (!actionLockRef.current.undo) {
          polygonsRef.current.pop();
          actionLockRef.current.undo = true;
        }
      } else {
        prevPanAnchorRef.current = null;
        prevPinchDistRef.current = null;
        actionLockRef.current.erase = false;
        actionLockRef.current.undo = false;
      }
    };

    const render = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // Use logical pixels (CSS size), not physical canvas size
      const w = parseInt(canvas.style.width || String(canvas.width));
      const h = parseInt(canvas.style.height || String(canvas.height));

      ctx.fillStyle = "#0b1117";
      ctx.fillRect(0, 0, w, h);

      const { tx, ty, scale } = transformRef.current;
      ctx.save();
      ctx.translate(tx, ty);
      ctx.scale(scale, scale);

      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      polygonsRef.current.forEach((poly) => {
        ctx.strokeStyle = poly.color;
        ctx.lineWidth = 3 / scale;
        ctx.beginPath();
        poly.points.forEach((p, i) => {
          if (i === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        });
        ctx.stroke();
      });

      if (draftRef.current.length) {
        ctx.strokeStyle = "#ffffff";
        ctx.setLineDash([6 / scale, 6 / scale]);
        ctx.lineWidth = 2 / scale;
        ctx.beginPath();
        draftRef.current.forEach((p, i) =>
          i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)
        );
        ctx.stroke();
        ctx.setLineDash([]);
      }

      ctx.restore();

      const hand = handsRef.current[0];
      if (hand) {
        const p = toScreen({
          x: hand.indexTip.x * w,
          y: hand.indexTip.y * h,
        });
        ctx.fillStyle = "#ffe066";
        ctx.beginPath();
        ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
        ctx.fill();
      }

      setPolygonCount(polygonsRef.current.length);
      setGestureLabel(handsRef.current[0]?.confirmed ?? "unknown");

      // FIX 8: Only schedule next frame if we're still rendering
      if (renderingRef.current) {
        rafRef.current = requestAnimationFrame(render);
      }
    };

    // FIX 5: Throttled OpenCV motion scoring (max ~10fps)
    const computeMotion = () => {
      const now = Date.now();
      if (now - lastMotionTimeRef.current < 100) return;
      lastMotionTimeRef.current = now;

      const video = videoRef.current;
      if (!video || !window.cv || video.readyState < 2) return;

      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = 160;
      tempCanvas.height = 120;
      const tempCtx = tempCanvas.getContext("2d");
      if (!tempCtx) return;

      tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
      const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
      const cv = window.cv;

      let src: CVMat | null = null;
      let gray: CVMat | null = null;
      let diff: CVMat | null = null;

      try {
        src = cv.matFromImageData(imageData);
        gray = new cv.Mat();
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

        if (lastGrayRef.current) {
          diff = new cv.Mat();
          cv.absdiff(gray, lastGrayRef.current, diff);
          const mean = cv.mean(diff)[0];
          setMotionScore(Math.round(mean));
          lastGrayRef.current.delete();
        }
        // FIX 4: Always store the latest gray frame
        lastGrayRef.current = gray.clone();
      } finally {
        // FIX 4: Ensure all mats are deleted even on error
        src?.delete();
        gray?.delete();
        diff?.delete();
      }
    };

    const init = async () => {
      try {
        await loadScript("https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js");
        await loadScript(
          "https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"
        );
        await loadScript("https://docs.opencv.org/4.x/opencv.js");

        const video = videoRef.current;
        if (!video) return;

        updateCanvasSize();
        window.addEventListener("resize", updateCanvasSize);

        const hands = new window.Hands({
          locateFile: (file) =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
        });

        hands.setOptions({
          maxNumHands: 2,
          modelComplexity: 1,
          minDetectionConfidence: 0.6,
          minTrackingConfidence: 0.6,
        });

        // FIX 2: Pass handedness from MediaPipe into classify()
        hands.onResults((results) => {
          const landmarks = results.multiHandLandmarks ?? [];
          const handednessList = results.multiHandedness ?? [];

          handsRef.current = landmarks.map((lm, idx) => {
            const handedness = handednessList[idx]?.label ?? "Right";
            const gesture = classify(lm, handedness);
            return {
              landmarks: lm,
              gesture,
              confirmed: confirmGesture(idx, gesture),
              wrist: lm[0],
              indexTip: lm[8],
              handedness,
            };
          });

          setHandsCount(handsRef.current.length);

          const canvas = canvasRef.current;
          if (canvas) processGestures(canvas);
          computeMotion();
        });

        camera = new window.Camera(video, {
          onFrame: async () => {
            await hands.send({ image: video });
          },
          width: 960,
          height: 540,
        });

        await camera.start();
        setStatus(
          "Ready — Point: draw · Peace: finalize · Open: pan · Two-hand pinch: zoom · Fist: clear · Rock-on: undo"
        );

        // FIX 8: Start render loop and handle tab visibility
        renderingRef.current = true;
        rafRef.current = requestAnimationFrame(render);

        const onVisibilityChange = () => {
          if (document.hidden) {
            renderingRef.current = false;
            cancelAnimationFrame(rafRef.current);
          } else {
            renderingRef.current = true;
            rafRef.current = requestAnimationFrame(render);
          }
        };
        document.addEventListener("visibilitychange", onVisibilityChange);

        // Store cleanup fn on the ref so the return fn can reach it
        (camera as unknown as { _visCleanup: () => void })._visCleanup =
          () => document.removeEventListener("visibilitychange", onVisibilityChange);
      } catch (error) {
        setStatus(
          error instanceof Error ? error.message : "Failed to initialize"
        );
      }
    };

    void init();

    return () => {
      renderingRef.current = false;
      cancelAnimationFrame(rafRef.current);
      if (camera) {
        camera.stop();
        const c = camera as unknown as { _visCleanup?: () => void };
        c._visCleanup?.();
      }
      window.removeEventListener("resize", updateCanvasSize);
      // FIX 4: Clean up last OpenCV mat on unmount
      if (lastGrayRef.current) {
        lastGrayRef.current.delete();
        lastGrayRef.current = null;
      }
    };
  }, [updateCanvasSize]);

  const clearAll = () => {
    undoRef.current.push(
      polygonsRef.current.map((p) => ({ ...p, points: [...p.points] }))
    );
    polygonsRef.current = [];
  };

  const undo = () => {
    polygonsRef.current.pop();
  };

  const resetView = () => {
    transformRef.current = { scale: 1, tx: 0, ty: 0 };
    setZoom(1);
  };

  const saveImage = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement("a");
    link.href = canvas.toDataURL("image/png");
    link.download = "gestureboard.png";
    link.click();
  };

  return (
    <div className="board">
      <canvas ref={canvasRef} className="draw-canvas" />
      <div className="hud">
        <h1>GestureBoard (Next.js)</h1>
        <p>{status}</p>
        <ul>
          <li>Hands: {handsCount}</li>
          <li>Gesture: {gestureLabel}</li>
          <li>Polygons: {polygonCount}</li>
          <li>Zoom: {zoom.toFixed(2)}x</li>
          <li>OpenCV Motion: {motionScore}</li>
        </ul>
      </div>

      <div className="camera-panel">
        <video
          ref={videoRef}
          className="camera"
          playsInline
          muted
          autoPlay
        />
      </div>

      <div className="toolbar">
        <button onClick={clearAll}>Clear</button>
        <button onClick={undo}>Undo</button>
        <button onClick={resetView}>Reset View</button>
        <button onClick={saveImage}>Save PNG</button>
      </div>
    </div>
  );
}
