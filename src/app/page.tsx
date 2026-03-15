"use client";

import { useEffect, useRef, useState } from "react";

type Point = { x: number; y: number };
type Gesture = "unknown" | "point" | "peace" | "open" | "fist" | "pinch" | "rockon";

type HandState = {
  landmarks: Point[];
  gesture: Gesture;
  confirmed: Gesture;
  wrist: Point;
  indexTip: Point;
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

declare global {
  interface Window {
    Hands: new (config: { locateFile: (file: string) => string }) => {
      setOptions: (options: Record<string, number>) => void;
      onResults: (cb: (results: { multiHandLandmarks?: Point[][] }) => void) => void;
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
    const existing = document.querySelector(`script[src=\"${src}\"]`);
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
  const holdRef = useRef<Map<number, { gesture: Gesture; hold: number; confirmed: Gesture }>>(new Map());
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

  useEffect(() => {
    let raf = 0;
    let camera: { start: () => Promise<void>; stop: () => void } | null = null;

    const updateCanvasSize = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    const classify = (lm: Point[]): Gesture => {
      const fingerUp = (tip: number, pip: number) => lm[tip].y < lm[pip].y;
      const indexUp = fingerUp(8, 6);
      const middleUp = fingerUp(12, 10);
      const ringUp = fingerUp(16, 14);
      const pinkyUp = fingerUp(20, 18);
      const thumbOpen = lm[4].x < lm[3].x;
      const pinch = dist(lm[4], lm[8]) < 0.05;

      if (pinch) return "pinch";
      if (indexUp && !middleUp && !ringUp && !pinkyUp) return "point";
      if (indexUp && middleUp && !ringUp && !pinkyUp) {
        return dist(lm[8], lm[12]) > 0.06 ? "peace" : "point";
      }
      if (indexUp && pinkyUp && !middleUp && !ringUp) return "rockon";
      if (thumbOpen && indexUp && middleUp && ringUp && pinkyUp) return "open";
      if (!indexUp && !middleUp && !ringUp && !pinkyUp) return "fist";
      return "unknown";
    };

    const confirmGesture = (idx: number, gesture: Gesture): Gesture => {
      const current = holdRef.current.get(idx) ?? { gesture: "unknown" as Gesture, hold: 0, confirmed: "unknown" as Gesture };
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
        if (draftRef.current.length >= 3) {
          polygonsRef.current.push({
            id: idRef.current++,
            color: colors[polygonsRef.current.length % colors.length],
            points: [...draftRef.current],
          });
        }
        draftRef.current = [];
        modeRef.current = "passive";
        prevPinchDistRef.current = null;
        prevPanAnchorRef.current = null;
        actionLockRef.current.erase = false;
        actionLockRef.current.undo = false;
        return;
      }

      if (hands.length >= 2 && [hands[0].confirmed, hands[1].confirmed].every((g) => g === "pinch" || g === "open")) {
        modeRef.current = "zooming";
        const a = hands[0].indexTip;
        const b = hands[1].indexTip;
        const d = dist(a, b);
        if (!prevPinchDistRef.current) {
          prevPinchDistRef.current = d;
        } else {
          const ratio = d / prevPinchDistRef.current;
          transformRef.current.scale = Math.min(3, Math.max(0.5, transformRef.current.scale * ratio));
          prevPinchDistRef.current = d;
          setZoom(transformRef.current.scale);
        }
        return;
      }

      const hand = hands[0];
      const screenPoint = { x: hand.indexTip.x * canvas.width, y: hand.indexTip.y * canvas.height };
      const worldPoint = toWorld(screenPoint);

      if (hand.confirmed === "point") {
        modeRef.current = "drawing";
        const draft = draftRef.current;
        if (!draft.length || dist(draft[draft.length - 1], worldPoint) > 8 / transformRef.current.scale) {
          draft.push(worldPoint);
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
        const wrist = { x: hand.wrist.x * canvas.width, y: hand.wrist.y * canvas.height };
        if (!prevPanAnchorRef.current) {
          prevPanAnchorRef.current = wrist;
        } else {
          transformRef.current.tx += wrist.x - prevPanAnchorRef.current.x;
          transformRef.current.ty += wrist.y - prevPanAnchorRef.current.y;
          prevPanAnchorRef.current = wrist;
        }
      } else if (hand.confirmed === "fist") {
        if (!actionLockRef.current.erase) {
          undoRef.current.push(polygonsRef.current.map((p) => ({ ...p, points: [...p.points] })));
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

      ctx.fillStyle = "#0b1117";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

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
        draftRef.current.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
        ctx.stroke();
        ctx.setLineDash([]);
      }

      ctx.restore();

      const hand = handsRef.current[0];
      if (hand) {
        const p = toScreen({ x: hand.indexTip.x * canvas.width, y: hand.indexTip.y * canvas.height });
        ctx.fillStyle = "#ffe066";
        ctx.beginPath();
        ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
        ctx.fill();
      }

      setPolygonCount(polygonsRef.current.length);
      setGestureLabel(handsRef.current[0]?.confirmed ?? "unknown");
      raf = requestAnimationFrame(render);
    };

    const computeMotion = () => {
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
      const src = cv.matFromImageData(imageData);
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
      if (lastGrayRef.current) {
        const diff = new cv.Mat();
        cv.absdiff(gray, lastGrayRef.current, diff);
        const mean = cv.mean(diff)[0];
        setMotionScore(Math.round(mean));
        diff.delete();
        lastGrayRef.current.delete();
      }
      lastGrayRef.current = gray.clone();
      src.delete();
      gray.delete();
    };

    const init = async () => {
      try {
        await loadScript("https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js");
        await loadScript("https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js");
        await loadScript("https://docs.opencv.org/4.x/opencv.js");

        const video = videoRef.current;
        if (!video) return;

        updateCanvasSize();
        window.addEventListener("resize", updateCanvasSize);

        const hands = new window.Hands({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
        });

        hands.setOptions({
          maxNumHands: 2,
          modelComplexity: 1,
          minDetectionConfidence: 0.6,
          minTrackingConfidence: 0.6,
        });

        hands.onResults((results) => {
          const landmarks = results.multiHandLandmarks ?? [];
          handsRef.current = landmarks.map((lm, idx) => {
            const gesture = classify(lm);
            return {
              landmarks: lm,
              gesture,
              confirmed: confirmGesture(idx, gesture),
              wrist: lm[0],
              indexTip: lm[8],
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
        setStatus("Ready. Point to draw, peace to finalize, open to pan, fist clear, rockon undo.");
        render();
      } catch (error) {
        setStatus(error instanceof Error ? error.message : "Failed to initialize");
      }
    };

    void init();

    return () => {
      if (camera) camera.stop();
      window.cancelAnimationFrame(raf);
      window.removeEventListener("resize", updateCanvasSize);
      if (lastGrayRef.current) {
        lastGrayRef.current.delete();
      }
    };
  }, []);

  const clearAll = () => {
    undoRef.current.push(polygonsRef.current.map((p) => ({ ...p, points: [...p.points] })));
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
        <video ref={videoRef} className="camera" playsInline muted autoPlay />
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
