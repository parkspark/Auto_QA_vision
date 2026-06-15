# 동영상 업로드 → v3 + ByteTrack + hysteresis 락으로 추적 → MJPEG 라이브 스트리밍 + 지표 + mp4 다운로드
# 브라우저가 mp4v를 못 재생할 수 있어, 화면 표시는 MJPEG(프레임 단위 JPEG)로 한다.
import sys
import time
import uuid
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from track_video import pick_my_character  # 동일 후처리 재사용

WEIGHTS = ROOT / "runs" / "df_yolo11s_1280_v3" / "weights" / "best.pt"
TRACKER = ROOT / "scripts" / "bytetrack_df.yaml"
OUTDIR = Path(__file__).parent / "video_out"
OUTDIR.mkdir(exist_ok=True)

COAST_GRACE = 10
CHALLENGE_FRAMES = 6
RELOCK_MARGIN = 0.05

app = Flask(__name__)
model = YOLO(WEIGHTS)
JOBS = {}  # token -> {src, status, metrics, out}


@app.route("/")
def index():
    return send_from_directory(Path(__file__).parent, "video_index.html")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if f is None:
        return jsonify({"error": "video 파일이 없습니다"}), 400
    token = uuid.uuid4().hex[:12]
    suffix = Path(f.filename).suffix or ".mp4"
    src = OUTDIR / f"{token}_in{suffix}"
    f.save(str(src))
    JOBS[token] = {"src": str(src), "status": "uploaded"}
    return jsonify({"token": token})


def track_stream(token, pace):
    job = JOBS[token]
    src = job["src"]
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    out_path = OUTDIR / f"{token}_out.mp4"
    writer = None

    locked_id = locked_score = None
    challenger_id = None
    challenger_count = 0
    coast_run = 0
    n = frames_my = any_char = any_uid = id_switches = max_coast = 0
    job["status"] = "running"
    frame_interval = 1.0 / fps if pace else 0.0
    t_prev = time.time()

    results = model.track(source=src, imgsz=1280, conf=0.4, persist=True,
                          tracker=str(TRACKER), stream=True, verbose=False)
    for r in results:
        n += 1
        chars, ids, by_tid = [], [], {}
        boxes = r.boxes
        tids = boxes.id.tolist() if boxes.id is not None else [None] * len(boxes)
        for box, cls, c, tid in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), tids):
            d = {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                 "conf": c, "tid": int(tid) if tid is not None else None}
            if int(cls) == 0:
                chars.append(d)
                if d["tid"] is not None:
                    by_tid[d["tid"]] = d
            else:
                ids.append(d)
        if chars:
            any_char += 1
        if ids:
            any_uid += 1

        my = pick_my_character(chars, ids)
        cand_tid = my["character"]["tid"] if my else None
        cand_score = my["score"] if my else None
        locked_present = locked_id is not None and locked_id in by_tid
        if locked_present:
            locked_score = by_tid[locked_id]["conf"]
            if (cand_tid is not None and cand_tid != locked_id and cand_score is not None
                    and cand_score > locked_score + RELOCK_MARGIN):
                if cand_tid == challenger_id:
                    challenger_count += 1
                else:
                    challenger_id, challenger_count = cand_tid, 1
                if challenger_count >= CHALLENGE_FRAMES:
                    locked_id, locked_score = challenger_id, cand_score
                    challenger_id, challenger_count = None, 0
                    id_switches += 1
            else:
                challenger_id, challenger_count = None, 0
        else:
            if locked_id is None or coast_run >= COAST_GRACE:
                if cand_tid is not None:
                    if locked_id is not None and cand_tid != locked_id:
                        id_switches += 1
                    locked_id, locked_score = cand_tid, cand_score
                    challenger_id, challenger_count = None, 0

        my_box = by_tid.get(locked_id) if locked_id is not None else None
        if my_box is not None:
            frames_my += 1
            coast_run = 0
        else:
            coast_run += 1
            max_coast = max(max_coast, coast_run)

        frame = r.plot()
        if my_box is not None:
            cv2.rectangle(frame, (int(my_box["x1"]), int(my_box["y1"])),
                          (int(my_box["x2"]), int(my_box["y2"])), (0, 0, 255), 4)
            cv2.putText(frame, "MINE", (int(my_box["x1"]), max(20, int(my_box["y1"]) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "(my character lost)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

        if pace:
            dt = time.time() - t_prev
            if dt < frame_interval:
                time.sleep(frame_interval - dt)
            t_prev = time.time()

    if writer is not None:
        writer.release()
    job["out"] = str(out_path)
    job["metrics"] = {
        "frames": n,
        "my_located_rate": round(frames_my / n, 3) if n else 0,
        "any_char_rate": round(any_char / n, 3) if n else 0,
        "any_uid_rate": round(any_uid / n, 3) if n else 0,
        "id_switches": id_switches,
        "max_coast_frames": max_coast,
    }
    job["status"] = "done"


@app.route("/stream/<token>")
def stream(token):
    if token not in JOBS:
        return "unknown token", 404
    pace = request.args.get("pace", "1") != "0"
    return Response(track_stream(token, pace),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/metrics/<token>")
def metrics(token):
    j = JOBS.get(token, {})
    return jsonify({"status": j.get("status", "unknown"), "metrics": j.get("metrics")})


@app.route("/download/<token>")
def download(token):
    j = JOBS.get(token)
    if not j or "out" not in j:
        return "not ready", 404
    return send_file(j["out"], as_attachment=True, download_name="tracked.mp4")


if __name__ == "__main__":
    print(f"모델: {WEIGHTS}")
    print("브라우저에서 http://localhost:5001 을 여세요")
    app.run(host="127.0.0.1", port=5001, threaded=True)
