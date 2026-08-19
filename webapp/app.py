"""PAC4200 NILM Live Monitor - local web GUI.

Runs the Milestone 3 detection pipeline (online step detector + running-set
tracker + Random Forest classifier) behind a web page styled like an energy
analyser. Two input sources:

  - a recorded CSV, replayed as if arriving live (for testing and the demo)
  - the real PAC4200 over Modbus TCP (pymodbus), for the live challenge

Run:
    cd webapp
    pip install flask
    python app.py
Then open http://127.0.0.1:5000 in a browser.

For the live meter, set METER_HOST below to your PAC4200 IP and pick "LIVE METER"
in the page. The Modbus register map lives in nilm/streaming.py and must match
your meter (verify against your Node-RED flow).
"""

import os
import sys
import time
import threading

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, make_response

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nilm import (aggregator, disaggregator, tracker, streaming,   # noqa: E402
                  calibration, loadtype, steadystate)

RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
METER_HOST = os.getenv("PAC4200_HOST", "127.0.0.1")  # Set PAC4200_HOST for live Modbus mode
LIVE_CSV_PATH = os.getenv("NILM_LIVE_CSV", os.path.join(os.path.dirname(__file__), "live.csv"))
SPS = 5
WARMUP_S = 5.0      # settled seconds needed before we judge the standing load
GIVEUP_S = 25.0     # if the signal never settles, assume nothing was already on

# Canonical PAC4200 column names. Used as a fallback when the live CSV has no
# header on line 1 - which happens when Node-RED appends to an existing file
# instead of recreating it, leaving a stale data row where the header should be.
LIVE_HEADER = ("timestamp_iso,device_name,run_id,sample_interval_ms,u_l1_n_v,"
               "i_l1_a,p_total_w,s_total_va,s_calc_va,q_total_var,pf_total,"
               "frequency_hz,thd_u_l1_percent,thd_i_l1_percent,"
               "block_time_difference_ms").split(",")

# electrical parameters to show: csv_column -> (label, unit)
ELEC = [
    ("u_l1_n_v", "Voltage L-N", "V"),
    ("i_l1_a", "Current", "A"),
    ("frequency_hz", "Frequency", "Hz"),
    ("p_total_w", "Active Power", "W"),
    ("q_total_var", "Reactive Power", "var"),
    ("s_total_va", "Apparent Power", "VA"),
    ("pf_total", "Power Factor", ""),
    ("thd_u_l1_percent", "THD Voltage", "%"),
    ("thd_i_l1_percent", "THD Current", "%"),
]

app = Flask(__name__)
LOCK = threading.Lock()


def _clean(v):
    try:
        v = float(v)
        return 0.0 if (v != v) else round(v, 2)   # v!=v catches NaN
    except (TypeError, ValueError):
        return 0.0


class Engine:
    def __init__(self):
        print("Training classifier (one-time, a few seconds)...")
        out = disaggregator.train_step_classifier(n_train=300, n_test=30)
        self.model = out["rf"]              # DisaggregatorModel wrapper
        self.rf = out["rf"].model           # raw RandomForest (for probabilities)
        calib = calibration.load(os.path.join(os.path.dirname(__file__), "..",
                                              calibration.CALIB_FILE))
        self.params = calibration.apply_calibration(aggregator.device_params(), calib)
        # Candidate set for DETECTION = on-site measured devices only. The raw
        # Milestone-1 fingerprints were recorded on other hardware (the transfer
        # gap); leaving them in the search only adds look-alikes the tracker then
        # confuses for real switches. Detection uses the calibrated set; the full
        # set is kept only for describing unknowns.
        self.detect_params = {d: v for d, v in self.params.items() if v.get("calibrated")}
        if not self.detect_params:
            self.detect_params = self.params
        print(f"Site calibration: {len(calib)} devices calibrated"
              if calib else "Site calibration: none found (using solo fingerprints)")
        self.worker = None
        self.reset()
        print("Engine ready.")

    def reset(self):
        with LOCK:
            self.det = streaming.OnlineStepDetector(sps=SPS)
            self.trk = tracker.RunningSetTracker(self.detect_params, classifier=self.model,
                                                 on_unknown=self._unknown)
            self.state = {
                "running": False, "source": "-", "t": 0.0,
                "elec": {k: 0.0 for k, _, _ in ELEC},
                "devices": [], "match": "-", "match_state": "idle",
                "confidence": 0, "last_dp": 0.0, "last_action": "",
                "hist_t": [], "hist_p": [], "events": [],
                "unknown": None, "rows_read": 0,
                "last_row_wall": 0.0, "age_s": 0.0, "start_iso": None,
                "standing": None,
            }
            self.standing_done = False
            self._warm = []

    def clear_graph(self):
        """Clear ONLY the chart history and the event markers. Keep the running
        device set, the standing-load result and the live detector untouched, so
        devices that are on (a fan, the cooler) stay recognised - only the piled-up
        plot and its labels are wiped."""
        with LOCK:
            self.state["hist_t"] = []
            self.state["hist_p"] = []
            self.state["events"] = []

    def _soft_reset(self):
        """New logging session on the same feed: forget the old signal and the
        old device set, but keep streaming."""
        with LOCK:
            self.det = streaming.OnlineStepDetector(sps=SPS)
            self.trk = tracker.RunningSetTracker(self.detect_params, classifier=self.model,
                                                 on_unknown=self._unknown)
            self.state["devices"] = []
            self.state["events"] = []
            self.state["hist_t"] = []; self.state["hist_p"] = []
            self.state["start_iso"] = None
            self.state["standing"] = None
            self.standing_done = False
            self._warm = []
            self.state["unknown"] = None
            self.state["match"] = "-"
            self.state["match_state"] = "idle"

    def _unknown(self, dp, dq):
        """A step that matches no known device or combination."""
        info = loadtype.describe_load(dp, dq)
        near = loadtype.nearest_known(dp, dq, self.params, k=2)
        with LOCK:
            self.state["unknown"] = {
                "label": info["label"], "description": info["description"],
                "class": info["class"], "p_w": info["p_w"], "q_var": info["q_var"],
                "s_va": info["s_va"], "pf": info["pf"],
                "delta_p": round(float(dp), 2), "delta_q": round(float(dq), 2),
                "nearest": [{"device": d, "distance": x} for d, x in near],
            }
            self.state["match"] = info["label"]
            self.state["confidence"] = 0

    def classify(self, dp, dq):
        feat = disaggregator.step_features(dp, dq)
        X = pd.DataFrame([feat], columns=disaggregator.STEP_FEATURE_NAMES)
        proba = self.rf.predict_proba(X)[0]
        i = int(proba.argmax())
        return str(self.rf.classes_[i]), int(round(float(proba[i]) * 100))

    def _check_standing_load(self, t, p, q, thd):
        """Identify whatever is ALREADY running when monitoring starts.

        The step detector only sees changes. A device switched on before we
        pressed Start produces no step and would stay invisible. So we look at
        the settled signal once, explain it with the known devices, and seed the
        tracker with the answer.
        """
        if self.standing_done:
            return
        self._warm.append((t, p, q, thd))
        if t < WARMUP_S:
            return
        win = [w for w in self._warm if w[0] >= t - WARMUP_S]
        ps = np.array([w[1] for w in win])
        qs = np.array([w[2] for w in win])
        thds = [w[3] for w in win if w[3] is not None and np.isfinite(w[3])]
        # A load is settled if its ACTIVE power holds steady, OR - for a small or
        # generating load such as PV, whose watts wander with the sun while its
        # reactive draw stays put - if its APPARENT power holds steady. Without
        # the second test the PV bus never settles and is never identified.
        ss = np.hypot(ps, qs)
        settled = (ps.std() <= max(1.5, 0.02 * max(abs(ps.mean()), 1.0)) or
                   ss.std() <= max(1.5, 0.02 * max(ss.mean(), 1.0)))
        if not settled:
            if t > GIVEUP_S:
                self.standing_done = True
                with LOCK:
                    self.state["standing"] = {"idle": False, "devices": [], "confidence": 0,
                        "note": "The signal was still changing when monitoring began, so "
                                "nothing was assumed to be already running."}
            return

        snap = steadystate.steady_snapshot(ps, qs)
        thd = float(np.median(thds)) if thds else None
        res = steadystate.identify_standing_load(snap["P"], snap["Q"], self.detect_params, thd_i=thd)
        self.standing_done = True
        devices = sorted(str(x) for x in res["best"])
        if not res["idle"] and devices:
            for d in devices:
                self.trk.on.add(d)
        with LOCK:
            self.state["standing"] = {
                "idle": res["idle"], "devices": devices,
                "confidence": res["confidence"], "residual": res["residual"],
                "note": res["note"], "p_w": round(snap["P"], 1),
                "q_var": round(snap["Q"], 1),
                "thd_i": round(thd, 1) if thd is not None else None,
                "alternatives": res["alternatives"],
            }
            self.state["devices"] = sorted(str(x) for x in self.trk.on)
            if devices:
                self.state["events"].insert(0, {"t": round(t, 2),
                    "device": " + ".join(devices) + " (already running)",
                    "dp": 0.0, "action": "ON", "conf": res["confidence"]})

    def feed(self, t, row):
        p = float(row.get("p_total_w", 0.0))
        q = float(row.get("q_total_var", 0.0))
        thd = row.get("thd_i_l1_percent")
        try:
            thd = float(thd)
        except (TypeError, ValueError):
            thd = None
        self._check_standing_load(t, p, q, thd)
        steps = self.det.push(t, p, q)
        for step in steps:
            dp, dq = step["delta_p"], step["delta_q"]
            before = set(self.trk.on)
            self.trk.process(pd.DataFrame([step]))
            after = set(self.trk.on)
            added, removed = after - before, before - after
            changed = sorted(str(x) for x in (added or removed))
            if not changed:
                # nothing known explains this step; _unknown() has characterised it
                with LOCK:
                    if self.state.get("unknown"):
                        self.state["events"].insert(0, {
                            "t": round(float(step["t_sec"]), 2),
                            "device": self.state["unknown"]["label"],
                            "dp": round(dp, 1),
                            "action": "ON" if dp > 0 else "OFF", "conf": 0})
                        self.state["events"] = self.state["events"][:40]
                continue
            action = "ON" if added else "OFF"
            # confidence from how well the chosen devices explain the step
            last = getattr(self.trk, "last", {}) or {}
            err = min(last.get("single_err", 1e9), last.get("combo_err", 1e9))
            tol = max(last.get("tol", 1.0), 1e-6)
            conf = int(max(5, min(99, round(100.0 * (1.0 - err / tol)))))
            name = " + ".join(changed)
            # the event belongs at the time of the STEP, not at the moment we
            # confirmed it a few seconds later. Using `t` here is what made the
            # marker on the graph sit to the right of the real switch.
            t_event = round(float(step["t_sec"]), 2)
            with LOCK:
                self.state["match"] = name
                self.state["match_state"] = "on" if added else "off"
                self.state["confidence"] = conf
                self.state["last_dp"] = round(dp, 1)
                self.state["last_action"] = action
                self.state["events"].insert(0, {"t": t_event, "device": name,
                                                "dp": round(dp, 1), "action": action,
                                                "conf": conf})
                self.state["events"] = self.state["events"][:40]
        with LOCK:
            self.state["elec"] = {k: _clean(row.get(k, 0.0)) for k, _, _ in ELEC}
            self.state["devices"] = sorted(str(x) for x in self.trk.on)
            self.state["hist_t"].append(round(t, 2))
            self.state["hist_p"].append(round(p, 1))
            if len(self.state["hist_t"]) > 30000:      # ~100 min at 5 Hz
                self.state["hist_t"] = self.state["hist_t"][-30000:]
                self.state["hist_p"] = self.state["hist_p"][-30000:]
            self.state["t"] = round(t, 1)
            self.state["last_row_wall"] = time.time()
            self.state["rows_read"] += 1

    # ---- input loops (run in a background thread) ----
    def start_csv(self, filename, speed):
        path = os.path.join(RECORDINGS_DIR, filename)
        df = pd.read_csv(path)
        df = df.dropna(subset=["timestamp_iso", "p_total_w"]).reset_index(drop=True)
        df["_t"] = pd.to_datetime(df["timestamp_iso"])
        df["t_sec"] = (df["_t"] - df["_t"].iloc[0]).dt.total_seconds()
        self.reset()
        with LOCK:
            self.state["running"] = True
            self.state["source"] = filename
        with LOCK:
            self.state["start_iso"] = str(df["_t"].iloc[0])
        for _, row in df.iterrows():
            if not self.state["running"]:
                break
            self.feed(float(row["t_sec"]), row.to_dict())
            time.sleep((1.0 / SPS) / max(speed, 0.1))
        self.det.flush()
        with LOCK:
            self.state["running"] = False

    def start_live_csv(self, path):
        """Follow the CSV that Node-RED writes live.

        Robustness matters more than cleverness here. Three things go wrong in a
        real lab and each one silently freezes the display:

        1. Node-RED recreates or truncates the file when a new run starts. The
           old file handle then points past the end and nothing is ever read
           again, so the dashboard keeps showing the last old row while the
           meter reads something completely different. We watch the file size
           and reopen when it shrinks.
        2. A row can be half-written when we read it. We only parse lines that
           end with a newline and keep the remainder for the next pass.
        3. If the feed dies, nothing on screen says so. We stamp every accepted
           row so the page can show the age of the data and turn red when stale.
        """
        self.reset()
        with LOCK:
            self.state["running"] = True
            self.state["source"] = f"LIVE (Node-RED): {os.path.basename(path)}"

        for _ in range(300):                       # wait up to 30 s for the file
            if os.path.exists(path) or not self.state["running"]:
                break
            time.sleep(0.1)
        if not os.path.exists(path):
            with LOCK:
                self.state["running"] = False
                self.state["source"] = "live file not found - check LIVE_CSV_PATH"
            return

        f = open(path, "r")
        header = self._read_header(f)
        f.seek(0, os.SEEK_END)                     # skip the backlog, start live
        pos = f.tell()
        remainder, t0 = "", None

        while self.state["running"]:
            try:
                size = os.path.getsize(path)
            except OSError:
                time.sleep(0.1)
                continue
            if size < pos:                         # truncated or recreated
                f.close()
                f = open(path, "r")
                header = self._read_header(f)
                pos = f.tell()
                remainder, t0 = "", None
                self._soft_reset()                 # new session: clear old state
                with LOCK:
                    self.state["source"] = (f"LIVE (Node-RED): "
                                            f"{os.path.basename(path)} (restarted)")
            f.seek(pos)
            chunk = f.read()
            pos = f.tell()
            if not chunk:
                time.sleep(0.05)
                continue
            data = remainder + chunk
            lines = data.split("\n")
            remainder = lines.pop()                # trailing partial line
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if "timestamp_iso" in line and "p_total_w" in line:
                    header = line.split(",")   # a header row appeared mid-stream
                    continue                    # (new Node-RED session): re-sync
                vals = line.split(",")
                if len(vals) != len(header):
                    continue
                row = dict(zip(header, vals))
                if "p_total_w" not in row or "timestamp_iso" not in row:
                    continue
                try:
                    ts = pd.to_datetime(row["timestamp_iso"])
                except Exception:
                    continue
                if t0 is None:
                    t0 = ts
                    with LOCK:
                        self.state["start_iso"] = str(ts)
                parsed = {}
                for k, v in row.items():
                    try:
                        parsed[k] = float(v)
                    except (TypeError, ValueError):
                        parsed[k] = v
                self.feed((ts - t0).total_seconds(), parsed)
        f.close()

    @staticmethod
    def _read_header(f):
        """Return the CSV column names.

        Node-RED sometimes appends to an existing file instead of recreating it,
        so line 1 can be a data row rather than the header. Accept a line as the
        header only if it actually names the required columns; otherwise fall
        back to the fixed PAC4200 schema so rows still parse instead of being
        silently dropped.
        """
        line = f.readline().strip()
        if "timestamp_iso" in line and "p_total_w" in line:
            return line.split(",")
        return list(LIVE_HEADER)

    def start_meter(self, speed):
        reader = streaming.PAC4200Reader(host=METER_HOST)
        try:
            reader.connect()
        except Exception as e:
            with LOCK:
                self.state["running"] = False
                self.state["source"] = f"meter error: {e}"
            return
        self.reset()
        with LOCK:
            self.state["running"] = True
            self.state["source"] = f"LIVE METER {METER_HOST}"
        t0 = time.time()
        last_ok = time.time()
        while self.state["running"]:
            tick = time.time()
            try:
                sample = reader.read_sample()
            except Exception:
                sample = None
            if sample is not None:
                last_ok = time.time()
                self.feed(round(time.time() - t0, 2), sample)
            elif time.time() - last_ok > 2.0:
                try:
                    reader.connect(); last_ok = time.time()
                except Exception:
                    pass
            dt = (1.0 / SPS) - (time.time() - tick)
            if dt > 0:
                time.sleep(dt)


engine = Engine()


def launch(target, *args):
    if engine.worker and engine.worker.is_alive():
        engine.state["running"] = False
        engine.worker.join(timeout=2.0)
    engine.worker = threading.Thread(target=target, args=args, daemon=True)
    engine.worker.start()


MAX_POINTS = 900


def _downsample(ts, ps, maxpts=MAX_POINTS):
    """Compress the whole session to at most maxpts points for the browser.

    The graph shows the entire run rather than a scrolling window, so the first
    switch stays visible no matter how long the session lasts.
    """
    n = len(ts)
    if n <= maxpts:
        return ts, ps
    stride = int(np.ceil(n / maxpts))
    return ts[::stride], ps[::stride]


@app.route("/state")
def state():
    with LOCK:
        st = dict(engine.state)
        lw = st.get("last_row_wall", 0.0)
        st["age_s"] = round(time.time() - lw, 1) if lw else -1.0
        ts, ps = _downsample(st.pop("hist_t", []), st.pop("hist_p", []))
        st["hist_t"] = ts
        st["hist_p"] = ps
        st["elec_meta"] = [[k, l, u] for k, l, u in ELEC]
        return jsonify(st)


@app.route("/onboard", methods=["POST"])
def onboard():
    """Teach the system a device it has never seen, from ONE human label.

    The unknown step already told us the device's active and reactive power.
    We store that as its signature, mark it as measured on site, save it to the
    calibration file so it survives a restart, and add it to the running set
    because it is on right now. From the next switch onward it is a known device.
    """
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    with LOCK:
        unk = engine.state.get("unknown")
    if not name:
        return jsonify(ok=False, error="a name is required")
    if not unk:
        return jsonify(ok=False, error="no unidentified load to label")

    dp, dq = float(unk["delta_p"]), float(unk["delta_q"])
    P, Q = abs(dp), (dq if dp >= 0 else -dq)

    calib_path = os.path.join(os.path.dirname(__file__), "..", calibration.CALIB_FILE)
    calib = calibration.load(calib_path)
    calib[name] = {"P": round(P, 2), "Q": round(Q, 2), "n_obs": 1}
    calibration.save(calib, calib_path)

    engine.params[name] = {"P": P, "Q": Q, "P_std": 1.0, "Q_std": 1.0, "calibrated": True}
    engine.detect_params[name] = engine.params[name]   # add to the on-site candidate set
    engine.trk.params = engine.detect_params            # keep the tracker on the clean set
    engine.trk.on.add(name)
    with LOCK:
        engine.state["unknown"] = None
        engine.state["match"] = name
        engine.state["match_state"] = "on"
        engine.state["confidence"] = 99
        engine.state["devices"] = sorted(str(x) for x in engine.trk.on)
        engine.state["events"].insert(0, {"t": engine.state.get("t", 0.0),
                                          "device": name + " (learned)",
                                          "dp": round(dp, 1), "action": "ON", "conf": 99})
    return jsonify(ok=True, name=name, p_w=round(P, 2), q_var=round(Q, 2),
                   known_devices=len(engine.params))


@app.route("/csvs")
def csvs():
    files = sorted(f for f in os.listdir(RECORDINGS_DIR) if f.endswith(".csv"))
    return jsonify(files)


@app.route("/start", methods=["POST"])
def start():
    data = request.get_json(force=True)
    src = data.get("source")
    speed = float(data.get("speed", 1.0))
    if src == "__meter__":
        launch(engine.start_meter, speed)
    elif src == "__livecsv__":
        launch(engine.start_live_csv, LIVE_CSV_PATH)
    else:
        launch(engine.start_csv, src, speed)
    return jsonify(ok=True)


@app.route("/stop", methods=["POST"])
def stop():
    engine.state["running"] = False
    return jsonify(ok=True)


@app.route("/reset", methods=["POST"])
def reset():
    engine.state["running"] = False
    time.sleep(0.3)
    engine.reset()
    return jsonify(ok=True)


@app.route("/clear", methods=["POST"])
def clear():
    # Clear ONLY the chart and event markers, mid-session, without touching the
    # device set - devices that are on (fan, cooler) stay listed. Use when the
    # colour segments and labels have piled up and become unreadable.
    engine.clear_graph()
    return jsonify(ok=True)


UI_VERSION = "v5-empro"


def _index_path():
    return os.path.join(os.path.dirname(__file__), "index.html")


@app.route("/")
def index():
    """Read index.html on EVERY request, in UTF-8, with caching switched off.

    The previous version read the file once at import and let the browser cache
    it, so replacing index.html appeared to do nothing: the old page kept being
    served. Reading per request plus no-store headers means what is on disk is
    what you see.
    """
    path = _index_path()
    if not os.path.exists(path):
        return f"index.html not found next to app.py (looked in {path})", 500
    with open(path, encoding="utf-8") as f:
        html = f.read()
    resp = make_response(html)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/version")
def version():
    """Confirm which index.html is actually being served."""
    path = _index_path()
    if not os.path.exists(path):
        return jsonify(ok=False, error="index.html missing", path=path)
    with open(path, encoding="utf-8") as f:
        html = f.read()
    return jsonify(ok=True, path=path, bytes=len(html),
                   ui_version=UI_VERSION,
                   has_chart='id="chart"' in html,
                   has_banner='class="banner"' in html,
                   has_stalebox='id="stalebox"' in html)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
