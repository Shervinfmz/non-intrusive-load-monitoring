"""Milestone 3 - real-time layer.

The batch pipeline reads a whole recording at once. The live lab challenge needs
detection as samples arrive. This module provides that without rebuilding the
detection logic: the same idea (a step is a sustained change in P or Q above the
local noise) is run causally on a stream.

Three parts:
  OnlineStepDetector  - a state machine fed one sample at a time; emits a step
                        only after the new level has HELD (so noise is not a
                        false switch). This is the source of the few-second
                        latency: we must wait for the level to settle.
  LiveDisaggregator   - wraps the online detector, the classifier, and the
                        running-set tracker; prints the device set as it changes.
  replay              - feeds a recorded DataFrame sample by sample, as if it were
                        arriving live, to validate the online path against batch.

PAC4200Reader (Modbus) is provided separately below and is the only part that
cannot be tested without the real meter.
"""

from collections import deque
import os
import numpy as np

from . import aggregator
from .tracker import RunningSetTracker


class OnlineStepDetector:
    """Fed one (t, p, q) at a time. Emits steps once they are confirmed.

    It runs the SAME batch detector on a rolling buffer, but finalises decisions
    with a monotonic time pointer: a region of the signal is only judged once its
    entire step-plus-settle window sits before the confirm cutoff, and once
    judged it is never revisited. So the live result matches the batch result on
    that region exactly, delayed by the confirm margin, and no transition is
    counted twice. Returns a list of newly confirmed steps (possibly empty).
    """

    def __init__(self, sps=5, win_s=3.0, confirm_s=4.0, keep_s=30.0, **batch_kw):
        self.sps = sps
        self.win_s = win_s
        self.confirm = confirm_s
        self.keep = keep_s
        self.batch_kw = batch_kw
        self.t, self.p, self.q = [], [], []
        self.finalized_t = -1e18
        self._min = int((2 * win_s + confirm_s) * sps)

    def push(self, t, p, q):
        from .disaggregator import detect_steps
        import pandas as pd
        self.t.append(float(t)); self.p.append(float(p)); self.q.append(float(q))
        if len(self.t) < self._min:
            return []
        cutoff = t - self.confirm            # only judge fully-formed regions
        if cutoff <= self.finalized_t:
            return []
        df = pd.DataFrame({"t_sec": self.t, "p_total_w": self.p, "q_total_var": self.q})
        steps = detect_steps(df, **self.batch_kw)
        new = []
        for _, s in steps.iterrows():
            ts = s["t_sec"]
            if self.finalized_t < ts <= cutoff:
                new.append({"t_sec": round(ts, 2), "delta_p": round(s["delta_p"], 2),
                            "delta_q": round(s["delta_q"], 2),
                            "thd_i_after": s.get("thd_i_after", float("nan"))})
        self.finalized_t = cutoff
        # prune history we no longer need to detect near the moving cutoff
        keep_from = t - self.keep
        if self.t[0] < keep_from:
            i = 0
            while i < len(self.t) and self.t[i] < keep_from:
                i += 1
            self.t = self.t[i:]; self.p = self.p[i:]; self.q = self.q[i:]
        return new


    def flush(self):
        """End of stream: finalise any remaining steps (no confirm margin)."""
        from .disaggregator import detect_steps
        import pandas as pd
        if len(self.t) < self._min:
            return []
        df = pd.DataFrame({"t_sec": self.t, "p_total_w": self.p, "q_total_var": self.q})
        steps = detect_steps(df, **self.batch_kw)
        new = [{"t_sec": round(s["t_sec"], 2), "delta_p": round(s["delta_p"], 2),
                "delta_q": round(s["delta_q"], 2)}
               for _, s in steps.iterrows() if s["t_sec"] > self.finalized_t]
        self.finalized_t = self.t[-1] if self.t else self.finalized_t
        return new


class LiveDisaggregator:
    """Online detector + classifier + running-set tracker, with live printout."""

    def __init__(self, params, classifier, sps=5, on_change=None, **det_kw):
        self.det = OnlineStepDetector(sps=sps, **det_kw)
        self.tr = RunningSetTracker(params, classifier=classifier)
        self.on_change = on_change or self._default_print
        self.last_set = set()

    def _default_print(self, t, changed, action, current):
        print(f"  [t={t:6.1f}s] {action:15s} {str(changed):20s} -> ON now: {sorted(str(x) for x in current)}")

    def feed(self, t, p, q):
        steps = self.det.push(t, p, q)
        if not steps:
            return
        import pandas as pd
        for step in steps:
            before = set(self.tr.on)
            self.tr.process(pd.DataFrame([step]))
            after = set(self.tr.on)
            for d in (after - before):
                self.on_change(t, d, "device ON", after)
            for d in (before - after):
                self.on_change(t, d, "device OFF", after)

    def flush(self):
        import pandas as pd
        for step in self.det.flush():
            before = set(self.tr.on)
            self.tr.process(pd.DataFrame([step]))
            after = set(self.tr.on)
            for d in (after - before):
                self.on_change(step["t_sec"], d, "device ON", after)
            for d in (before - after):
                self.on_change(step["t_sec"], d, "device OFF", after)

    def current(self):
        return set(self.tr.on)


def replay(df, live, realtime=False, sps=5):
    """Feed a recorded DataFrame to a LiveDisaggregator sample by sample."""
    import time
    for _, row in df.iterrows():
        live.feed(row["t_sec"], row["p_total_w"], row.get("q_total_var", 0.0))
        if realtime:
            time.sleep(1.0 / sps)
    live.flush()      # end of recording: confirm the final steps
    return live.current()


# ----------------------------------------------------------------------
# Modbus reader for the real PAC4200. CANNOT be validated without hardware.
# Register map must be confirmed against your working Node-RED flow.
# ----------------------------------------------------------------------
class PAC4200Reader:
    """Reads the PAC4200 directly over Modbus TCP with pymodbus.

    The register offsets below follow the standard Siemens PAC4200 input-register
    map (function code 4, 32-bit float = two 16-bit registers, big-endian). VERIFY
    every offset against your Node-RED flow before trusting the numbers: a wrong
    offset gives clean-looking but wrong values.
    """

    # offset (0-based input register) -> field, each value spans 2 registers
    FLOAT_MAP = {
        0: "v_l1_n", 2: "v_l2_n", 4: "v_l3_n",
        12: "i_l1", 14: "i_l2", 16: "i_l3",
        24: "p_l1", 26: "p_l2", 28: "p_l3",
        36: "q_l1", 38: "q_l2", 40: "q_l3",
        54: "power_factor", 60: "frequency",
        64: "p_total_w", 66: "q_total_var", 68: "s_total_va",
    }
    THD_MAP = {260: "thd_u", 262: "thd_i"}   # confirm against your flow

    def __init__(self, host=None, port=502, unit=1):
        self.host, self.port, self.unit = host, port, unit
        self.client = None

    def connect(self):
        from pymodbus.client import ModbusTcpClient
        self.client = ModbusTcpClient(self.host, port=self.port, timeout=1.0)
        if not self.client.connect():
            raise ConnectionError(f"cannot reach PAC4200 at {self.host}:{self.port}")

    @staticmethod
    def _f32(regs, i):
        import struct
        hi, lo = regs[i], regs[i + 1]
        return struct.unpack(">f", struct.pack(">HH", hi, lo))[0]

    def read_sample(self):
        """One reading as a dict. Reads the two blocks the flow uses."""
        blockA = self.client.read_input_registers(address=0, count=70, slave=self.unit)
        blockT = self.client.read_input_registers(address=260, count=4, slave=self.unit)
        if blockA.isError() or blockT.isError():
            return None
        out = {name: self._f32(blockA.registers, off) for off, name in self.FLOAT_MAP.items()}
        for off, name in self.THD_MAP.items():
            out[name] = self._f32(blockT.registers, off - 260)
        return out

    def stream(self, live, period_s=0.2, watchdog_s=2.0):
        """Live loop: read the meter, feed the model, keep a simple watchdog.

        Single thread at 5 Hz. Each tick: read, feed, check liveness. If a read
        fails or stalls longer than watchdog_s, it reconnects rather than hanging.
        """
        import time
        self.connect()
        last_ok = time.time()
        t0 = time.time()
        while True:
            tick = time.time()
            sample = None
            try:
                sample = self.read_sample()
            except Exception:
                sample = None
            if sample is not None:
                last_ok = time.time()
                live.feed(round(time.time() - t0, 2),
                          sample["p_total_w"], sample["q_total_var"])
            elif time.time() - last_ok > watchdog_s:
                print("  [watchdog] no valid read; reconnecting")
                try:
                    self.connect()
                    last_ok = time.time()
                except Exception:
                    pass
            dt = period_s - (time.time() - tick)
            if dt > 0:
                time.sleep(dt)


def _demo():
    import warnings; warnings.filterwarnings("ignore")
    from .disaggregator import detect_steps, train_step_classifier
    print("=" * 78)
    print("REAL-TIME LAYER - replay validation (online vs batch)")
    print("=" * 78)
    params = aggregator.device_params()
    clf = train_step_classifier(n_train=300, n_test=30)["rf"]

    combo = ["Foen Stufe 1", "COOLER_FAN1", "Leuchtstoffroehre"]
    df, events, active, meta = aggregator.aggregate(combo, params, mode="sequential",
                                                    rng=np.random.default_rng(3))
    print(f"\nReplaying {combo} as if arriving live:\n")
    live = LiveDisaggregator(params, clf, sps=5)
    final = replay(df, live, realtime=False)

    batch_steps = detect_steps(df)
    print(f"\nBatch detector found {len(batch_steps)} steps; "
          f"online detector emitted events above.")
    print(f"Final device set (online): {sorted(str(x) for x in final)}")
    print(f"True devices:              {sorted(combo)}")
    print("\n(Live stream keeps reading, so end-of-record steps that need lookahead")
    print(" are confirmed a few seconds later - the replay of a finite file truncates them.)")


if __name__ == "__main__":
    _demo()
