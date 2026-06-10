"""
Dependencies
------------
  pip install numpy opencv-python matplotlib scipy
"""

import os
import struct
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes


# ════════════════════════════════════════════════════════════════════
#  ✏️  EDIT THESE VALUES THEN RUN
# ════════════════════════════════════════════════════════════════════

CINE_PATH     = r"./Measurements 08-06-2026\Measurement_1(1100).cine"  # path to your .cine file

#r"C:\My Files\Experiments\Umemura\Blue Needle\55\blue_needle_55.cine"


# ── jet direction ─────────────────────────────────────────────────────
ROTATION      = 270      # Rotate each frame BEFORE processing so that the jet
                       # always ends up flowing right → left.
                       #
                       #   0   – jet already flows right → left  (no rotation)
                       #   90  – jet flows top  → bottom  (rotate 90° CW)
                       #   180 – jet flows left → right   (rotate 180°)
                       #   270 – jet flows bottom → top   (rotate 90° CCW)
                       #
                       # After rotation, NEEDLE_LENGTH is measured from the
                       # RIGHT edge of the rotated frame, and CROP_TOP/BOTTOM
                       # refer to the TOP/BOTTOM of the rotated frame.

# ── jet geometry ─────────────────────────────────────────────────────
NEEDLE_LENGTH = 38     # px — length of the nozzle/needle INSIDE the frame,
                       #      measured from the right edge of the ROTATED frame.
                       #      Set to 0 if the nozzle exit is flush with that edge.

OFFSET        = 0      # px — length of jet that is OFF-SCREEN on the entry side
                       #      (outside the frame). Added to the measured length.
                       #      Set to 0 if the full jet is visible.

# ── detection ────────────────────────────────────────────────────────
THRESHOLD     = 175    # 0-255 intensity threshold; None = auto (Otsu)
INVERT        = True   # False = jet is BRIGHT on dark background
                       # True  = jet is DARK  on bright background

MIN_JET_PX    = 5      # minimum number of bright pixels in a column to
                       # count it as containing jet
MIN_ROWS      = 3      # minimum contiguous vertical run (px) required
                       # for a column to count as jet

GAP_TOLERANCE = 2      # number of consecutive empty columns to tolerate
                       # before declaring a breakup. Increase to 3-5 if
                       # the jet has internal dark spots / shadows.

# ── video crop (applied to the ROTATED frame) ─────────────────────────
CROP_TOP      = 0      # px to remove from the TOP    of the (rotated) output video
CROP_BOTTOM   = 0      # px to remove from the BOTTOM of the (rotated) output video

# ── calibration ──────────────────────────────────────────────────────
SCALE         = 23.25  # pixels per mm
FPS           = 5000.0 # recording frame rate
VIDEO_FPS     = 20.0   # output annotated video frame rate

OUTPUT_STEM   = None   # None = save next to the .cine file
                       # or e.g. r"C:\results\run1"

# ════════════════════════════════════════════════════════════════════
#  everything below runs automatically — no edits needed
# ════════════════════════════════════════════════════════════════════


# ── .cine reader ─────────────────────────────────────────────────────────────

class CineReader:
    """
    Minimal reader for Vision Research / Phantom .cine files.
    Auto-detects actual pixel format from frame data size.
    Supports 8-bit, 10-bit packed, 12-bit packed, and 16-bit grey frames.
    """
    _CINE_HDR_FMT  = "<2sHHHiIiIIIIQ"
    _CINE_HDR_SIZE = struct.calcsize(_CINE_HDR_FMT)
    _BMP_HDR_FMT   = "<IiiHHIIiiII"
    _BMP_HDR_SIZE  = struct.calcsize(_BMP_HDR_FMT)

    def __init__(self, path):
        self.path = path
        self._fh  = open(path, "rb")
        self._parse_headers()
        self._detect_format()

    def close(self):
        self._fh.close()

    def _parse_headers(self):
        fh  = self._fh
        raw = fh.read(self._CINE_HDR_SIZE)
        if len(raw) < self._CINE_HDR_SIZE:
            raise ValueError("File too small to be a valid .cine")
        fields = struct.unpack(self._CINE_HDR_FMT, raw)
        (ctype, headersize, compression, version,
         first_movie_image, total_image_count,
         first_image_no, image_count,
         off_image_header, off_setup, off_image_offsets,
         trigger_time) = fields
        if ctype != b"CI":
            raise ValueError(f"Not a .cine file (magic={ctype!r})")
        self.image_count      = image_count
        self.off_image_header = off_image_header

        fh.seek(off_image_header)
        bmp = struct.unpack(self._BMP_HDR_FMT, fh.read(self._BMP_HDR_SIZE))
        (bi_size, bi_width, bi_height, bi_planes, bi_bit_count,
         bi_compression, bi_size_image, bi_xppm, bi_yppm,
         bi_clr_used, bi_clr_important) = bmp
        self.width     = bi_width
        self.height    = abs(bi_height)
        self.bit_count = bi_bit_count
        self._top_down = bi_height < 0

        fh.seek(0, 2)
        file_size = fh.tell()
        fh.seek(off_image_offsets)
        raw_off    = fh.read(8 * image_count)
        offsets_64 = struct.unpack(f"<{image_count}q", raw_off)
        if offsets_64[0] < 44 or offsets_64[0] > file_size:
            fh.seek(off_image_offsets)
            self.offsets = list(struct.unpack(f"<{image_count}I",
                                              fh.read(4 * image_count)))
        else:
            self.offsets = list(offsets_64)

    def _detect_format(self):
        if self.image_count < 2:
            self.actual_bpp = self.bit_count
            return
        fh = self._fh
        fh.seek(self.offsets[0])
        ann0        = struct.unpack("<I", fh.read(4))[0]
        pixel_bytes = self.offsets[1] - self.offsets[0] - ann0
        n_pixels    = self.width * self.height
        ratio       = pixel_bytes / n_pixels
        if   abs(ratio - 1.0)  < 0.05: self.actual_bpp = 8
        elif abs(ratio - 1.25) < 0.05: self.actual_bpp = 10
        elif abs(ratio - 1.5)  < 0.05: self.actual_bpp = 12
        elif abs(ratio - 2.0)  < 0.05: self.actual_bpp = 16
        else:
            raise NotImplementedError(
                f"Unknown pixel format: {ratio:.4f} B/px")
        if self.actual_bpp != self.bit_count:
            print(f"[INFO] Header says {self.bit_count}-bit but data is "
                  f"{self.actual_bpp}-bit packed — using {self.actual_bpp}-bit.")
        else:
            print(f"[INFO] Pixel format: {self.actual_bpp}-bit.")

    def read_frame(self, index):
        fh = self._fh
        fh.seek(self.offsets[index])
        ann_size = struct.unpack("<I", fh.read(4))[0]
        fh.seek(ann_size - 4, 1)
        W, H, bpp = self.width, self.height, self.actual_bpp
        if bpp == 8:
            raw   = np.frombuffer(fh.read(W * H), dtype=np.uint8).reshape(H, W)
            frame = raw.copy()
        elif bpp == 16:
            raw   = np.frombuffer(fh.read(W * H * 2), dtype=np.uint16).reshape(H, W)
            frame = (raw >> 8).astype(np.uint8)
        elif bpp == 10:
            nb    = (W * H * 10 + 7) // 8
            frame = self._unpack_10bit(np.frombuffer(fh.read(nb), dtype=np.uint8), W, H)
        elif bpp == 12:
            nb    = (W * H * 12 + 7) // 8
            frame = self._unpack_12bit(np.frombuffer(fh.read(nb), dtype=np.uint8), W, H)
        if not self._top_down:
            frame = np.flipud(frame)
        return frame

    @staticmethod
    def _unpack_10bit(data, W, H):
        n = W * H
        pad = (5 - len(data) % 5) % 5
        if pad: data = np.concatenate([data, np.zeros(pad, dtype=np.uint8)])
        d = data.reshape(-1, 5)
        p = np.zeros((len(d), 4), dtype=np.uint16)
        p[:,0] = (d[:,0].astype(np.uint16) << 2) | (d[:,1] >> 6)
        p[:,1] = ((d[:,1] & 0x3F).astype(np.uint16) << 4) | (d[:,2] >> 4)
        p[:,2] = ((d[:,2] & 0x0F).astype(np.uint16) << 6) | (d[:,3] >> 2)
        p[:,3] = ((d[:,3] & 0x03).astype(np.uint16) << 8) | d[:,4]
        return (p.ravel()[:n] >> 2).astype(np.uint8).reshape(H, W)

    @staticmethod
    def _unpack_12bit(data, W, H):
        n = W * H
        pad = (3 - len(data) % 3) % 3
        if pad: data = np.concatenate([data, np.zeros(pad, dtype=np.uint8)])
        d = data.reshape(-1, 3)
        p = np.zeros((len(d), 2), dtype=np.uint16)
        p[:,0] = (d[:,0].astype(np.uint16) << 4) | (d[:,1] >> 4)
        p[:,1] = ((d[:,1] & 0x0F).astype(np.uint16) << 8) | d[:,2]
        return (p.ravel()[:n] >> 4).astype(np.uint8).reshape(H, W)

    def __len__(self): return self.image_count
    def __enter__(self): return self
    def __exit__(self, *_): self.close()


# ── frame rotation ────────────────────────────────────────────────────────────

def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    """
    Rotate a frame so the jet always flows right → left.

      rotation=0   – no-op             (jet already right → left)
      rotation=90  – 90° clockwise     (jet was top → bottom)
      rotation=180 – 180°              (jet was left → right)
      rotation=270 – 90° counter-CW   (jet was bottom → top)
    """
    if rotation == 0:
        return frame
    elif rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError(f"ROTATION must be 0, 90, 180, or 270. Got {rotation}.")


# ── jet tip detection ─────────────────────────────────────────────────────────

def col_has_jet(col: np.ndarray, min_jet_px: int, min_rows: int) -> bool:
    """Return True if column `col` (1-D bool array) contains a valid jet run."""
    if col.sum() < min_jet_px:
        return False
    changes = np.diff(col.astype(np.int8), prepend=0, append=0)
    starts  = np.where(changes ==  1)[0]
    ends    = np.where(changes == -1)[0]
    if len(starts) == 0:
        return False
    return bool((ends - starts).max() >= min_rows)


def find_jet_tip(binary: np.ndarray,
                 start_x: int,
                 min_jet_px: int,
                 min_rows: int,
                 gap_tolerance: int) -> int | None:
    """
    Walk LEFT from `start_x` (nozzle exit column) and return the x of the
    leftmost column that is still part of the continuous liquid column.

    `gap_tolerance` allows skipping over narrow dark gaps inside the jet
    (e.g. internal shadows, reflections) before declaring a breakup.

    Returns the jet tip x-coordinate, or None if there is no jet at start_x.
    """
    H, W = binary.shape

    # jet must exist at the nozzle exit column
    if not col_has_jet(binary[:, start_x], min_jet_px, min_rows):
        return None

    tip_x       = start_x
    gap_count   = 0

    for x in range(start_x - 1, -1, -1):
        if col_has_jet(binary[:, x], min_jet_px, min_rows):
            tip_x     = x        # extend the jet tip leftward
            gap_count = 0        # reset gap counter
        else:
            gap_count += 1
            if gap_count > gap_tolerance:
                break            # real breakup — stop here

    return int(tip_x)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if ROTATION not in (0, 90, 180, 270):
        sys.exit(f"[ERROR] ROTATION must be 0, 90, 180, or 270. Got {ROTATION}.")

    stem       = OUTPUT_STEM or os.path.splitext(CINE_PATH)[0]
    txt_path   = stem + "_lengths.txt"
    plot_path  = stem + "_lengths.png"
    video_path = stem + "_lengths.mp4"

    crop_top    = max(0, int(CROP_TOP))
    crop_bottom = max(0, int(CROP_BOTTOM))

    # ── load & rotate ─────────────────────────────────────────────────────────
    print(f"[INFO] Loading '{CINE_PATH}' …")
    reader   = CineReader(CINE_PATH)
    n_frames = len(reader)
    frames   = [rotate_frame(reader.read_frame(i), ROTATION) for i in range(n_frames)]
    reader.close()

    # H and W are now the dimensions of the ROTATED frame
    H, W = frames[0].shape[:2]
    print(f"[INFO] {n_frames} frames loaded  "
          f"(raw {reader.width}×{reader.height} px → rotated {W}×{H} px, "
          f"{reader.actual_bpp}-bit, rotation={ROTATION}°)")

    if crop_top + crop_bottom >= H:
        sys.exit(f"[ERROR] CROP_TOP+CROP_BOTTOM exceeds (rotated) frame height.")

    # nozzle exit column (jet starts here, inside the rotated frame)
    nozzle_x = int(W - 1 - NEEDLE_LENGTH)
    print(f"[INFO] Nozzle exit column (rotated frame): x={nozzle_x} px")

    # ── threshold ─────────────────────────────────────────────────────────────
    if THRESHOLD is None:
        thresh = int(cv2.threshold(frames[0], 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
        print(f"[INFO] Auto threshold (Otsu): {thresh}")
    else:
        thresh = int(THRESHOLD)
        print(f"[INFO] Manual threshold: {thresh}")

    # ── per-frame measurement ──────────────────────────────────────────────────
    times_s    = []
    tips_x     = []
    lengths_px = []
    lengths_mm = []

    for i, frame in enumerate(frames):
        t = i / FPS

        # binarise once per frame
        img    = frame.astype(np.float32)
        if INVERT:
            img = 255.0 - img
        binary = binary_fill_holes(img > thresh)

        tip_x = find_jet_tip(binary, nozzle_x, MIN_JET_PX, MIN_ROWS, GAP_TOLERANCE)

        if tip_x is not None:
            l_px = int(nozzle_x - tip_x) + int(OFFSET)
            l_mm = l_px / SCALE
        else:
            l_px = 0
            l_mm = 0.0

        times_s.append(t)
        tips_x.append(tip_x)
        lengths_px.append(l_px)
        lengths_mm.append(l_mm)

    times_s    = np.array(times_s)
    lengths_px = np.array(lengths_px, dtype=float)
    lengths_mm = np.array(lengths_mm)

    # ── text output ───────────────────────────────────────────────────────────
    with open(txt_path, "w") as f:
        f.write("# Jet breakup length measurement\n")
        f.write(f"# Input         : {CINE_PATH}\n")
        f.write(f"# Rotation      : {ROTATION} deg\n")
        f.write(f"# Nozzle exit x : {nozzle_x} px  (in rotated frame)\n")
        f.write(f"# Needle length : {NEEDLE_LENGTH} px  ({NEEDLE_LENGTH/SCALE:.3f} mm)\n")
        f.write(f"# Offset        : {OFFSET} px  ({OFFSET/SCALE:.3f} mm)\n")
        f.write(f"# Scale         : {SCALE} px/mm\n")
        f.write(f"# FPS           : {FPS}\n")
        f.write(f"# Threshold     : {thresh}\n")
        f.write(f"# Inverted      : {INVERT}\n")
        f.write(f"# Gap tolerance : {GAP_TOLERANCE}\n#\n")
        f.write("frame\ttime_s\ttip_x_px\tlength_px\tlength_mm\n")
        for i in range(n_frames):
            tx = tips_x[i] if tips_x[i] is not None else -1
            f.write(f"{i}\t{times_s[i]:.6f}\t{tx}\t"
                    f"{lengths_px[i]:.1f}\t{lengths_mm[i]:.4f}\n")
    print(f"[INFO] Length data → {txt_path}")

    # ── plot ──────────────────────────────────────────────────────────────────
    valid_mask = lengths_mm > 0
    fig, axes  = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(times_s * 1e3, lengths_mm, color="steelblue", linewidth=0.8)
    axes[0].set_ylabel("Jet length  [mm]", fontsize=11)
    axes[0].set_title("Jet breakup length vs. time", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    if valid_mask.any():
        mean_l = lengths_mm[valid_mask].mean()
        axes[0].axhline(mean_l, color="tomato", linestyle="--",
                        linewidth=1.2, label=f"mean = {mean_l:.3f} mm")
        axes[0].legend(fontsize=10)

    axes[1].plot(times_s * 1e3, lengths_px, color="darkorange", linewidth=0.8)
    axes[1].set_ylabel("Jet length  [px]", fontsize=11)
    axes[1].set_xlabel("Time  [ms]", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[INFO] Plot        → {plot_path}")

    # ── annotated video ───────────────────────────────────────────────────────
    out_H  = H - crop_top - crop_bottom
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw     = cv2.VideoWriter(video_path, fourcc, float(VIDEO_FPS), (int(W), int(out_H)))

    for i, frame in enumerate(frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # nozzle exit line (orange)
        cv2.line(bgr, (int(nozzle_x), 0), (int(nozzle_x), int(H - 1)),
                 (0, 180, 255), 1)

        tx = tips_x[i]

        if tx is not None:
            tx = int(tx)

            # jet tip line (green)
            cv2.line(bgr, (tx, 0), (tx, int(H - 1)), (0, 255, 100), 1)

            # arrow from tip → nozzle
            mid_y = int(H // 2)
            cv2.arrowedLine(bgr,
                            (tx,            mid_y),
                            (int(nozzle_x), mid_y),
                            (255, 200, 0), 2, tipLength=0.015)

            # length label
            span    = int(nozzle_x) - tx
            label_x = int(tx + max(5, span // 2 - 60))
            label_y = int(max(mid_y - 16, 20))
            cv2.putText(bgr, f"{lengths_mm[i]:.2f} mm",
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(bgr, f"({int(lengths_px[i])} px)",
                        (label_x, label_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

            # tip dot
            cv2.circle(bgr, (tx, mid_y), 5, (0, 255, 100), -1)

        else:
            cv2.putText(bgr, "NO JET AT NOZZLE",
                        (int(W // 2 - 100), int(H // 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # frame / time stamp
        cv2.putText(bgr, f"f={i:04d}  t={times_s[i]*1e3:.3f} ms",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # crop
        if crop_top > 0 or crop_bottom > 0:
            bgr = bgr[crop_top: H - crop_bottom, :]

        vw.write(bgr)

    vw.release()
    print(f"[INFO] Video       → {video_path}")

    # ── summary ───────────────────────────────────────────────────────────────
    valid = lengths_mm[valid_mask]
    if len(valid):
        print(f"\n{'─'*50}")
        print(f"  Frames processed    : {n_frames}")
        print(f"  Frames with jet     : {len(valid)}")
        print(f"  Mean length         : {valid.mean():.4f} mm")
        print(f"  Std  length         : {valid.std():.4f} mm")
        print(f"  Min  length         : {valid.min():.4f} mm")
        print(f"  Max  length         : {valid.max():.4f} mm")
        print(f"  Needle length used  : {NEEDLE_LENGTH} px  ({NEEDLE_LENGTH/SCALE:.3f} mm)")
        print(f"  Offset used         : {OFFSET} px  ({OFFSET/SCALE:.3f} mm)")
        print(f"{'─'*50}\n")
    else:
        print("[WARN] No jet detected at the nozzle exit column. "
              "Try adjusting THRESHOLD, INVERT, or NEEDLE_LENGTH.")


main()
