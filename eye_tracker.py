#!/usr/bin/env python3
"""Eye-gaze based tmux pane switcher using webcam and MediaPipe."""

import argparse
import os
import subprocess
import sys
import time

import cv2
import mediapipe as mp
import numpy as np


# MediaPipe Face Mesh landmark indices (478 landmarks with iris refinement)
# Left eye
LEFT_IRIS_CENTER = 468
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

# Right eye
RIGHT_IRIS_CENTER = 473
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Head pose estimation landmarks (nose tip, chin, left/right eye outer, left/right mouth)
HEAD_POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

# 3D model points for head pose (approximate, in arbitrary units)
HEAD_POSE_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -63.6, -12.5),    # Chin
    (-43.3, 32.7, -26.0),   # Left eye outer corner
    (43.3, 32.7, -26.0),    # Right eye outer corner
    (-28.9, -28.9, -24.1),  # Left mouth corner
    (28.9, -28.9, -24.1),   # Right mouth corner
], dtype=np.float64)

# Path to the face landmarker model (relative to this script)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")


class GazeEstimator:
    """Extracts gaze features from webcam frames using MediaPipe FaceLandmarker.

    Returns a 4-element feature vector: (iris_ratio_x, iris_ratio_y, head_yaw, head_pitch)
    This provides more information than raw iris ratios alone, improving calibration accuracy.
    """

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            print(f"エラー: モデルファイルが見つかりません: {MODEL_PATH}")
            print("README.mdのセットアップ手順を参照してください。")
            sys.exit(1)

        base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def process_frame(self, frame: np.ndarray) -> np.ndarray | None:
        """Process a BGR frame and return feature vector [iris_x, iris_y, yaw, pitch] or None."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_timestamp_ms += 33  # ~30fps
        result = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        h, w = frame.shape[:2]

        # Iris ratios
        left_h = self._ratio(landmarks, LEFT_IRIS_CENTER, LEFT_EYE_INNER, LEFT_EYE_OUTER, "x")
        left_v = self._ratio(landmarks, LEFT_IRIS_CENTER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, "y")
        right_h = self._ratio(landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, "x")
        right_v = self._ratio(landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, "y")

        iris_x = (left_h + right_h) / 2.0
        iris_y = (left_v + right_v) / 2.0

        # Head pose estimation
        yaw, pitch = self._estimate_head_pose(landmarks, w, h)

        return np.array([iris_x, iris_y, yaw, pitch])

    def _ratio(self, landmarks, iris_idx: int, a_idx: int, b_idx: int, axis: str) -> float:
        iris = getattr(landmarks[iris_idx], axis)
        a = getattr(landmarks[a_idx], axis)
        b = getattr(landmarks[b_idx], axis)
        denom = b - a
        if abs(denom) < 1e-6:
            return 0.5
        return (iris - a) / denom

    def _estimate_head_pose(self, landmarks, img_w: int, img_h: int) -> tuple[float, float]:
        """Estimate head yaw and pitch in degrees using solvePnP."""
        image_points = np.array([
            (landmarks[idx].x * img_w, landmarks[idx].y * img_h)
            for idx in HEAD_POSE_LANDMARKS
        ], dtype=np.float64)

        focal_length = img_w
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vec, _ = cv2.solvePnP(
            HEAD_POSE_3D_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return (0.0, 0.0)

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        # Decompose rotation matrix to Euler angles
        sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
        pitch = np.degrees(np.arctan2(-rotation_mat[2, 0], sy))
        yaw = np.degrees(np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2]))

        # Normalize to roughly [-1, 1] range (typical head rotation ~30 degrees max)
        yaw_norm = np.clip(yaw / 30.0, -1.0, 1.0)
        pitch_norm = np.clip(pitch / 30.0, -1.0, 1.0)

        return (yaw_norm, pitch_norm)

    def close(self):
        self.landmarker.close()


class Calibrator:
    """Maps gaze feature vectors to screen coordinates via 9-point GUI calibration
    with 2nd-order polynomial regression for non-linear gaze mapping."""

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.coeffs_x: np.ndarray | None = None
        self.coeffs_y: np.ndarray | None = None

    def run_calibration(
        self,
        gaze_estimator: GazeEstimator,
        cap: cv2.VideoCapture,
        sample_frames: int = 45,
    ) -> bool:
        """Run GUI-based 9-point calibration. Returns True on success."""
        margin_x = int(self.screen_width * 0.08)
        margin_y = int(self.screen_height * 0.08)
        sw, sh = self.screen_width, self.screen_height

        # 9-point grid (3x3)
        points = []
        for row in range(3):
            for col in range(3):
                x = margin_x + col * (sw - 2 * margin_x) // 2
                y = margin_y + row * (sh - 2 * margin_y) // 2
                points.append((x, y))

        gaze_samples = []
        screen_coords = []

        # Create fullscreen calibration window
        win_name = "Calibration"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print("\n=== キャリブレーション (9点) ===")
        print("画面に表示される緑の点を見つめてください。自動で進みます。\n")

        for i, (sx, sy) in enumerate(points):
            # Show target dot with countdown
            collecting = False
            countdown_start = time.monotonic()
            SETTLE_TIME = 1.0  # seconds to settle before collecting
            samples = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)

                elapsed = time.monotonic() - countdown_start

                # Draw calibration screen
                calib_frame = np.zeros((sh, sw, 3), dtype=np.uint8)

                # Draw all points (dim)
                for j, (px, py) in enumerate(points):
                    color = (40, 40, 40) if j != i else (0, 255, 0)
                    if j < i:
                        color = (0, 80, 0)  # completed points
                    cv2.circle(calib_frame, (px, py), 20, color, -1)

                # Current target: pulsing animation
                pulse = int(10 + 10 * np.sin(elapsed * 4))
                cv2.circle(calib_frame, (sx, sy), pulse, (0, 255, 0), -1)
                cv2.circle(calib_frame, (sx, sy), 3, (255, 255, 255), -1)

                # Progress text
                if elapsed < SETTLE_TIME:
                    remaining = SETTLE_TIME - elapsed
                    cv2.putText(
                        calib_frame,
                        f"Point {i + 1}/9 - Look at the green dot ({remaining:.1f}s)",
                        (sw // 2 - 250, sh - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1,
                    )
                else:
                    # Collecting samples
                    gaze = gaze_estimator.process_frame(frame)
                    if gaze is not None:
                        samples.append(gaze)

                    progress = len(samples) / sample_frames
                    bar_w = int(300 * progress)
                    cv2.rectangle(calib_frame, (sw // 2 - 150, sh - 30), (sw // 2 - 150 + bar_w, sh - 20), (0, 200, 255), -1)
                    cv2.rectangle(calib_frame, (sw // 2 - 150, sh - 30), (sw // 2 + 150, sh - 20), (100, 100, 100), 1)
                    cv2.putText(
                        calib_frame,
                        f"Point {i + 1}/9 - Collecting... ({len(samples)}/{sample_frames})",
                        (sw // 2 - 200, sh - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1,
                    )

                    if len(samples) >= sample_frames:
                        break

                cv2.imshow(win_name, calib_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    cv2.destroyWindow(win_name)
                    return False

            if len(samples) < 10:
                print(f"  ✗ Point {i + 1}: 十分なサンプルが取れませんでした。")
                cv2.destroyWindow(win_name)
                return False

            # Use median for robustness against outliers
            sample_arr = np.array(samples)
            median_features = np.median(sample_arr, axis=0)
            gaze_samples.append(median_features)
            screen_coords.append((sx, sy))
            print(f"  ✓ Point {i + 1}/9: iris=({median_features[0]:.4f}, {median_features[1]:.4f}) head=({median_features[2]:.4f}, {median_features[3]:.4f})")

        cv2.destroyWindow(win_name)
        self._compute_transform(gaze_samples, screen_coords)
        print("\nキャリブレーション完了!\n")
        return True

    def _compute_transform(
        self,
        gaze_samples: list[np.ndarray],
        screen_coords: list[tuple[int, int]],
    ) -> None:
        """Compute 2nd-order polynomial regression from gaze features to screen coordinates.

        Features: [ix, iy, yaw, pitch] -> polynomial basis:
        [ix, iy, yaw, pitch, ix^2, iy^2, ix*iy, ix*yaw, iy*pitch, 1]
        """
        A = np.array([self._poly_basis(g) for g in gaze_samples])
        bx = np.array([sx for sx, _ in screen_coords])
        by = np.array([sy for _, sy in screen_coords])

        self.coeffs_x = np.linalg.lstsq(A, bx, rcond=None)[0]
        self.coeffs_y = np.linalg.lstsq(A, by, rcond=None)[0]

    def _poly_basis(self, g: np.ndarray) -> np.ndarray:
        """Build polynomial feature vector from raw gaze features."""
        ix, iy, yaw, pitch = g[0], g[1], g[2], g[3]
        return np.array([
            ix, iy, yaw, pitch,
            ix * ix, iy * iy,
            ix * iy,
            ix * yaw, iy * pitch,
            1.0,
        ])

    def map_gaze_to_screen(self, gaze_features: np.ndarray) -> tuple[int, int]:
        """Map gaze feature vector to screen pixel coordinates."""
        if self.coeffs_x is None or self.coeffs_y is None:
            return (self.screen_width // 2, self.screen_height // 2)

        v = self._poly_basis(gaze_features)
        sx = int(np.clip(v @ self.coeffs_x, 0, self.screen_width - 1))
        sy = int(np.clip(v @ self.coeffs_y, 0, self.screen_height - 1))
        return (sx, sy)


class TmuxController:
    """Reads tmux pane layout and switches panes based on screen coordinates."""

    def __init__(
        self,
        terminal_x: int = 0,
        terminal_y: int = 0,
        char_width: float = 8.0,
        char_height: float = 16.0,
    ):
        self.terminal_x = terminal_x
        self.terminal_y = terminal_y
        self.char_width = char_width
        self.char_height = char_height
        self.panes: list[dict] = []

    def refresh_panes(self) -> list[dict]:
        """Fetch current tmux pane layout and convert to pixel rectangles."""
        try:
            result = subprocess.run(
                [
                    "tmux",
                    "list-panes",
                    "-F",
                    "#{pane_id} #{pane_left} #{pane_top} #{pane_width} #{pane_height}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("エラー: tmuxセッションが見つかりません。tmux内で実行してください。")
            return []

        self.panes = []
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            pane_id = parts[0]
            left = int(parts[1])
            top = int(parts[2])
            width = int(parts[3])
            height = int(parts[4])

            px_left = self.terminal_x + left * self.char_width
            px_top = self.terminal_y + top * self.char_height
            px_right = px_left + width * self.char_width
            px_bottom = px_top + height * self.char_height

            self.panes.append(
                {
                    "id": pane_id,
                    "left": px_left,
                    "top": px_top,
                    "right": px_right,
                    "bottom": px_bottom,
                    "chars": {"left": left, "top": top, "width": width, "height": height},
                }
            )

        return self.panes

    def find_pane_at(self, screen_x: int, screen_y: int) -> str | None:
        """Find which pane contains the given screen coordinate."""
        for pane in self.panes:
            if (
                pane["left"] <= screen_x < pane["right"]
                and pane["top"] <= screen_y < pane["bottom"]
            ):
                return pane["id"]
        return None

    def switch_to_pane(self, pane_id: str) -> None:
        """Switch tmux focus to the given pane and show feedback."""
        subprocess.run(
            ["tmux", "select-pane", "-t", pane_id],
            capture_output=True,
        )
        subprocess.run(
            ["tmux", "display-message", f"[Eye Tracker] Switched to {pane_id}"],
            capture_output=True,
        )
        subprocess.Popen(
            ["afplay", "/System/Library/Sounds/Tink.aiff"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def get_active_pane_id(self) -> str | None:
        """Get the currently active pane ID."""
        try:
            result = subprocess.run(
                ["tmux", "display-message", "-p", "#{pane_id}"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None


def get_screen_size() -> tuple[int, int]:
    """Get screen resolution using system_profiler on macOS."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if "Resolution" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "x" and i > 0 and i + 1 < len(parts):
                        w = int(parts[i - 1])
                        h = int(parts[i + 1])
                        if "Retina" in line:
                            w //= 2
                            h //= 2
                        return (w, h)
    except Exception:
        pass
    return (1920, 1080)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="視線トラッキングでtmux paneを自動切り替え"
    )
    parser.add_argument(
        "--dwell-time", type=float, default=0.7,
        help="pane切り替えまでの注視時間（秒）。デフォルト: 0.7",
    )
    parser.add_argument(
        "--char-width", type=float, default=8.0,
        help="ターミナルの1文字あたりのピクセル幅。デフォルト: 8.0",
    )
    parser.add_argument(
        "--char-height", type=float, default=16.0,
        help="ターミナルの1文字あたりのピクセル高さ。デフォルト: 16.0",
    )
    parser.add_argument(
        "--terminal-x", type=int, default=0,
        help="ターミナルウィンドウの画面上のX座標。デフォルト: 0",
    )
    parser.add_argument(
        "--terminal-y", type=int, default=0,
        help="ターミナルウィンドウの画面上のY座標。デフォルト: 0",
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.2,
        help="EMA平滑化係数（0=最大平滑化, 1=平滑化なし）。デフォルト: 0.2",
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="デバッグプレビューウィンドウを無効にする",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="カメラデバイスインデックス。デフォルト: 0",
    )
    return parser.parse_args()


def draw_debug_overlay(
    frame: np.ndarray,
    screen_pos: tuple[int, int] | None,
    current_pane: str | None,
    target_pane: str | None,
    dwell_progress: float,
    switch_flash: float = 0.0,
    screen_w: int = 1920,
    screen_h: int = 1080,
) -> np.ndarray:
    """Draw debug information on the camera frame."""
    h, w = frame.shape[:2]

    # Green flash overlay on pane switch
    if switch_flash > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
        alpha = min(switch_flash, 0.4)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(
            frame, f"SWITCHED -> {current_pane}",
            (w // 2 - 120, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
        )

    # Draw gaze cursor as a minimap
    if screen_pos is not None:
        # Mini screen map in top-right corner
        map_w, map_h = 160, 100
        map_x, map_y = w - map_w - 10, 10
        cv2.rectangle(frame, (map_x, map_y), (map_x + map_w, map_y + map_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (map_x, map_y), (map_x + map_w, map_y + map_h), (150, 150, 150), 1)
        # Gaze point on minimap
        gx = map_x + int(screen_pos[0] / screen_w * map_w)
        gy = map_y + int(screen_pos[1] / screen_h * map_h)
        cv2.circle(frame, (gx, gy), 4, (0, 255, 0), -1)

        cv2.putText(
            frame, f"({screen_pos[0]}, {screen_pos[1]})",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1,
        )

    pane_text = f"Pane: {current_pane or 'N/A'}"
    if target_pane and target_pane != current_pane:
        pane_text += f" -> {target_pane} ({dwell_progress:.0%})"
    cv2.putText(frame, pane_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

    # Dwell progress bar
    if dwell_progress > 0:
        bar_w = int(200 * dwell_progress)
        cv2.rectangle(frame, (10, 65), (10 + bar_w, 75), (0, 200, 255), -1)
        cv2.rectangle(frame, (10, 65), (210, 75), (100, 100, 100), 1)

    cv2.putText(
        frame, "'q' quit / 'c' recalibrate",
        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
    )

    return frame


def main() -> None:
    args = parse_args()

    # Screen size
    screen_w, screen_h = get_screen_size()
    print(f"画面サイズ: {screen_w}x{screen_h}")

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("エラー: カメラを開けません。")
        print("システム環境設定 > プライバシーとセキュリティ > カメラ でアクセスを許可してください。")
        sys.exit(1)
    print("カメラ起動OK")

    # Initialize components
    gaze_estimator = GazeEstimator()
    calibrator = Calibrator(screen_w, screen_h)
    tmux_ctrl = TmuxController(
        terminal_x=args.terminal_x,
        terminal_y=args.terminal_y,
        char_width=args.char_width,
        char_height=args.char_height,
    )

    # Check tmux
    panes = tmux_ctrl.refresh_panes()
    if not panes:
        print("エラー: tmux paneが見つかりません。tmuxセッション内で実行してください。")
        cap.release()
        gaze_estimator.close()
        sys.exit(1)

    print(f"検出されたtmux pane数: {len(panes)}")
    for p in panes:
        c = p["chars"]
        print(f"  {p['id']}: {c['width']}x{c['height']} at ({c['left']},{c['top']})")

    # Calibration
    if not calibrator.run_calibration(gaze_estimator, cap):
        print("キャリブレーション失敗。終了します。")
        cap.release()
        gaze_estimator.close()
        sys.exit(1)

    # Main loop
    current_pane = tmux_ctrl.get_active_pane_id()
    dwell_target: str | None = None
    dwell_start: float | None = None
    smooth_features: np.ndarray | None = None
    alpha = args.smoothing
    pane_refresh_interval = 5.0
    last_pane_refresh = time.monotonic()
    switch_flash = 0.0
    FLASH_DURATION = 0.5
    last_switch_time: float | None = None

    print("=== トラッキング開始 ===")
    print("'q'キーで終了、'c'キーで再キャリブレーション\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            features = gaze_estimator.process_frame(frame)

            screen_pos = None
            target_pane = None
            dwell_progress = 0.0

            if features is not None:
                # Apply EMA smoothing to all features
                if smooth_features is None:
                    smooth_features = features.copy()
                else:
                    smooth_features = alpha * features + (1 - alpha) * smooth_features

                screen_pos = calibrator.map_gaze_to_screen(smooth_features)

                # Periodically refresh pane layout
                now = time.monotonic()
                if now - last_pane_refresh > pane_refresh_interval:
                    tmux_ctrl.refresh_panes()
                    last_pane_refresh = now

                target_pane = tmux_ctrl.find_pane_at(*screen_pos)

                # Dwell logic
                if target_pane is not None and target_pane != current_pane:
                    if target_pane == dwell_target:
                        elapsed = time.monotonic() - (dwell_start or time.monotonic())
                        dwell_progress = min(elapsed / args.dwell_time, 1.0)
                        if elapsed >= args.dwell_time:
                            tmux_ctrl.switch_to_pane(target_pane)
                            print(f"  → Pane切り替え: {current_pane} → {target_pane}")
                            current_pane = target_pane
                            dwell_target = None
                            dwell_start = None
                            dwell_progress = 0.0
                            last_switch_time = time.monotonic()
                    else:
                        dwell_target = target_pane
                        dwell_start = time.monotonic()
                        dwell_progress = 0.0
                else:
                    dwell_target = None
                    dwell_start = None
            else:
                smooth_features = None
                dwell_target = None
                dwell_start = None

            # Update flash intensity
            if last_switch_time is not None:
                elapsed_since_switch = time.monotonic() - last_switch_time
                if elapsed_since_switch < FLASH_DURATION:
                    switch_flash = 1.0 - (elapsed_since_switch / FLASH_DURATION)
                else:
                    switch_flash = 0.0
                    last_switch_time = None

            # Debug preview
            if not args.no_preview:
                frame = draw_debug_overlay(
                    frame, screen_pos, current_pane, target_pane,
                    dwell_progress, switch_flash, screen_w, screen_h,
                )
                cv2.imshow("Eye Tracker - Tmux Pane Switcher", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                print("\n再キャリブレーション...")
                smooth_features = None
                calibrator.run_calibration(gaze_estimator, cap)

    except KeyboardInterrupt:
        print("\n中断されました。")
    finally:
        cap.release()
        gaze_estimator.close()
        cv2.destroyAllWindows()
        print("終了しました。")


if __name__ == "__main__":
    main()
