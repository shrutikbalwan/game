"""Virtual Steering Wheel: webcam hand tracking for the Racing Game."""


from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any, Deque, Optional

import cv2
import mediapipe as mp
import numpy as np

from controller_bridge import ControlInput, write_control, cleanup_bridge

WINDOW_TITLE = "Virtual Steering Wheel"
STEERING_POINT_INDEX = 9  # Requested hand point used for the steering line.
WRIST_LANDMARK_INDEX = 0

STEERING_DEAD_ZONE_DEGREES = 8.0
ACCELERATION_DISTANCE_RATIO = 0.35
CALIBRATION_DURATION_SECONDS = 1.0
CALIBRATION_STEADY_TOLERANCE_PIXELS = 25.0
ANGLE_FILTER_SIZE = 3

# Each pair is (fingertip index, matching knuckle/MCP joint index).
FINGERTIP_KNUCKLE_PAIRS = (
    (4, 2),
    (8, 5),
    (12, 9),
    (16, 13),
    (20, 17),
)

MP_HANDS = None
MP_DRAWING = None
MP_DRAWING_STYLES = None
Point = tuple[int, int]


def init_mediapipe() -> Optional[Any]:
    """Try to obtain MediaPipe solution handles safely.

    Accessing `mp.solutions.hands` can trigger native library initialization
    that may crash on some Windows installs. Return None on failure so the
    caller can run a graceful fallback instead of crashing the process.
    """
    global MP_HANDS, MP_DRAWING, MP_DRAWING_STYLES
    try:
        MP_HANDS = mp.solutions.hands
        MP_DRAWING = mp.solutions.drawing_utils
        MP_DRAWING_STYLES = mp.solutions.drawing_styles
        print("INFO: MediaPipe initialized successfully.")
        return MP_HANDS
    except BaseException as e:
        print(f"Warning: MediaPipe initialization failed ({e}). Running simulated controls.")
        MP_HANDS = None
        MP_DRAWING = None
        MP_DRAWING_STYLES = None
        return None


@dataclass
class HandInfo:
    """The useful landmarks and classification for one detected hand."""

    label: str
    landmarks: Any
    steering_point: Point
    wrist_point: Point
    is_closed_fist: bool


@dataclass
class DetectionResult:
    """All hand data extracted from one video frame."""

    hands: list[HandInfo] = field(default_factory=list)

    def hand(self, label: str) -> Optional[HandInfo]:
        """Return the first hand with the requested MediaPipe label."""
        return next((hand for hand in self.hands if hand.label == label), None)

    @property
    def left_hand(self) -> Optional[HandInfo]:
        return self.hand("Left")

    @property
    def right_hand(self) -> Optional[HandInfo]:
        return self.hand("Right")

    @property
    def both_hands_detected(self) -> bool:
        return self.left_hand is not None and self.right_hand is not None

    @property
    def closed_fist_detected(self) -> bool:
        return any(hand.is_closed_fist for hand in self.hands)


@dataclass
class CalibrationState:
    """Stores the startup neutral pose and steering-angle smoothing history."""

    is_calibrated: bool = False
    started_at: Optional[float] = None
    reference_left_point: Optional[Point] = None
    reference_right_point: Optional[Point] = None
    neutral_angle_degrees: float = 0.0
    filtered_angles: Deque[float] = field(
        default_factory=lambda: deque(maxlen=ANGLE_FILTER_SIZE)
    )

    def reset_hold(self) -> None:
        """Reset an incomplete steady-hands calibration hold."""
        self.started_at = None
        self.reference_left_point = None
        self.reference_right_point = None


@dataclass
class UIState:
    """Everything the renderer needs for the current frame."""

    calibrated: bool
    angle_degrees: Optional[float]
    steering_state: Optional[str]
    accelerating: bool
    braking: bool
    fps: float


def draw_label(
    frame: np.ndarray,
    text: str,
    origin: Point,
    color: tuple[int, int, int],
    font_scale: float = 0.65,
    thickness: int = 2,
) -> None:
    """Draw high-contrast text using an opaque dark background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    x, y = origin
    padding = 6
    cv2.rectangle(
        frame,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_steering_wheel_icon(
    frame: np.ndarray, angle_degrees: float, active: bool
) -> None:
    """Draw a compact steering-wheel icon whose spokes rotate with the angle."""
    frame_height, frame_width = frame.shape[:2]
    radius = max(20, min(42, min(frame_width, frame_height) // 10))
    center = (frame_width - radius - 18, frame_height - radius - 18)
    rim_color = (220, 220, 220) if active else (110, 110, 110)
    spoke_color = (0, 255, 0) if active else (110, 110, 110)

    cv2.circle(frame, center, radius, rim_color, 2, cv2.LINE_AA)
    cv2.circle(frame, center, max(5, radius // 6), rim_color, cv2.FILLED, cv2.LINE_AA)

    angle_radians = np.deg2rad(angle_degrees)
    for offset_degrees in (0, 120, 240):
        spoke_angle = angle_radians + np.deg2rad(offset_degrees)
        endpoint = (
            int(center[0] + radius * 0.78 * np.cos(spoke_angle)),
            int(center[1] + radius * 0.78 * np.sin(spoke_angle)),
        )
        cv2.line(frame, center, endpoint, spoke_color, 2, cv2.LINE_AA)

    marker = (
        int(center[0] + radius * 0.91 * np.cos(angle_radians)),
        int(center[1] + radius * 0.91 * np.sin(angle_radians)),
    )
    cv2.circle(frame, marker, 4, spoke_color, cv2.FILLED, cv2.LINE_AA)


def open_webcam() -> cv2.VideoCapture:
    """Open the default webcam or raise a helpful error message."""
    # Try DirectShow (DShow) backend first on Windows; fall back to MSMF auto.
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    camera = None
    for backend in backends:
        camera = cv2.VideoCapture(0, backend)
        if camera.isOpened():
            break
        camera.release()
        camera = None

    if camera is None:
        raise RuntimeError(
            "No webcam was found or it could not be opened. "
            "Check the camera connection and application permissions."
        )

    # A modest capture size keeps MediaPipe responsive on typical laptops.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Warm up the camera: the first few frames are often black/empty
    # on some Windows cameras. Discard them.
    warm_up_frames = 10
    for _ in range(warm_up_frames):
        success, frame = camera.read()
        if success and frame is not None and frame.mean() > 5.0:
            break

    return camera


def capture_frame(camera: cv2.VideoCapture) -> Optional[np.ndarray]:
    """Capture and horizontally mirror one webcam frame."""
    success, frame = camera.read()
    if not success:
        return None
    return cv2.flip(frame, 1)


def is_closed_fist(hand_landmarks: Any) -> bool:
    """Return True when every fingertip is lower than its knuckle in the image."""
    landmarks = hand_landmarks.landmark
    # OpenCV coordinates grow downward, so a greater y value means "below".
    return all(
        landmarks[fingertip].y > landmarks[knuckle].y
        for fingertip, knuckle in FINGERTIP_KNUCKLE_PAIRS
    )


def detect_hands(frame: np.ndarray, hands: Any) -> DetectionResult:
    """Run MediaPipe Hands and return hand labels plus the required landmarks."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)

    landmarks_list = results.multi_hand_landmarks or []
    handedness_list = results.multi_handedness or []
    detected_hands: list[HandInfo] = []

    for index, hand_landmarks in enumerate(landmarks_list):
        label = "Unknown"
        if index < len(handedness_list) and handedness_list[index].classification:
            label = handedness_list[index].classification[0].label

        steering_landmark = hand_landmarks.landmark[STEERING_POINT_INDEX]
        wrist_landmark = hand_landmarks.landmark[WRIST_LANDMARK_INDEX]
        steering_point = (
            int(steering_landmark.x * frame.shape[1]),
            int(steering_landmark.y * frame.shape[0]),
        )
        wrist_point = (
            int(wrist_landmark.x * frame.shape[1]),
            int(wrist_landmark.y * frame.shape[0]),
        )
        detected_hands.append(
            HandInfo(
                label=label,
                landmarks=hand_landmarks,
                steering_point=steering_point,
                wrist_point=wrist_point,
                is_closed_fist=is_closed_fist(hand_landmarks),
            )
        )

    return DetectionResult(hands=detected_hands)


def calculate_angle(left_point: Point, right_point: Point) -> float:
    """Calculate the line angle from the Left point to the Right point in degrees."""
    delta_x = right_point[0] - left_point[0]
    delta_y = right_point[1] - left_point[1]
    return float(np.degrees(np.arctan2(delta_y, delta_x)))


def normalize_angle(angle_degrees: float) -> float:
    """Wrap an angle to [-180, 180) so neutral-angle subtraction stays stable."""
    return (angle_degrees + 180.0) % 360.0 - 180.0


def calibrate(
    calibration: CalibrationState,
    left_point: Optional[Point],
    right_point: Optional[Point],
    raw_angle_degrees: Optional[float],
    now: float,
) -> bool:
    """Calibrate after both hands have remained steady for two seconds."""
    if calibration.is_calibrated:
        return True

    if left_point is None or right_point is None or raw_angle_degrees is None:
        calibration.reset_hold()
        return False

    if (
        calibration.reference_left_point is None
        or calibration.reference_right_point is None
    ):
        calibration.reference_left_point = left_point
        calibration.reference_right_point = right_point
        calibration.started_at = now
    else:
        left_displacement = float(
            np.linalg.norm(np.subtract(left_point, calibration.reference_left_point))
        )
        right_displacement = float(
            np.linalg.norm(np.subtract(right_point, calibration.reference_right_point))
        )
        if (
            left_displacement > CALIBRATION_STEADY_TOLERANCE_PIXELS
            or right_displacement > CALIBRATION_STEADY_TOLERANCE_PIXELS
        ):
            # Any noticeable movement restarts the two-second steady hold.
            calibration.reference_left_point = left_point
            calibration.reference_right_point = right_point
            calibration.started_at = now

    if (
        calibration.started_at is not None
        and now - calibration.started_at >= CALIBRATION_DURATION_SECONDS
    ):
        calibration.neutral_angle_degrees = raw_angle_degrees
        calibration.filtered_angles.clear()
        calibration.is_calibrated = True

    return calibration.is_calibrated


def apply_angle_filter(
    raw_angle_degrees: float, calibration: CalibrationState
) -> float:
    """Subtract neutral and smooth the steering angle over the latest 5 frames."""
    relative_angle = normalize_angle(
        raw_angle_degrees - calibration.neutral_angle_degrees
    )
    calibration.filtered_angles.append(relative_angle)
    return float(np.mean(calibration.filtered_angles))


def detect_acceleration(detection: DetectionResult, frame_width: int) -> bool:
    """Return True while both wrist points are closer than the acceleration threshold."""
    left_hand = detection.left_hand
    right_hand = detection.right_hand
    if left_hand is None or right_hand is None:
        return False

    wrist_distance = float(
        np.linalg.norm(np.subtract(right_hand.wrist_point, left_hand.wrist_point))
    )
    threshold = frame_width * ACCELERATION_DISTANCE_RATIO
    # Holding both wrists close together keeps W held until the gesture stops.
    return wrist_distance < threshold


def steering_state_for(angle_degrees: float) -> str:
    """Map a filtered steering angle to its screen label."""
    if -STEERING_DEAD_ZONE_DEGREES <= angle_degrees <= STEERING_DEAD_ZONE_DEGREES:
        return "STRAIGHT"
    if angle_degrees < -STEERING_DEAD_ZONE_DEGREES:
        return "LEFT"
    return "RIGHT"


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Draw the FPS counter at the top-right of the frame."""
    label = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    margin = 12
    x = frame.shape[1] - text_width - margin
    y = text_height + margin
    cv2.rectangle(
        frame,
        (x - 7, y - text_height - 7),
        (frame.shape[1] - margin + 7, y + baseline + 7),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        label,
        (x, y),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_ui(frame: np.ndarray, detection: DetectionResult, state: UIState) -> None:
    """Draw landmarks, steering feedback, gesture indicators, and FPS."""
    for hand in detection.hands:
        # MediaPipe drawing utilities may be unavailable when mp failed to
        # initialize; draw a minimal indicator instead.
        if MP_DRAWING is not None and MP_HANDS is not None:
            MP_DRAWING.draw_landmarks(
                frame,
                hand.landmarks,
                MP_HANDS.HAND_CONNECTIONS,
                MP_DRAWING_STYLES.get_default_hand_landmarks_style(),
                MP_DRAWING_STYLES.get_default_hand_connections_style(),
            )
        else:
            cv2.circle(frame, hand.steering_point, 7, (0, 255, 255), cv2.FILLED)

        label_x = max(10, hand.steering_point[0])
        label_y = max(28, hand.steering_point[1] - 12)
        draw_label(frame, hand.label, (label_x, label_y), (255, 255, 0))

    left_hand = detection.left_hand
    right_hand = detection.right_hand
    both_hands = left_hand is not None and right_hand is not None

    if both_hands:
        cv2.line(
            frame,
            left_hand.steering_point,
            right_hand.steering_point,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
        draw_label(frame, "BOTH HANDS DETECTED", (12, 36), (0, 255, 0), 0.75, 2)

        if state.calibrated and state.angle_degrees is not None:
            state_color = (0, 255, 0)
            if state.steering_state in {"LEFT", "RIGHT"}:
                state_color = (0, 165, 255)
            draw_label(
                frame,
                f"ANGLE: {state.angle_degrees:.1f} deg",
                (12, 70),
                (255, 255, 255),
                0.65,
                2,
            )
            draw_label(
                frame,
                f"STATE: {state.steering_state}",
                (12, 104),
                state_color,
                0.65,
                2,
            )
            draw_steering_wheel_icon(frame, state.angle_degrees, active=True)
        else:
            draw_label(
                frame,
                "Calibrating... hold hands steady",
                (12, 70),
                (0, 215, 255),
                0.65,
                2,
            )
            draw_steering_wheel_icon(frame, 0.0, active=False)
    else:
        draw_label(frame, "SHOW BOTH HANDS", (12, 36), (0, 0, 255), 0.75, 2)
        draw_label(frame, "ANGLE: --", (12, 70), (180, 180, 180), 0.65, 2)
        if not state.calibrated:
            draw_label(
                frame,
                "Calibrating... hold hands steady",
                (12, 104),
                (0, 215, 255),
                0.65,
                2,
            )
        draw_steering_wheel_icon(frame, 0.0, active=False)

    indicator_y = 138 if both_hands else 104
    if state.accelerating:
        draw_label(frame, "ACCELERATING", (12, indicator_y), (0, 255, 0), 0.7, 2)
    if state.braking:
        draw_label(frame, "BRAKING", (12, indicator_y), (0, 0, 255), 0.7, 2)

    draw_fps(frame, state.fps)


def build_control_input(
    calibration: CalibrationState,
    raw_angle: Optional[float],
    angle_degrees: Optional[float],
    accelerating: bool,
    braking: bool,
) -> ControlInput:
    """Build a ControlInput from the current hand tracking state."""
    control = ControlInput()
    if calibration.is_calibrated:
        control.calibrated = True
        if angle_degrees is not None:
            # INVERTED: When hands tilt left (positive angle), car should steer left
            if angle_degrees > STEERING_DEAD_ZONE_DEGREES:
                control.steer_left = True
            elif angle_degrees < -STEERING_DEAD_ZONE_DEGREES:
                control.steer_right = True
        control.accelerate = accelerating and not braking
        control.brake = braking
    return control


def main() -> None:
    """Run the webcam application. Press Q in the preview window to exit."""
    calibration = CalibrationState()
    camera: Optional[cv2.VideoCapture] = None

    try:
        camera = open_webcam()
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

        previous_frame_time = time.perf_counter()
        fps = 0.0
        mp_hands = init_mediapipe()
        if mp_hands is None:
            # Fallback simulation: allow keyboard-driven synthetic gestures so
            # the rest of the control logic (calibration, steering, pedals)
            # can be exercised without MediaPipe.
            print(
                "Warning: MediaPipe failed to initialize. Running simulated controls."
            )

            sim_angle = 0.0
            sim_target_angle = 0.0
            sim_both_hands = True
            sim_accel = False
            sim_brake = False
            instr = (
                "Sim controls: A/D adjust angle, H toggle hands, G toggle accel, "
                "B toggle brake, R reset calibration, Q quit"
            )

            while True:
                frame = capture_frame(camera)
                if frame is None:
                    print("Error: Unable to read a frame from the webcam.")
                    break

                now = time.perf_counter()
                # handle simple keyboard controls to modify the simulated state
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    if key == ord("q"):
                        break
                    # Adjust the target angle; the displayed `sim_angle`
                    # will smoothly move toward this target each frame.
                    if key == ord("a"):
                        sim_target_angle -= 15.0
                    elif key == ord("d"):
                        sim_target_angle += 15.0
                    elif key == ord("h"):
                        sim_both_hands = not sim_both_hands
                    elif key == ord("g"):
                        sim_accel = not sim_accel
                    elif key == ord("b"):
                        sim_brake = not sim_brake
                    elif key == ord("r"):
                        calibration.is_calibrated = False
                        calibration.reset_hold()

                frame_interval = now - previous_frame_time
                previous_frame_time = now
                if frame_interval > 0:
                    fps = 1.0 / frame_interval

                # Smoothly move the simulated steering angle toward the
                # requested target for a more realistic steering feel.
                # Limit target to reasonable bounds.
                sim_target_angle = max(-90.0, min(90.0, sim_target_angle))
                if frame_interval > 0:
                    # easing factor scaled by elapsed time
                    ease = 1.0 - pow(0.001, frame_interval)
                else:
                    ease = 0.2
                sim_angle += (sim_target_angle - sim_angle) * ease

                # Build a synthetic DetectionResult matching the real structure.
                frame_h, frame_w = frame.shape[:2]
                detection = DetectionResult()
                if sim_both_hands:
                    # place two steering points separated horizontally and rotated
                    center = (frame_w // 2, frame_h // 2)
                    span = int(frame_w * 0.18)
                    angle_rad = np.deg2rad(sim_angle)
                    dx = int(span * np.cos(angle_rad))
                    dy = int(span * np.sin(angle_rad))
                    left_point = (center[0] - dx, center[1] - dy)
                    right_point = (center[0] + dx, center[1] + dy)

                    left_hand = HandInfo(
                        label="Left",
                        landmarks=None,
                        steering_point=left_point,
                        wrist_point=left_point,
                        is_closed_fist=sim_brake,
                    )
                    right_hand = HandInfo(
                        label="Right",
                        landmarks=None,
                        steering_point=right_point,
                        wrist_point=right_point,
                        is_closed_fist=sim_brake,
                    )
                    detection.hands = [left_hand, right_hand]

                # Use the same control/calibration logic as the normal loop.
                left_hand = detection.left_hand
                right_hand = detection.right_hand

                raw_angle: Optional[float] = None
                if left_hand is not None and right_hand is not None:
                    raw_angle = calculate_angle(
                        left_hand.steering_point, right_hand.steering_point
                    )

                calibrate(
                    calibration,
                    left_hand.steering_point if left_hand else None,
                    right_hand.steering_point if right_hand else None,
                    raw_angle,
                    now,
                )

                acceleration_gesture = sim_accel

                angle_degrees: Optional[float] = None
                steering_state: Optional[str] = None
                accelerating = False
                braking = False

                if calibration.is_calibrated:
                    braking = detection.closed_fist_detected or sim_brake
                    accelerating = acceleration_gesture and not braking

                    if raw_angle is not None:
                        angle_degrees = apply_angle_filter(raw_angle, calibration)
                        steering_state = steering_state_for(angle_degrees)

                # Write control state to the bridge file
                control = build_control_input(
                    calibration, raw_angle, angle_degrees, accelerating, braking
                )
                write_control(control)

                draw_ui(
                    frame,
                    detection,
                    UIState(
                        calibrated=calibration.is_calibrated,
                        angle_degrees=angle_degrees,
                        steering_state=steering_state,
                        accelerating=accelerating,
                        braking=braking,
                        fps=fps,
                    ),
                )
                # Draw an on-screen slider for the simulated steering angle.
                slider_w = int(frame_w * 0.6)
                slider_h = 6
                slider_x = (frame_w - slider_w) // 2
                slider_y = frame_h - 48
                cv2.rectangle(
                    frame,
                    (slider_x, slider_y),
                    (slider_x + slider_w, slider_y + slider_h),
                    (50, 50, 50),
                    cv2.FILLED,
                )
                # map angle [-90,90] to slider position
                pos = int(
                    slider_x + (sim_angle + 90.0) / 180.0 * slider_w
                )
                cv2.circle(frame, (pos, slider_y + slider_h // 2), 8, (0, 200, 255), cv2.FILLED)
                draw_label(frame, instr, (12, frame_h - 18), (200, 200, 200), 0.6, 1)
                cv2.imshow(WINDOW_TITLE, frame)
        else:
            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as hands:
                while True:
                    frame = capture_frame(camera)
                    if frame is None:
                        print("Error: Unable to read a frame from the webcam.")
                        break

                    now = time.perf_counter()
                    detection = detect_hands(frame, hands)
                    left_hand = detection.left_hand
                    right_hand = detection.right_hand

                    raw_angle: Optional[float] = None
                    if left_hand is not None and right_hand is not None:
                        raw_angle = calculate_angle(
                            left_hand.steering_point, right_hand.steering_point
                        )

                    calibrate(
                        calibration,
                        left_hand.steering_point if left_hand else None,
                        right_hand.steering_point if right_hand else None,
                        raw_angle,
                        now,
                    )
                    acceleration_gesture = detect_acceleration(detection, frame.shape[1])

                    angle_degrees: Optional[float] = None
                    steering_state: Optional[str] = None
                    accelerating = False
                    braking = False

                    if calibration.is_calibrated:
                        # Braking works with either hand; acceleration requires both wrists.
                        braking = detection.closed_fist_detected
                        accelerating = acceleration_gesture and not braking

                        if raw_angle is not None:
                            angle_degrees = apply_angle_filter(raw_angle, calibration)
                            steering_state = steering_state_for(angle_degrees)

                    # Write control state to the bridge file
                    control = build_control_input(
                        calibration, raw_angle, angle_degrees, accelerating, braking
                    )
                    write_control(control)

                    frame_interval = now - previous_frame_time
                    previous_frame_time = now
                    if frame_interval > 0:
                        fps = 1.0 / frame_interval

                    draw_ui(
                        frame,
                        detection,
                        UIState(
                            calibrated=calibration.is_calibrated,
                            angle_degrees=angle_degrees,
                            steering_state=steering_state,
                            accelerating=accelerating,
                            braking=braking,
                            fps=fps,
                        ),
                    )
                    cv2.imshow(WINDOW_TITLE, frame)

                    # waitKey processes window events as well as checking for the exit key.
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    except RuntimeError as error:
        print(f"Error: {error}")
    except cv2.error as error:
        print(f"OpenCV error: {error}")
    finally:
        cleanup_bridge()
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
