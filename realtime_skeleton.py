# realtime_hands.py
import cv2
import time
import mediapipe as mp

# ===== 설정 =====
CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
MIN_DET_CONF, MIN_TRACK_CONF = 0.5, 0.5
PRINT_INTERVAL_SEC = 1.0     # 1초마다 출력
PRINT_PIXELS = True          # True면 x,y를 픽셀 좌표로 변환해 출력 (z는 mediapipe normalized)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def landmarks_to_list(hand_landmarks, w, h):
    """21개 랜드마크를 (x,y,z) 리스트로 변환.
    x,y: 픽셀(PRINT_PIXELS=True) 또는 [0,1] 정규화값(False)
    z: 미디어파이프 normalized depth (작을수록 카메라에 가까움)
    """
    pts = []
    for lm in hand_landmarks.landmark:
        if PRINT_PIXELS:
            x = int(lm.x * w)
            y = int(lm.y * h)
        else:
            x, y = lm.x, lm.y
        pts.append((x, y, lm.z))
    return pts

def main():
    # 맥에서 안정성을 위해 AVFoundation 백엔드를 권장
    # cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRACK_CONF,
    )

    last_print = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("카메라 프레임을 읽을 수 없습니다.")
                break

            h, w = frame.shape[:2]
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True
            out = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # 스켈레톤 그리기
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        out,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            # 1초마다 터미널 출력
            now = time.time()
            if now - last_print >= PRINT_INTERVAL_SEC:
                last_print = now
                if results.multi_hand_landmarks:
                    # handedness와 랜드마크 매칭
                    infos = []
                    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        label = None
                        if results.multi_handedness and len(results.multi_handedness) > i:
                            label = results.multi_handedness[i].classification[0].label  # 'Left' or 'Right'
                        pts = landmarks_to_list(hand_landmarks, w, h)
                        infos.append((label or f"Hand{i}", pts))

                    # 깔끔히 출력
                    print("=== Hand Landmarks (21, every 1s) ===")
                    for label, pts in infos:
                        print(f"- {label}:")
                        for j, (x, y, z) in enumerate(pts):
                            if PRINT_PIXELS:
                                print(f"  [{j:02d}] x={x:4d}, y={y:4d}, z={z:+.4f}")
                            else:
                                print(f"  [{j:02d}] x={x:.4f}, y={y:.4f}, z={z:+.4f}")
                else:
                    print("=== Hand Landmarks: None detected ===")

            cv2.imshow("MediaPipe Hands (press 'q' to quit)", out)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()