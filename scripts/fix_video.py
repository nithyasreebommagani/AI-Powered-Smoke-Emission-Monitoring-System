import cv2

input_video = r"C:\Users\E028.26\Downloads\WhatsApp Video 2026-06-16 at 7.09.35 PM.mp4"
output_video = r"C:\Users\E028.26\Downloads\fixed_video.mp4"

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Could not open input video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps == 0:
    fps = 30

out = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)

cap.release()
out.release()

print("Fixed video saved to:", output_video)