from django.shortcuts import render
from django.http import StreamingHttpResponse
import yolov5, torch
from yolov5.utils.general import (xyxy2xywh)
from yolov5.utils.plots import Annotator, colors
import cv2


def index(request):
    return render(request, 'index.html')


print(torch.cuda.is_available())

# load model
model = yolov5.load('fire.pt')

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names


def stream():
    cap = cv2.VideoCapture("input4.mp4")
    model.conf = 0.2
    model.iou = 0.1
    # model.classes = [0, 1, 2]

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: failed to capture image")
            break

        results = model(frame, augment=True)
        annotator = Annotator(frame, line_width=2, pil=not ascii)

        det = results.pred[0]
        if det is not None and len(det):

            confs = det[:, 4]
            clss = det[:, 5]

            for xywh, conf, cls in zip(det[:, :4], confs, clss):
                x, y, w, h = map(int, xywh)
                c = int(cls)

                if names[c] != "default":
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label((x, y, x + w, y + h), label, color=colors(c, True))

        im0 = annotator.result()
        image_bytes = cv2.imencode('.jpg', im0)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')


def video_feed(request):
    return StreamingHttpResponse(stream(), content_type='multipart/x-mixed-replace; boundary=frame')
