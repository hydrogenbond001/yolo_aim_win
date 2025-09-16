import cv2
import torch
import numpy as np
import mss
from ultralytics import YOLO

# 加载 YOLOv8 模型（比如 nano 版本，速度快）
model = YOLO("yolov8n.pt")  # 第一次运行会自动下载

# 屏幕捕获器
sct = mss.mss()

# 获取主屏幕尺寸
monitor = sct.monitors[1]  # [0] 是所有屏幕，[1] 是主屏幕

while True:
    # 捕获屏幕
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)

    # mss 默认是 BGRA，要转 BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # YOLO 推理
    results = model.predict(frame, conf=0.5, verbose=False)

    # 画检测结果
    annotated_frame = results[0].plot()

    # 显示
    cv2.imshow("YOLOv8 Screen Detection", cv2.resize(annotated_frame,None,fx=0.5,fy=0.5))

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
