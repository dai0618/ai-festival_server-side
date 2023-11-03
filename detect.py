import os
from ultralytics import YOLO
from PIL import Image

os.makedirs("./images", exist_ok=True)

def detect(file_path, predict_num):
    # Load a pretrained YOLOv8n-seg Segment model
    model = YOLO('yolov8n-seg.pt')

    # Run inference on an image
    results = model(file_path)  # results list

    box_list = []

    # View results
    for result in results:
        boxes = result.boxes

        for i,box in enumerate(boxes):
            if i < predict_num:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]

                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)

                im = Image.open(file_path)
                im_trimmed = im.crop((x1, y1, x2, y2))
                save_path = f'./images/trimed_image_{i}.jpg'
                im_trimmed.save(save_path)

                box_data = {
                    "x1" : x1,
                    "y1" : y1,
                    "x2" : x2,
                    "y2" : y2,
                    "path" : save_path
                }

                box_list.append(box_data)

    return (box_list)

if __name__=="__main__":
    # detect_data = detect("./get_image/test.jpg", 3)
    # print(detect_data)
    print(True)