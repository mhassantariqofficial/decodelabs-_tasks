import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import pytesseract
import os
import sys

CONFIDENCE_THRESHOLD = 0.80

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROTOTXT_PATH = os.path.join(SCRIPT_DIR, "MobileNetSSD_deploy.prototxt")
MODEL_PATH = os.path.join(SCRIPT_DIR, "MobileNetSSD_deploy.caffemodel")

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def find_tesseract():
    if sys.platform.startswith("win"):
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
    return None


tesseract_path = find_tesseract()
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, blurred, thresh


def run_ocr(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the selected image file.")

    gray, blurred, thresh = preprocess_image(image)

    try:
        text = pytesseract.image_to_string(thresh).strip()
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract engine not found. Install it from "
            "github.com/UB-Mannheim/tesseract/wiki, or set the path manually "
            "by editing pytesseract.pytesseract.tesseract_cmd in this script."
        )

    cv2.imwrite(output_path, thresh)
    return text


def run_object_detection(image_path, output_path):
    if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model files not found. Place MobileNetSSD_deploy.prototxt and "
            "MobileNetSSD_deploy.caffemodel in the same folder as this script."
        )

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the selected image file.")

    preprocess_image(image)

    h, w = image.shape[:2]
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    results = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        if confidence >= CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx] if idx < len(CLASSES) else "unknown"

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{label}: {confidence * 100:.1f}%"
            cv2.putText(image, label_text, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            results.append((label, confidence))

    cv2.imwrite(output_path, image)
    return results


class VisionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Recognition - OCR and Object Detection")
        self.root.geometry("460x460")
        self.root.configure(bg="#1e1e2f")
        self.root.resizable(False, False)

        self.image_path = None

        title = tk.Label(root, text="Image Recognition Pipeline", font=("Segoe UI", 14, "bold"),
                          bg="#1e1e2f", fg="white")
        title.pack(pady=(15, 5))

        subtitle = tk.Label(root, text="Preprocessing + OCR / Object Detection | 80% Confidence Gate",
                             font=("Segoe UI", 9), bg="#1e1e2f", fg="#a0a0b0")
        subtitle.pack(pady=(0, 5))

        status_text = f"Tesseract: {'found' if tesseract_path else 'not found'}"
        model_text = f"Detection model: {'found' if os.path.exists(MODEL_PATH) else 'not found'}"
        status = tk.Label(root, text=f"{status_text}  |  {model_text}", font=("Segoe UI", 8),
                           bg="#1e1e2f", fg="#6e6e80")
        status.pack(pady=(0, 10))

        select_btn = tk.Button(root, text="Select Image", command=self.select_image,
                                bg="#2b2b3d", fg="white", relief="flat",
                                font=("Segoe UI", 10))
        select_btn.pack(pady=5, ipadx=10, ipady=5)

        self.file_label = tk.Label(root, text="No image selected", font=("Segoe UI", 9),
                                    bg="#1e1e2f", fg="#a0a0b0")
        self.file_label.pack(pady=(0, 10))

        btn_frame = tk.Frame(root, bg="#1e1e2f")
        btn_frame.pack(pady=5)

        ocr_btn = tk.Button(btn_frame, text="Run OCR", command=self.handle_ocr,
                             bg="#4e9af1", fg="white", relief="flat",
                             font=("Segoe UI", 10, "bold"))
        ocr_btn.pack(side="left", padx=10, ipadx=10, ipady=5)

        detect_btn = tk.Button(btn_frame, text="Run Object Detection", command=self.handle_detection,
                                bg="#4e9af1", fg="white", relief="flat",
                                font=("Segoe UI", 10, "bold"))
        detect_btn.pack(side="left", padx=10, ipadx=10, ipady=5)

        self.result_text = tk.Text(root, height=11, width=52, bg="#2b2b3d", fg="white",
                                    font=("Consolas", 9), state="disabled")
        self.result_text.pack(pady=15)

    def select_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if path:
            self.image_path = path
            self.file_label.config(text=os.path.basename(path))

    def write_result(self, text):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.config(state="disabled")

    def handle_ocr(self):
        if not self.image_path:
            messagebox.showerror("No Image", "Please select an image first.")
            return

        output_path = os.path.join(SCRIPT_DIR, "ocr_output.png")

        try:
            text = run_ocr(self.image_path, output_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        if text:
            result = f"Extracted Text:\n\n{text}\n\nProcessed image saved as ocr_output.png"
        else:
            result = "No confident text found.\n\nProcessed image saved as ocr_output.png"

        self.write_result(result)

    def handle_detection(self):
        if not self.image_path:
            messagebox.showerror("No Image", "Please select an image first.")
            return

        output_path = os.path.join(SCRIPT_DIR, "detection_output.png")

        try:
            results = run_object_detection(self.image_path, output_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        if results:
            lines = [f"{label}: {conf * 100:.1f}%" for label, conf in results]
            result = "Detected Objects (>= 80% confidence):\n\n" + "\n".join(lines)
            result += "\n\nProcessed image saved as detection_output.png"
        else:
            result = "No objects met the 80% confidence threshold.\n\nProcessed image saved as detection_output.png"

        self.write_result(result)


if __name__ == "__main__":
    root = tk.Tk()
    app = VisionGUI(root)
    root.mainloop()
