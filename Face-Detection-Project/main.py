
import cv2
import os
import time
import tkinter as tk

from datetime import datetime
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO


# ---------------- CONFIGURATION ----------------

os.makedirs("results", exist_ok=True)

# Face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Face detection model not found!")

# Load YOLO object detection model
print("Loading AI object detection model...")
model = YOLO("yolo11n.pt")
print("YOLO model loaded successfully!")


class FaceDetectionApp:

    def __init__(self, root):

        self.root = root
        self.root.title("AI Face and Object Detection System")
        self.root.geometry("1100x800")
        self.root.configure(bg="#f0f0f0")

        self.camera = None
        self.camera_running = False

        self.photo = None
        self.result_image = None
        self.uploaded_image = None

        self.auto_capture = True
        self.last_capture_time = 0
        self.capture_interval = 3

        self.total_captures = 0

        # ---------------- TITLE ----------------

        tk.Label(
            root,
            text="AI FACE & OBJECT DETECTION SYSTEM",
            font=("Arial", 24, "bold"),
            fg="#333399",
            bg="#f0f0f0"
        ).pack(pady=15)

        # ---------------- IMAGE DISPLAY ----------------

        self.image_label = tk.Label(
            root,
            text="Start camera or upload an image",
            bg="white",
            font=("Arial", 16),
            width=100,
            height=25
        )

        self.image_label.pack(pady=10)

        # ---------------- BUTTONS ----------------

        frame = tk.Frame(root, bg="#f0f0f0")
        frame.pack(pady=10)

        buttons = [
            ("Upload Image", self.upload_image, "#333399"),
            ("Start Camera", self.start_camera, "green"),
            ("Stop Camera", self.stop_camera, "red"),
            ("Save Result", self.save_result, "orange"),
            ("Open Results", self.open_results, "#800080")
        ]

        for i, (name, command, color) in enumerate(buttons):

            tk.Button(
                frame,
                text=name,
                command=command,
                bg=color,
                fg="white",
                font=("Arial", 10, "bold"),
                width=15,
                height=2
            ).grid(row=0, column=i, padx=5)

        # ---------------- AUTO CAPTURE ----------------

        self.auto_var = tk.BooleanVar(value=True)

        tk.Checkbutton(
            root,
            text="Enable Automatic Capture",
            variable=self.auto_var,
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            fg="green"
        ).pack(pady=5)

        # ---------------- RESULTS ----------------

        self.face_label = tk.Label(
            root,
            text="Faces: 0",
            font=("Arial", 14, "bold"),
            fg="green",
            bg="#f0f0f0"
        )
        self.face_label.pack()

        self.object_label = tk.Label(
            root,
            text="Objects: 0",
            font=("Arial", 14, "bold"),
            fg="#333399",
            bg="#f0f0f0"
        )
        self.object_label.pack()

        self.capture_label = tk.Label(
            root,
            text="Auto Captures: 0",
            font=("Arial", 12, "bold"),
            fg="#800080",
            bg="#f0f0f0"
        )
        self.capture_label.pack(pady=5)

        self.status_label = tk.Label(
            root,
            text="System Ready",
            font=("Arial", 11),
            bg="#f0f0f0"
        )
        self.status_label.pack()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_app
        )

    # ---------------- FACE DETECTION ----------------

    def detect_faces(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(35, 35)
        )

        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "HUMAN FACE",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        return frame, len(faces)

    # ---------------- OBJECT DETECTION ----------------

    def detect_objects(self, frame):

        results = model.predict(
            source=frame,
            conf=0.35,
            imgsz=640,
            verbose=False
        )

        detected_objects = []

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                class_id = int(box.cls[0])

                confidence = float(box.conf[0])

                object_name = model.names[class_id]

                detected_objects.append(object_name)

                label = (
                    f"{object_name} "
                    f"{confidence * 100:.1f}%"
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

        return frame, detected_objects

    # ---------------- PROCESS IMAGE ----------------

    def process_image(self, image):

        frame = image.copy()

        # Face detection
        frame, face_count = self.detect_faces(frame)

        # Object detection
        frame, objects = self.detect_objects(frame)

        # Update counts
        self.face_label.config(
            text=f"Faces: {face_count}"
        )

        self.object_label.config(
            text=f"Objects: {len(objects)}"
        )

        self.result_image = frame.copy()

        self.display_image(frame)

        return face_count, objects

    # ---------------- DISPLAY IMAGE ----------------

    def display_image(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(rgb)

        image.thumbnail((950, 500))

        self.photo = ImageTk.PhotoImage(image)

        self.image_label.config(
            image=self.photo,
            text="",
            width=950,
            height=500
        )

    # ---------------- UPLOAD IMAGE ----------------

    def upload_image(self):

        self.stop_camera()

        path = filedialog.askopenfilename(
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png")
            ]
        )

        if not path:
            return

        image = cv2.imread(path)

        if image is None:
            messagebox.showerror(
                "Error",
                "Unable to load image"
            )
            return

        self.uploaded_image = image

        self.process_image(image)

        self.status_label.config(
            text="Image processed successfully"
        )

    # ---------------- START CAMERA ----------------

    def start_camera(self):

        if self.camera_running:
            return

        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.camera.isOpened():

            self.camera.release()

            self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():

            self.camera.release()
            self.camera = None

            messagebox.showerror(
                "Camera Error",
                "Unable to access webcam!"
            )
            return

        self.camera_running = True

        self.uploaded_image = None

        self.status_label.config(
            text="Live AI Detection Running..."
        )

        self.update_camera()

    # ---------------- LIVE DETECTION ----------------

    def update_camera(self):

        if not self.camera_running:
            return

        if self.camera is None:
            return

        ret, frame = self.camera.read()

        if not ret:

            self.stop_camera()

            messagebox.showerror(
                "Error",
                "Camera frame unavailable"
            )

            return

        frame = cv2.flip(frame, 1)

        # Detect faces
        frame, face_count = self.detect_faces(frame)

        # Detect objects
        frame, objects = self.detect_objects(frame)

        # Update counts
        self.face_label.config(
            text=f"Faces: {face_count}"
        )

        self.object_label.config(
            text=f"Objects: {len(objects)}"
        )

        # Save latest processed frame
        self.result_image = frame.copy()

        # AUTO CAPTURE
        detected = face_count > 0 or len(objects) > 0

        current_time = time.time()

        if (
            detected
            and self.auto_var.get()
            and current_time - self.last_capture_time
            >= self.capture_interval
        ):

            self.auto_save(frame)

            self.last_capture_time = current_time

        self.display_image(frame)

        self.root.after(
            30,
            self.update_camera
        )

    # ---------------- AUTO SAVE ----------------

    def auto_save(self, frame):

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = f"capture_{timestamp}.jpg"

        filepath = os.path.join(
            "results",
            filename
        )

        cv2.imwrite(filepath, frame)

        self.total_captures += 1

        self.capture_label.config(
            text=f"Auto Captures: {self.total_captures}"
        )

        self.status_label.config(
            text=f"Automatically saved: {filename}"
        )

        print("Saved:", filepath)

    # ---------------- MANUAL SAVE ----------------

    def save_result(self):

        if self.result_image is None:

            messagebox.showwarning(
                "Warning",
                "No result image available"
            )

            return

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        filepath = filedialog.asksaveasfilename(
            initialdir="results",
            initialfile=f"result_{timestamp}.jpg",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG", "*.jpg"),
                ("PNG", "*.png")
            ]
        )

        if filepath:

            cv2.imwrite(
                filepath,
                self.result_image
            )

            messagebox.showinfo(
                "Saved",
                "Result image saved successfully!"
            )

    # ---------------- OPEN RESULTS ----------------

    def open_results(self):

        os.startfile(
            os.path.abspath("results")
        )

    # ---------------- STOP CAMERA ----------------

    def stop_camera(self):

        self.camera_running = False

        if self.camera is not None:

            self.camera.release()

            self.camera = None

        self.status_label.config(
            text="Camera stopped"
        )

    # ---------------- CLOSE APP ----------------

    def close_app(self):

        self.stop_camera()

        self.root.destroy()


# ---------------- MAIN PROGRAM ----------------

if __name__ == "__main__":

    root = tk.Tk()

    app = FaceDetectionApp(root)

    root.mainloop()


""""
import cv2
import os
import tkinter as tk

from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


os.makedirs("output_images", exist_ok=True)

# Load Haar Cascade model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Unable to load face detection model!")


class FaceDetectionApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Face Detection Using Machine Learning")
        self.root.geometry("1000x750")
        self.root.configure(bg="#f0f0f0")

        self.image = None
        self.result_image = None
        self.photo = None

        self.camera = None
        self.camera_running = False

        # Heading
        tk.Label(
            root,
            text="FACE DETECTION SYSTEM",
            font=("Arial", 25, "bold"),
            bg="#f0f0f0",
            fg="#333399"
        ).pack(pady=20)

        # Image display
        self.image_label = tk.Label(
            root,
            text="Upload an image or start your camera",
            font=("Arial", 15),
            bg="white",
            width=85,
            height=23
        )
        self.image_label.pack(pady=15)

        # Buttons
        button_frame = tk.Frame(root, bg="#f0f0f0")
        button_frame.pack(pady=15)

        tk.Button(
            button_frame,
            text="Upload Image",
            command=self.upload_image,
            bg="#333399",
            fg="white",
            font=("Arial", 11, "bold"),
            width=15
        ).grid(row=0, column=0, padx=7)

        tk.Button(
            button_frame,
            text="Detect Faces",
            command=self.detect_faces,
            bg="green",
            fg="white",
            font=("Arial", 11, "bold"),
            width=15
        ).grid(row=0, column=1, padx=7)

        tk.Button(
            button_frame,
            text="Start Camera",
            command=self.start_camera,
            bg="#008080",
            fg="white",
            font=("Arial", 11, "bold"),
            width=15
        ).grid(row=0, column=2, padx=7)

        tk.Button(
            button_frame,
            text="Stop Camera",
            command=self.stop_camera,
            bg="red",
            fg="white",
            font=("Arial", 11, "bold"),
            width=15
        ).grid(row=0, column=3, padx=7)

        tk.Button(
            button_frame,
            text="Save Result",
            command=self.save_result,
            bg="orange",
            fg="white",
            font=("Arial", 11, "bold"),
            width=15
        ).grid(row=0, column=4, padx=7)

        # Face count
        self.result_label = tk.Label(
            root,
            text="Detected Faces: 0",
            font=("Arial", 18, "bold"),
            bg="#f0f0f0",
            fg="green"
        )
        self.result_label.pack(pady=15)

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    # Display image
    def display_image(self, image):

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(rgb)

        pil_image.thumbnail((850, 480))

        self.photo = ImageTk.PhotoImage(pil_image)

        self.image_label.config(
            image=self.photo,
            text="",
            width=850,
            height=480
        )

    # Upload image
    def upload_image(self):

        self.stop_camera()

        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png")
            ]
        )

        if not file_path:
            return

        self.image = cv2.imread(file_path)

        if self.image is None:
            messagebox.showerror(
                "Error", "Unable to open image"
            )
            return

        self.result_image = None

        self.result_label.config(
            text="Detected Faces: 0"
        )

        self.display_image(self.image)

    # Detect faces in uploaded image
    def detect_faces(self):

        if self.image is None:
            messagebox.showwarning(
                "Warning",
                "Please upload an image first"
            )
            return

        gray = cv2.cvtColor(
            self.image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(25, 25)
        )

        output = self.image.copy()

        for (x, y, w, h) in faces:

            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

            cv2.putText(
                output,
                "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        self.result_image = output

        self.result_label.config(
            text=f"Detected Faces: {len(faces)}"
        )

        self.display_image(output)

        if len(faces) == 0:
            messagebox.showinfo(
                "Result",
                "No faces detected. Try a clearer image."
            )

    # Start webcam
    def start_camera(self):

        if self.camera_running:
            return

        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Fallback camera initialization
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():

            self.camera.release()
            self.camera = None

            messagebox.showerror(
                "Camera Error",
                "Unable to access webcam. Check camera permissions."
            )
            return

        self.camera_running = True
        self.image = None

        self.update_camera()

    # Live camera face detection
    def update_camera(self):

        if not self.camera_running:
            return

        if self.camera is None:
            return

        ret, frame = self.camera.read()

        if not ret:
            self.stop_camera()
            messagebox.showerror(
                "Error",
                "Unable to read webcam frame."
            )
            return

        # Mirror camera view
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(40, 40)
        )

        # Draw boxes around faces
        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                "FACE DETECTED",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Update face count
        self.result_label.config(
            text=f"Live Detected Faces: {len(faces)}"
        )

        self.result_image = frame.copy()

        # Show live video
        self.display_image(frame)

        # Repeat after 15 milliseconds
        self.root.after(15, self.update_camera)

    # Stop webcam
    def stop_camera(self):

        self.camera_running = False

        if self.camera is not None:
            self.camera.release()
            self.camera = None

    # Save output
    def save_result(self):

        if self.result_image is None:
            messagebox.showwarning(
                "Warning",
                "No detected image available to save."
            )
            return

        save_path = filedialog.asksaveasfilename(
            initialdir="output_images",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG Image", "*.jpg"),
                ("PNG Image", "*.png")
            ]
        )

        if save_path:

            cv2.imwrite(
                save_path,
                self.result_image
            )

            messagebox.showinfo(
                "Success",
                "Image saved successfully!"
            )

    # Close application safely
    def close_app(self):

        self.stop_camera()
        self.root.destroy()


# Main application
if __name__ == "__main__":

    root = tk.Tk()

    app = FaceDetectionApp(root)

    root.mainloop()
"""