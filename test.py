import os
# Tắt thông báo rác của TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tkinter as tk
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image, ImageDraw, ImageOps

# ================= CẤU HÌNH FINAL =================
MODEL_PATH = "best_efficientnet_kanji.h5"
LABEL_PATH = "kanji_labels_map.pkl"
CANVAS_SIZE = 300
MODEL_INPUT_SIZE = 64
BRUSH_SIZE = 18  # Nét bút đậm vừa phải

class KanjiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện Kanji - Final Model (4 Fonts)")
        self.root.geometry("500x750")
        
        self.model, self.labels = self.load_resources()
        self.setup_ui()
        
        # Mặc định: Nền Đen (0)
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        
    def load_resources(self):
        try:
            print(f"Loading model: {MODEL_PATH}...")
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(LABEL_PATH, "rb") as f:
                labels = pickle.load(f)
            print(">>> Sẵn sàng!")
            return model, labels
        except Exception as e:
            print(f"Lỗi load: {e}")
            return None, []

    def setup_ui(self):
        tk.Label(self.root, text="Vẽ Kanji (Nét trắng - Nền đen)", font=("Arial", 14)).pack(pady=10)
        
        self.canvas_frame = tk.Frame(self.root, bd=3, relief="groove")
        self.canvas_frame.pack()
        
        self.canvas = tk.Canvas(self.canvas_frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", cursor="cross")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Xóa bảng", command=self.clear_canvas, bg="#ff9999", width=15).pack(side="left")

        # Checkbox đảo màu (Thường không cần dùng với model này)
        self.invert_var = tk.IntVar(value=0) 
        tk.Checkbutton(btn_frame, text="Đảo màu (Nếu đoán sai)", variable=self.invert_var, command=self.predict).pack(side="left", padx=10)

        self.lbl_result = tk.Label(self.root, text="Hãy viết gì đó...", font=("Arial", 22, "bold"), fg="blue")
        self.lbl_result.pack(pady=5)
        
        self.suggestion_frame = tk.Frame(self.root)
        self.suggestion_frame.pack(pady=10)
        self.suggestion_buttons = []
        for i in range(5):
            btn = tk.Button(self.suggestion_frame, text="", font=("MS Gothic", 24), width=3,
                            command=lambda idx=i: self.copy_to_clipboard(idx))
            btn.pack(side="left", padx=5)
            self.suggestion_buttons.append(btn)

    def paint(self, event):
        # Tính toán tọa độ hình tròn đầu bút
        r = BRUSH_SIZE // 2
        x1 = event.x - r
        y1 = event.y - r
        x2 = event.x + r
        y2 = event.y + r
        
        # 1. Vẽ lên màn hình cho mắt người xem
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        
        # 2. Vẽ vào bộ nhớ cho AI xem (Dùng ellipse để đảm bảo nét vẽ đặc)
        self.draw.ellipse([x1, y1, x2, y2], fill=255)
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.lbl_result.config(text="...")
        for btn in self.suggestion_buttons:
            btn.config(text="")

    def on_release(self, event):
        self.root.after(10, self.predict)

    def process_image(self):
        bbox = self.image.getbbox()
        if not bbox: return None 
        
        # Lọc nhiễu chấm nhỏ
        if (bbox[2] - bbox[0] < 20) or (bbox[3] - bbox[1] < 20): return None

        cropped = self.image.crop(bbox)
        w, h = cropped.size
        max_side = max(w, h) + 40 
        new_img = Image.new("L", (max_side, max_side), color=0)
        new_img.paste(cropped, ((max_side - w) // 2, (max_side - h) // 2))
        
        try:
            final_img = new_img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.LANCZOS)
        except AttributeError:
            final_img = new_img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.ANTIALIAS)
            
        # Làm đậm nét chữ để giống font in đậm
        final_img = final_img.point(lambda p: 255 if p > 50 else 0)

        if self.invert_var.get() == 1:
            final_img = ImageOps.invert(final_img)
            
        return final_img

    def predict(self):
        if self.model is None: return
        processed_img = self.process_image()
        if processed_img is None: return

        # --- THÊM DÒNG NÀY ĐỂ SOI LỖI ---
        processed_img.save("debug_final.png")
        print("Đã lưu ảnh debug_final.png. Hãy mở lên xem!")
        # -------------------------------

        img_array = np.array(processed_img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        # EfficientNet cần 3 kênh màu
        if self.model.input_shape[-1] == 3:
            img_array = np.repeat(img_array, 3, axis=-1)

        prediction = self.model.predict(img_array, verbose=0)[0]
        top_indices = np.argsort(prediction)[-5:][::-1]
        
        top_char = self.labels[top_indices[0]]
        confidence = prediction[top_indices[0]] * 100
        
        self.lbl_result.config(text=f"Kết quả: {top_char}  ({confidence:.1f}%)")
        for i, idx in enumerate(top_indices):
            self.suggestion_buttons[i].config(text=self.labels[idx])

    def copy_to_clipboard(self, idx):
        char = self.suggestion_buttons[idx].cget("text")
        if char:
            self.root.clipboard_clear()
            self.root.clipboard_append(char)

if __name__ == "__main__":
    root = tk.Tk()
    try: root.option_add('*font', 'MSGothic 12')
    except: pass
    app = KanjiApp(root)
    root.mainloop()