import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

class NSTApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Neural Style Transfer App")

        # Create buttons
        self.select_content_button = tk.Button(master, text="Select Content Image", command=self.select_content_image)
        self.select_style_button = tk.Button(master, text="Select Style Image", command=self.select_style_image)
        self.stylize_button = tk.Button(master, text="Stylize Image", command=self.stylize_image)

        # Create labels to display selected images
        self.content_image_label = tk.Label(master, text="Content Image: None")
        self.style_image_label = tk.Label(master, text="Style Image: None")

        # Display buttons and labels
        self.select_content_button.pack()
        self.select_style_button.pack()
        self.content_image_label.pack()
        self.style_image_label.pack()
        self.stylize_button.pack()

    def select_content_image(self):
        content_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg")])
        self.content_image_label.config(text=f"Content Image: {content_path}")
        self.content_image_path = content_path

    def select_style_image(self):
        style_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg")])
        self.style_image_label.config(text=f"Style Image: {style_path}")
        self.style_image_path = style_path

    def stylize_image(self):
        if hasattr(self, 'content_image_path') and hasattr(self, 'style_image_path'):
            # Load content and style images
            content_image = load_image(self.content_image_path)
            style_image = load_image(self.style_image_path)

            # Stylize the image
            stylized_image = stylize_images(content_image, style_image)

            # Display the stylized image
            self.display_image(stylized_image)

    def display_image(self, image_tensor):
        # Convert the tensor to a PIL Image
        image_array = np.squeeze(image_tensor.numpy())
        pil_image = Image.fromarray((image_array * 255).astype(np.uint8))

        # Create a PhotoImage object to display in the label
        image_tk = ImageTk.PhotoImage(pil_image)

        # Update the stylized image label
        stylized_image_label = tk.Label(self.master, image=image_tk)
        stylized_image_label.image = image_tk
        stylized_image_label.photo = image_tk
        stylized_image_label.pack()

def load_image(img_path):
    try:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img[tf.newaxis, :]
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def stylize_images(content_image, style_image):
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    return stylized_image

if __name__ == "__main__":
    root = tk.Tk()
    app = NSTApp(root)
    root.mainloop()
