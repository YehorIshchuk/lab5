"""Розпізнавання рукописних цифр збереженою CNN (MNIST).

Використання:
    python predict.py digits/0.png digits/1.png ...
"""
import sys
import numpy as np
from PIL import Image, ImageFilter
from tensorflow import keras

MODEL_PATH = "model/cnn_mnist_final.keras"


def prepare(path):
    img = Image.open(path).convert("RGBA")
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white, img).convert("L")
    img.thumbnail((300, 300))

    arr = np.array(img, dtype="float32")
    bg = np.array(img.filter(ImageFilter.GaussianBlur(max(img.size) // 6)), dtype="float32")
    ink = np.clip(bg - arr, 0, None)          # лінія темніша за фон
    if ink.max() > 0:
        ink = ink / ink.max()
    ink[ink < 0.25] = 0
    ink_img = Image.fromarray((ink * 255).astype("uint8"))

    bbox = ink_img.getbbox()
    if bbox:
        ink_img = ink_img.crop(bbox)
    k = max(3, 2 * (max(ink_img.size) // 25) + 1)
    ink_img = ink_img.filter(ImageFilter.MaxFilter(k))

    w, h = ink_img.size
    scale = 20 / max(w, h)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    ink_img = ink_img.resize(new_size, Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(ink_img, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))
    return (np.array(canvas, dtype="float32") / 255).reshape(1, 28, 28, 1)


def main(paths):
    model = keras.models.load_model(MODEL_PATH)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    for p in paths:
        probs = model.predict(prepare(p), verbose=0)[0]
        print(f"{p}: цифра {int(probs.argmax())}, впевненість {probs.max():.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
