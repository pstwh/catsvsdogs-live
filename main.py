import sys
import argparse
import numpy as np
import onnxruntime as ort

from PIL import Image


if __name__ == "__main__":

    classes = ["Cat", "Dog"]

    parser = argparse.ArgumentParser(prog="Dogs vs Cats", description="Dogs vs Cats")
    parser.add_argument("image_path", type=str)

    session = ort.InferenceSession("catsvsdogs.onnx")

    args = parser.parse_args()
    try:
        image = Image.open(args.image_path).convert("RGB").resize((224, 224))
    except:
        print("Invalid image!")
        sys.exit(0)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = np.array(image)
    image = image / 255.0
    image = image - mean
    image = image / std
    image = np.moveaxis(image, -1, 0)
    image = np.expand_dims(image, 0)
    image = image.astype(np.float32)

    output = session.run(None, {"input": image})
    output = output[0]

    idx = np.argmax(output)

    print(classes[idx])
