"""
Generates representative sample skin lesion images for all 7 HAM10000 classes.
Saves PNG files into static/samples/ directory for instant UI demo testing.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SAMPLES_DIR = os.path.join(".", "static", "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Visual characteristics for generating realistic dermatoscopic lesion samples
LESION_PROPERTIES = {
    "akiec": {
        "bg_color": (230, 200, 185),
        "lesion_color": (210, 140, 120),
        "border_color": (195, 120, 100),
        "texture_roughness": 35,
        "asymmetry": 0.35,
        "filename": "akiec_sample.jpg"
    },
    "bcc": {
        "bg_color": (240, 210, 195),
        "lesion_color": (215, 150, 145),
        "border_color": (245, 230, 225),  # Pearly translucent border
        "texture_roughness": 15,
        "asymmetry": 0.25,
        "filename": "bcc_sample.jpg"
    },
    "bkl": {
        "bg_color": (235, 205, 190),
        "lesion_color": (140, 100, 70),   # Tan/waxy brown
        "border_color": (110, 75, 50),
        "texture_roughness": 40,
        "asymmetry": 0.20,
        "filename": "bkl_sample.jpg"
    },
    "df": {
        "bg_color": (238, 208, 192),
        "lesion_color": (170, 105, 90),   # Button-like brownish reddish
        "border_color": (140, 80, 70),
        "texture_roughness": 20,
        "asymmetry": 0.15,
        "filename": "df_sample.jpg"
    },
    "nv": {
        "bg_color": (242, 215, 198),
        "lesion_color": (75, 45, 30),     # Uniform dark brown mole
        "border_color": (60, 35, 25),
        "texture_roughness": 10,
        "asymmetry": 0.10,
        "filename": "nv_sample.jpg"
    },
    "vasc": {
        "bg_color": (235, 205, 190),
        "lesion_color": (180, 30, 50),    # Bright cherry red / purple vascular
        "border_color": (140, 20, 35),
        "texture_roughness": 15,
        "asymmetry": 0.15,
        "filename": "vasc_sample.jpg"
    },
    "mel": {
        "bg_color": (235, 205, 190),
        "lesion_color": (40, 25, 25),     # Highly irregular dark multi-colored lesion
        "border_color": (160, 60, 40),
        "texture_roughness": 50,
        "asymmetry": 0.65,
        "filename": "mel_sample.jpg"
    }
}

def generate_sample_lesions():
    size = (300, 300)
    for code, props in LESION_PROPERTIES.items():
        # Create base skin canvas with slight skin color variance
        canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        bg_r, bg_g, bg_b = props["bg_color"]
        
        # Add natural skin tone gradient
        for y in range(size[1]):
            for x in range(size[0]):
                noise = np.random.randint(-8, 8)
                canvas[y, x, 0] = np.clip(bg_r + noise, 0, 255)
                canvas[y, x, 1] = np.clip(bg_g + noise, 0, 255)
                canvas[y, x, 2] = np.clip(bg_b + noise, 0, 255)

        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)

        # Draw lesion shape with asymmetry
        cx, cy = 150, 150
        rx, ry = 65, 55
        asymmetry = props["asymmetry"]

        # Create irregular points around circle
        num_points = 24
        points = []
        for i in range(num_points):
            angle = (2 * np.pi * i) / num_points
            r_var = 1.0 + np.random.uniform(-asymmetry, asymmetry)
            px = cx + rx * r_var * np.cos(angle)
            py = cy + ry * r_var * np.sin(angle)
            points.append((px, py))

        # Draw main lesion body
        draw.polygon(points, fill=props["lesion_color"], outline=props["border_color"])

        # Add texture & pigment variations
        arr = np.array(img, dtype=np.int16)
        roughness = props["texture_roughness"]
        texture_noise = np.random.randint(-roughness, roughness, arr.shape, dtype=np.int16)
        
        # Only apply noise to lesion interior
        lesion_mask = (arr[:,:,0] != bg_r) | (arr[:,:,1] != bg_g)
        arr[lesion_mask] = np.clip(arr[lesion_mask] + texture_noise[lesion_mask], 0, 255)

        final_img = Image.fromarray(arr.astype(np.uint8))
        final_img = final_img.filter(ImageFilter.GaussianBlur(radius=1.2))

        filepath = os.path.join(SAMPLES_DIR, props["filename"])
        final_img.save(filepath, "JPEG", quality=92)
        print(f"Generated sample image for {code}: {filepath}")

if __name__ == "__main__":
    generate_sample_lesions()
