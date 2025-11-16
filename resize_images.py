from PIL import Image
from pathlib import Path

images = list(Path("cool_duck_images").rglob('*'))
for p in images:
    img = Image.open(p)
    new_width = img.size[0] // 3
    new_height = img.size[1] // 3
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    img.save(str(p).replace("cool_duck_images", "downscaled_duck_images"))
