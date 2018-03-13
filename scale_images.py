import os
import concurrent.futures

from PIL import Image

input_dir = os.fsencode("/mnt/diskB/ChestXrayNIHCC/images/")
output_dir = os.fsencode("../images_scaled/")

thread_pool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

def resize_img(input_path, output_path):
    img = Image.open(input_path)
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = img.convert("L")
    img.save(output_path)

for file in os.listdir(input_dir):

    full_input_path = os.fsdecode(os.path.join(input_dir, file))
    full_output_path = os.fsdecode(os.path.join(output_dir, file))

    thread_pool_executor.submit(resize_img, full_input_path, full_output_path)

thread_pool_executor.shutdown(wait=True)

