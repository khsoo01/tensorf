from PIL import Image

image_paths = [f'./output/output{i}.png' for i in range(200)]
images = [Image.open(image_path) for image_path in image_paths]

images[0].save(
    './output/animated.gif',
    save_all=True,
    append_images=images[1:],
    duration=0.1,  # Duration between frames in milliseconds
    loop=0  # Loop forever
)
