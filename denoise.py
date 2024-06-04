from IsingGridVaryingField import IsingGridVaryingField
import skimage.io  # scikit-image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio.v2 as imageio
import os


def save_frame(avg: np.array, directory_name: str = 'image', i: int = 1):
    avg[avg >= 0] = 1
    avg[avg < 0] = -1
    avg = avg.astype(np.int32)
    plt.imshow(avg, cmap=cm.gray, aspect="equal", interpolation="none", vmin=-1, vmax=1)
    directory_path = f"frames/{directory_name}"
    os.mkdir(directory_path) if not os.path.exists(directory_path) else None
    plt.savefig(f"{directory_path}/frame_{i}.png")


def create_gif(output_file_name: str, input_directory_name: str = 'image', frame_duration: float = 0.1):
    images = []
    for file_name in sorted(os.listdir(f"frames/{input_directory_name}")):
        if file_name.endswith('.png'):
            file_path = os.path.join(f"frames/{input_directory_name}", file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(f"gifs/{output_file_name}.gif", images, format='GIF', duration=frame_duration)


def isingdenoise(noisy: np.array, q: float, burnin: int = 50000, loops: int = 500000, invtemp: float = 2.0,
                 title: str = "denoise", save_frames_iter: int = 100):
    h = 0.5 * np.log(q / (1 - q))
    gg = IsingGridVaryingField(noisy.shape[0], noisy.shape[1], h * noisy, invtemp)
    gg.grid = np.array(noisy)

    # Burn-in
    for _ in range(burnin):
        gg.gibbs_move()

    # Sample
    avg = np.zeros_like(noisy).astype(np.float64)
    for i in range(loops):
        gg.gibbs_move()
        avg += gg.grid
        if i % (loops / save_frames_iter) == 0:
            save_frame(avg / (i + 1), title, i + 1)

    return avg / loops


def denoise(file_path: str, noise_strength: float = 0.9, extfield_strength: float = 0.9, burnin: int = 50000,
            loops: int = 500000, invtemp: float = 2.0, title="denoise", save_frames_iter: int = 100):
    image = skimage.io.imread(file_path)
    image = (image[:, :, 0].astype(np.int32) * 2) - 1
    noise = np.random.random(size=image.size).reshape(image.shape) > noise_strength
    noisy = np.array(image)
    noisy[noise] = -noisy[noise]
    avg = isingdenoise(noisy, extfield_strength, burnin, loops, invtemp, title, save_frames_iter)
    avg[avg >= 0] = 1
    avg[avg < 0] = -1
    avg = avg.astype(np.int32)

    fig, axes = plt.subplots(ncols=2, figsize=(11, 6))
    axes[0].imshow(avg, cmap=cm.gray, aspect="equal", interpolation="none", vmin=-1, vmax=1)
    axes[1].imshow(noisy, cmap=cm.gray, aspect="equal", interpolation="none", vmin=-1, vmax=1)
    create_gif(title, title)


def main():
    # denoise("img/mini_logo.png", noise_strength=0.9, extfield_strength=0.9, burnin=50000, loops=500000, title="mini")
    # plt.show()
    create_gif("mini_slower", "mini", 5)


if __name__ == "__main__":
    main()
