from IsingGridVaryingField import IsingGridVaryingField
import skimage.io  # scikit-image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio.v2 as imageio
import os
import shutil


def save_frame(avg: np.array, digits, i: int = 1):
    avg[avg >= 0] = 1
    avg[avg < 0] = -1
    avg = avg.astype(np.int32)
    plt.imshow(avg, cmap=cm.gray, aspect="equal", interpolation="none", vmin=-1, vmax=1)
    plt.axis('off')  # This line removes the axes
    plt.gca().set_position([0, 0, 1, 1])  # This line makes the figure fill the whole plot
    os.mkdir("frames") if not os.path.exists("frames") else None
    plt.savefig(f"frames/frame_{str(i).zfill(digits)}.png", bbox_inches='tight', pad_inches=0,
                transparent=True)
    plt.clf()  # This line clears the current figure


def create_gif(output_file_name: str, fps: int = 10):
    images = []
    for file_name in sorted(os.listdir("frames")):
        if file_name.endswith('.png'):
            file_path = os.path.join("frames", file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(f"gifs/{output_file_name}.gif", images, fps=fps)
    # Delete the directory with the frames
    shutil.rmtree("frames")


def calculate_accuracy(original: np.array, denoised: np.array) -> float:
    denoised[denoised >= 0] = 1
    denoised[denoised < 0] = -1
    denoised = denoised.astype(np.int32)
    correct_pixels = np.sum(original == denoised)
    total_pixels = original.size
    accuracy = correct_pixels / total_pixels
    return accuracy


def isingdenoise(
        original: np.array,
        noisy: np.array,
        q: float,
        burnin: int = 50000,
        loops: int = 500000,
        invtemp: float = 2.0,
        use_default_neighbours=True,
        make_gif: bool = False,
        save_frames_iter: int = 100
) -> (np.array, list):
    h = 0.5 * np.log(q / (1 - q))
    gg = IsingGridVaryingField(
        noisy.shape[0], noisy.shape[1], h * noisy, invtemp, use_default_neighbours
    )
    gg.grid = np.array(noisy)

    # Burn-in
    for _ in range(burnin):
        gg.gibbs_move()

    digits = len(str(loops))
    # Sample
    avg = np.zeros_like(noisy).astype(np.float64)
    accuracies = []
    for i in range(loops):
        gg.gibbs_move()
        avg += gg.grid
        accuracy = calculate_accuracy(original, avg / (i + 1))
        accuracies.append(accuracy)
        if not make_gif:
            continue
        # use the commented line below if you want to make the beginning slower
        # if (i < loops / 500 and i % (loops / 5000) == 0) or (i > loops / 500 and i % (loops / save_frames_iter) == 0):
        if i % (loops / save_frames_iter) == 0:
            save_frame(avg / (i + 1), digits, i + 1)

    return avg / loops, accuracies


def plot_accuracy(accuracies1: list, accuracies2: list):
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies1, label='default neighbours', color='blue', linewidth=3)
    plt.plot(accuracies2, label='improved neighbours', color='red', linewidth=3)
    plt.title('Accuracy of Denoising Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def denoise(
    image_source: str,
    noise_strength: float = 0.9,
    extfield_strength: float = 0.9,
    burnin: int = 50000,
    loops: int = 500000,
    invtemp: float = 2.0,
    use_default_neighbours: bool = True,
    fig_title: str = "",
    make_gif: bool = False,
    gif_title: str = "denoise",
    save_frames_iter: int = 100,
    fps: int = 10
):
    if isinstance(image_source, str):
        image = skimage.io.imread(image_source)
        image = (image[:, :, 0].astype(np.int32) * 2) - 1
    else:
        image = image_source
    noise = np.random.random(size=image.size).reshape(image.shape) > noise_strength
    noisy = np.array(image)
    noisy[noise] = -noisy[noise]
    avg, accuracies = isingdenoise(
        image, noisy, extfield_strength, burnin, loops, invtemp, use_default_neighbours, make_gif,
        save_frames_iter
    )
    avg[avg >= 0] = 1
    avg[avg < 0] = -1
    avg = avg.astype(np.int32)

    fig, axes = plt.subplots(ncols=2, figsize=(11, 6))
    axes[0].imshow(
        avg, cmap=cm.gray, aspect="equal", interpolation="none", vmin=-1, vmax=1
    )
    axes[0].set_title("Denoised")
    axes[1].imshow(
        noisy, cmap=cm.gray, aspect="equal", interpolation="none", vmin=-1, vmax=1
    )
    axes[1].set_title("Noisy with q = {:.2f}".format(noise_strength))
    for ax in axes:
        # remove the y-axis
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)

    fig.suptitle(fig_title)
    if make_gif:
        create_gif(gif_title, fps)
    return accuracies


def main():
    accuracies1 = denoise(
        "img/mini_logo.png",
        noise_strength=0.8,
        extfield_strength=0.9,
        burnin=50000,
        loops=500000,
        use_default_neighbours=True,
        make_gif=False
    )
    accuracies2 = denoise(
        "img/mini_logo.png",
        noise_strength=0.8,
        extfield_strength=0.9,
        burnin=50000,
        loops=500000,
        use_default_neighbours=False,
        make_gif=False
    )
    plot_accuracy(accuracies1, accuracies2)
    plt.show()


if __name__ == "__main__":
    main()
