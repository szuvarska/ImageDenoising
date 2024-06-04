from IsingGridVaryingField import IsingGridVaryingField
import skimage.io  # scikit-image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def isingdenoise(noisy, q, burnin=50000, loops=500000):
    h = 0.5 * np.log(q / (1 - q))
    gg = IsingGridVaryingField(noisy.shape[0], noisy.shape[1], h * noisy, 2)
    gg.grid = np.array(noisy)

    # Burn-in
    for _ in range(burnin):
        gg.gibbs_move()

    # Sample
    avg = np.zeros_like(noisy).astype(np.float64)
    for _ in range(loops):
        gg.gibbs_move()
        avg += gg.grid
    return avg / loops


def denoise(file_path: str, noise_strength: float = 0.9, extfield_strength: float = 0.9, burnin: int = 50000,
            loops: int = 500000):
    image = skimage.io.imread(file_path)
    image = (image[:, :, 0].astype(np.int32) * 2) - 1
    noise = np.random.random(size=image.size).reshape(image.shape) > noise_strength
    noisy = np.array(image)
    noisy[noise] = -noisy[noise]
    avg = isingdenoise(noisy, extfield_strength, burnin, loops)
    avg[avg >= 0] = 1
    avg[avg < 0] = -1
    avg = avg.astype(np.int32)

    fig, axes = plt.subplots(ncols=2, figsize=(11, 6))
    axes[0].imshow(avg, cmap=cm.gray, aspect="equal", interpolation="none", vmin=-1, vmax=1)
    axes[1].imshow(noisy, cmap=cm.gray, aspect="equal", interpolation="none", vmin=-1, vmax=1)


def main():
    denoise("img/mini_logo.png", noise_strength=0.9, extfield_strength=0.9, burnin=50000, loops=500000)
    plt.show()


if __name__ == "__main__":
    main()
