import numpy as np
import matplotlib.pyplot as plt


class IsingGrid:
    def __init__(self, height, width, extfield, invtemp):
        self.width, self.height, self.extfield, self.invtemp = (
            height,
            width,
            extfield,
            invtemp,
        )
        self.grid = np.zeros([self.width, self.height], dtype=np.int8) + 1

    def plot(self):
        plt.imshow(
            self.grid,
            cmap=cm.gray,
            aspect="equal",
            interpolation="none",
            vmin=-1,
            vmax=1,
        )

    def make_random(self):
        self.grid = (
            np.random.randint(0, 2, size=self.width * self.height).reshape(
                self.width, self.height
            )
            * 2
        ) - 1

    def neighbours(self, x, y):
        n = []
        if x == 0:
            n.append((self.width - 1, y))
        else:
            n.append((x - 1, y))
        if x == self.width - 1:
            n.append((0, y))
        else:
            n.append((x + 1, y))
        if y == 0:
            n.append((x, self.height - 1))
        else:
            n.append((x, y - 1))
        if y == self.height - 1:
            n.append((x, 0))
        else:
            n.append((x, y + 1))
        return n

    def local_energy(self, x, y):
        return self.extfield + sum(
            self.grid[xx, yy] for (xx, yy) in self.neighbours(x, y)
        )

    def total_energy(self):
        # Could maybe do some numpy games here, but periodic boundary conditions make this tricky.
        # This function is only ever useful for very small grids anyway.
        energy = -self.extfield * np.sum(self.grid)
        energy += (
            -sum(
                self.grid[x, y]
                * sum(self.grid[xx, yy] for (xx, yy) in self.neighbours(x, y))
                for x in range(self.width)
                for y in range(self.height)
            )
            / 2
        )
        return energy

    def probability(self):
        return np.exp(-self.invtemp * self.total_energy())

    def gibbs_move(self):
        n = np.random.randint(0, self.width * self.height)
        y = n // self.width
        x = n % self.width
        p = 1 / (
            1 + np.exp(-2 * self.invtemp * self.local_energy(x, y), dtype=np.float128)
        )
        if np.random.random() <= p:
            self.grid[x, y] = 1
        else:
            self.grid[x, y] = -1

    def from_number(self, n):
        """Convert an integer 0 <= n < 2**(width*height) into a grid."""
        binstring = bin(n)[2:]
        binstring = "0" * (N - len(binstring)) + binstring
        self.grid = np.array(
            [int(x) * 2 - 1 for x in binstring], dtype=np.int8
        ).reshape(self.width, self.height)

    def to_number(self):
        """Convert grid into an integer."""
        flat = [self.grid[x, y] for x in range(self.width) for y in range(self.height)]
        return sum(2**n * (int(x) + 1) // 2 for n, x in enumerate(reversed(flat)))
