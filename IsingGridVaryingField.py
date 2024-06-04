from IsingGrid import IsingGrid


class IsingGridVaryingField(IsingGrid):
    def __init__(self, height, width, extfield, invtemp):
        super().__init__(height, width, 0, invtemp)
        self.vextfield = extfield

    def local_energy(self, x, y):
        return self.vextfield[x, y] + sum(self.grid[xx, yy] for (xx, yy) in self.neighbours(x, y))
