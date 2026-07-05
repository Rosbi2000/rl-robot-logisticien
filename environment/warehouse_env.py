import numpy as np
import matplotlib.pyplot as plt

class WarehouseEnv:
    def __init__(self, grid_size=(5, 5), obstacles=None, delivery_points=None):
        self.grid_size = grid_size
        self.obstacles = obstacles if obstacles else [(1, 1), (2, 3)]
        self.delivery_points = delivery_points if delivery_points else [(4, 4)]
        self.start_pos = (0, 0)
        self.state = self.start_pos
        self.done = False
        self.action_space = [0, 1, 2, 3]  # 0=haut, 1=bas, 2=gauche, 3=droite

    def reset(self):
        self.state = self.start_pos
        self.done = False
        return self.state

    def step(self, action):
        if self.done:
            raise Exception("Episode terminé, reset nécessaire.")

        x, y = self.state
        if action == 0:    # haut
            x = max(0, x - 1)
        elif action == 1:  # bas
            x = min(self.grid_size[0] - 1, x + 1)
        elif action == 2:  # gauche
            y = max(0, y - 1)
        elif action == 3:  # droite
            y = min(self.grid_size[1] - 1, y + 1)

        new_state = (x, y)
        reward = -1  # pénalité par défaut

        if new_state in self.obstacles:
            reward = -5
        elif new_state in self.delivery_points:
            reward = +10
            self.done = True

        self.state = new_state
        return new_state, reward, self.done

    def render(self):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-0.5, self.grid_size[1]-0.5)
        ax.set_ylim(-0.5, self.grid_size[0]-0.5)
        ax.set_xticks(range(self.grid_size[1]))
        ax.set_yticks(range(self.grid_size[0]))
        ax.grid(True)

        # Obstacles
        for (ox, oy) in self.obstacles:
            ax.add_patch(plt.Rectangle((oy-0.5, ox-0.5), 1, 1, color="black"))

        # Delivery points
        for (dx, dy) in self.delivery_points:
            ax.add_patch(plt.Rectangle((dy-0.5, dx-0.5), 1, 1, color="green"))

        # Robot
        x, y = self.state
        ax.add_patch(plt.Circle((y, x), 0.3, color="red"))

        ax.invert_yaxis()
        plt.axis("off")

        # Convertir en image numpy (méthode correcte)
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        return image
