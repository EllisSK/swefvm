import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from collections.abc import Sequence

from swefvm.core.simulation import Simulation
from swefvm.core.mesh import Mesh1D, Mesh2D

class Animation1D:
    def __init__(self, simulation: Simulation, max_elevation: int | float = 1, end_time: int | float | None = None, convergance_threshold: int | float | None = None, record_times: Sequence[int | float] | None = None) -> None:
        self.sim = simulation
        self.end = end_time
        self.conv = convergance_threshold
        self.rt = record_times
        self.maxy = max_elevation

        self.t: int | float = 0

        if not isinstance(simulation.mesh, Mesh1D):
            raise TypeError("Please provide a simulation with a 1D Mesh")

    def show(self):
        fig, ax = plt.subplots()

        x_vals = self.sim.mesh.x_vals
        ax.plot(x_vals, self.sim.mesh.zb[1:-1], label="Bed Surface", color="black")
        line_eta, = ax.plot(x_vals, self.sim.mesh.Q_array[1:-1, 0], label="Water Surface", color="blue")

        ax.set_ylim(0, self.maxy)
        ax.set_xlabel("Position Along Domain (m)")
        ax.set_ylabel("Elevation (m)")
        ax.legend()

        title = ax.set_title("Time: 0.000s")

        def update_frame(data):
            t, Q_array = data
            self.t = t

            line_eta.set_ydata(Q_array[1:-1, 0])
            title.set_text(f"Time: {t:.3f}s")

            return line_eta, title
        
        animate = animation.FuncAnimation(
            fig,
            update_frame,
            frames=self.sim.run_generator(self.end, self.conv, self.rt),
            repeat=False,
            cache_frame_data=False
        )

        plt.show(block=False)

        while plt.fignum_exists(fig.number):
            plt.pause(0.001)
            if hasattr(self, 't') and self.t == self.end:
                fig.canvas.draw_idle()
                fig.canvas.start_event_loop(0.001)
                break

        plt.close(fig)

class Animation2D:
    def __init__(self, simulation: Simulation, max_elevation: int | float = 1, end_time: int | float | None = None, convergance_threshold: int | float | None = None, record_times: Sequence[int | float] | None = None) -> None:
        self.sim = simulation
        self.end = end_time
        self.conv = convergance_threshold
        self.rt = record_times
        self.maxy = max_elevation

        self.t: int | float = 0

        if not isinstance(simulation.mesh, Mesh2D):
            raise TypeError("Please provide a simulation with a 2D Mesh")

    def show(self):
        fig = plt.figure(figsize=(14, 6))
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax2d = fig.add_subplot(1, 2, 2)

        mesh = self.sim.mesh
        X, Y = np.meshgrid(mesh.x_vals, mesh.y_vals, indexing="ij")

        zb = mesh.zb[1:-1, 1:-1]
        eta = mesh.Q_array[1:-1, 1:-1, 0]

        ax3d.plot_surface(X, Y, zb, color="black", alpha=1.0)
        surf_eta = [ax3d.plot_surface(X, Y, eta, color="blue", alpha=0.6)]

        ax3d.set_zlim(0, self.maxy)
        ax3d.set_xlabel("X Position (m)")
        ax3d.set_ylabel("Y Position (m)")
        ax3d.set_zlabel("Elevation (m)")
        ax3d.view_init(elev=30, azim=-45)

        heatmap = ax2d.imshow(
            (eta - zb).T,
            origin="lower",
            extent=(0, mesh.width, 0, mesh.height),
            cmap="Blues",
            vmin=0,
            vmax=self.maxy,
            aspect="auto",
        )
        ax2d.set_xlabel("X Position (m)")
        ax2d.set_ylabel("Y Position (m)")
        fig.colorbar(heatmap, ax=ax2d, label="Water Depth (m)")

        title = fig.suptitle("Time: 0.000s")

        def update_frame(data):
            t, Q_array = data
            self.t = t

            new_eta = Q_array[1:-1, 1:-1, 0]

            surf_eta[0].remove()
            surf_eta[0] = ax3d.plot_surface(X, Y, new_eta, color="blue", alpha=0.6)

            heatmap.set_array((new_eta - zb).T)

            title.set_text(f"Time: {t:.3f}s")

            return surf_eta[0], heatmap, title

        animate = animation.FuncAnimation(
            fig,
            update_frame,
            frames=self.sim.run_generator(self.end, self.conv, self.rt),
            repeat=False,
            cache_frame_data=False,
        )

        plt.show(block=False)

        while plt.fignum_exists(fig.number):
            plt.pause(0.001)
            if hasattr(self, "t") and self.t == self.end:
                fig.canvas.draw_idle()
                fig.canvas.start_event_loop(0.001)
                break

        plt.close(fig)