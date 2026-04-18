import matplotlib.pyplot as plt
import matplotlib.animation as animation

from collections.abc import Sequence

from src.SotAMiH.core.simulation import Simulation
from src.SotAMiH.core.mesh import Mesh1D

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
