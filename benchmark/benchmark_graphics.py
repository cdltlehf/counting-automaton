import json
import os
import matplotlib.pyplot as plt
import benchmark_utils as bench

"""
    For creating the different graphics of the experiments.
"""

linestyles = ['--', '-', '--', ':']
markers = ['o', 's', 'D', 'x']
markersizes = [4, 4, 4, 6]

"""
    Plot the static benchmark runtime graphics.
"""
def sbr_graphics(directory: str="SBR"):
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    files.sort()

    plots_per_page = 6
    rows, cols = 3, 2

    for start_idx in range(0, len(files), plots_per_page):  # Go through the different regexes to plot
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12), dpi=100)
        axes = axes.flatten()

        for i, file in enumerate(files[start_idx:start_idx + plots_per_page]):
            ax = axes[i]
            filepath = os.path.join(directory, file)

            with open(filepath, "r") as f:
                data = json.load(f)

            regex = data["Regex"]
            x = data["Range"]

            for id, method in enumerate(bench.methods):

                if method == bench.flatten:
                    continue

                if method != bench.flatten:
                    method_str = method.value
                else:
                    method_str = method

                y = data["Results"][method_str]
                y = list(filter(lambda x: x != 0.0, y))

                # Create subplots
                ax.plot(
                    x[:len(y)], y,
                    label=method_str,
                    color=bench.method_colour[method],
                    linewidth=2,
                    linestyle=linestyles[id % len(linestyles)],
                    marker=markers[id % len(markers)],
                    markersize=markersizes[id % len(markersizes)]
                )
            
            # Print the details of each plot
            ax.set_title(f"`{regex}`", fontsize=13, fontweight='bold')
            ax.set_xlabel("Length of Input String", fontsize=11)
            ax.set_ylabel("CPU Time (ms)", fontsize=11)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

            ax.tick_params(labelsize=9)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle("Regex Runtime Benchmarks", fontsize=18, fontweight='bold')
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10, title="Method", title_fontsize=11)

        # Save as multipage PDFs if necessary
        output_file = f"SBRGraphics.pdf"
        plt.savefig(output_file)
        plt.close(fig)

        print(f"Saved {output_file}")

def dbr_graphics(directory: str="DBR"):
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    files.sort()

    plots_per_page = 6
    rows, cols = 3, 2

    for start_idx in range(0, len(files), plots_per_page): # Go through the different regexes to plot
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12), dpi=100)
        axes = axes.flatten()

        for i, file in enumerate(files[start_idx:start_idx + plots_per_page]):
            ax = axes[i]
            filepath = os.path.join(directory, file)

            with open(filepath, "r") as f:
                data = json.load(f)

            regex = data["Regex"]
            x = data["Range"]

            for id, method in enumerate(bench.methods):
                if method == bench.flatten:
                    continue

                if method != bench.flatten:
                    method_str = method.value
                else:
                    method_str = method

                y = data["Results"][method_str]
                y = list(filter(lambda x: x != 0.0, y))

                ax.plot(
                    x[:len(y)], y,
                    label=method_str,
                    color=bench.method_colour[method],
                    linewidth=2,
                    linestyle=linestyles[id % len(linestyles)],
                    marker=markers[id % len(markers)],
                    markersize=markersizes[id % len(markersizes)]
                )
                
            ax.set_title(f"`{regex}`", fontsize=13, fontweight='bold')
            ax.set_xlabel("Length of Input String", fontsize=11)
            ax.set_ylabel("CPU Time (ms)", fontsize=11)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

            ax.tick_params(labelsize=9)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle("Regex Runtime Benchmarks", fontsize=18, fontweight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])

        output_file = f"DBRGraphics.pdf"
        plt.savefig(output_file)
        plt.close(fig)

        print(f"Saved {output_file}")

def sbd_graphics(directory: str="SBD"):
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    files.sort()

    plots_per_page = 6
    rows, cols = 3, 2

    for start_idx in range(0, len(files), plots_per_page): # Go through the different regexes to plot
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12), dpi=100)
        axes = axes.flatten()

        for i, file in enumerate(files[start_idx:start_idx + plots_per_page]):
            ax = axes[i]
            filepath = os.path.join(directory, file)

            with open(filepath, "r") as f:
                data = json.load(f)

            regex = data["Regex"]
            x = data["Range"]

            for id, method in enumerate(bench.methods):
                if method == bench.flatten: # Density experiments do not include flatten
                    continue
                method_str = method.value

                y = data["Results"][method_str]

                # Create subplots
                ax.plot(
                    x, y,
                    label=method_str,
                    color=bench.method_colour[method],
                    linewidth=2,
                    linestyle=linestyles[id % len(linestyles)],
                    marker=markers[id % len(markers)],
                    markersize=markersizes[id % len(markersizes)]
                )
            
            # Print the details of each plot
            ax.set_title(f"`{regex}`", fontsize=13, fontweight='bold')
            ax.set_xlabel("Length of Input String", fontsize=11)
            ax.set_ylabel("Maximum Counter Density", fontsize=11)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.legend(fontsize=9, title="Method", title_fontsize=10, frameon=True)
            ax.tick_params(labelsize=9)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle("Regex Counter Densities", fontsize=18, fontweight='bold')
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10, title="Method", title_fontsize=11)

        output_file = f"SBDGraphics.pdf"
        plt.savefig(output_file)
        plt.close(fig)

        print(f"Saved {output_file}")

def dbd_graphics(directory: str="DBD"):
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    files.sort()

    plots_per_page = 6
    rows, cols = 3, 2

    for start_idx in range(0, len(files), plots_per_page):
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12), dpi=100)
        axes = axes.flatten()

        for i, file in enumerate(files[start_idx:start_idx + plots_per_page]):
            ax = axes[i]
            filepath = os.path.join(directory, file)

            with open(filepath, "r") as f:
                data = json.load(f)

            regex = data["Regex"]
            x = data["Range"]

            for id, method in enumerate(bench.methods):

                if method == bench.flatten:
                    continue
                method_str = method.value

                y = data["Results"][method_str]

                # Create subplots
                ax.plot(
                    x, y,
                    label=method_str,
                    color=bench.method_colour[method],
                    linewidth=2,
                    linestyle=linestyles[id % len(linestyles)],
                    marker=markers[id % len(markers)],
                    markersize=markersizes[id % len(markersizes)]
                )
                
            ax.set_title(f"`{regex}`", fontsize=13, fontweight='bold')
            ax.set_xlabel("Length of Input String", fontsize=11)
            ax.set_ylabel("Maximum Counter Density", fontsize=11)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.legend(fontsize=9, title="Method", title_fontsize=10, frameon=True)
            ax.tick_params(labelsize=9)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle("Regex Counter Densities", fontsize=18, fontweight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])

        output_file = f"DBDGraphics.pdf"
        plt.savefig(output_file)
        plt.close(fig)

        print(f"Saved {output_file}")


if __name__ == "__main__":
    sbr_graphics()
    sbd_graphics()
