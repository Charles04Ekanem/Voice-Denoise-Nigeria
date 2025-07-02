import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_image(path: str):
    try:
        return mpimg.imread(path)
    except FileNotFoundError:
        print(f"[WARN] Missing: {path}")
        return None

def plot_spectrogram_grid(plot_dir: str,
                          epochs: list[int],
                          output_name: str = "composite_spectrogram_grid.png"):
    """
    Generate a grid of spectrogram plots:
    Rows: clean, noisy, denoised
    Cols: different epochs
    """
    rows = ["clean", "noisy", "denoised"]
    n_rows, n_cols = len(rows), len(epochs)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.4 * n_cols, 2.0 * n_rows),
                             squeeze=False)

    for r, row in enumerate(rows):
        for c, epoch in enumerate(epochs):
            path = os.path.join(plot_dir, f"{row}_spec_epoch{epoch}.png")
            img = load_image(path)
            ax = axes[r][c]

            if img is not None:
                ax.imshow(img, aspect='auto')
            else:
                ax.text(0.5, 0.5, 'Missing', ha='center', va='center', fontsize=8)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

            if r == 0:
                ax.set_title(f"Epoch {epoch}", fontsize=9)
            if c == 0:
                ax.set_ylabel(row.capitalize(), fontsize=9, rotation=0, labelpad=20)

    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    out_path = os.path.join(plot_dir, output_name)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Composite spectrogram grid saved --> {out_path}")

# Run it
if __name__ == "__main__":
    plot_dir = r"C:/Users/CHARLES EKANEM/Documents/ICMEAS/plots"
    epochs = [1, 3, 5, 7, 10]
    plot_spectrogram_grid(plot_dir, epochs)

