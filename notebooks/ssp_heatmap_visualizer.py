from __future__ import annotations

import io
import sys
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ssp import ThreeAxisSSP, ThreeAxisSSPConfig


def fig_to_rgb_array(fig: plt.Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def build_similarity_views(dim: int, seed: int, x0: int, y0: int, z0: int):
    config = ThreeAxisSSPConfig(dim=int(dim), seed=int(seed))
    ssp = ThreeAxisSSP(config).to("cpu")

    target = torch.tensor([x0, y0, z0], dtype=torch.float32, device=ssp.device)
    target_ssp = ssp.encode(target)
    similarities = ssp.similarity_to_codebook(target_ssp).cpu().numpy()
    # codebook is defined over x,y,z in [0, 9], so this is 10x10x10=1000.
    full_volume = similarities.reshape(10, 10, 10)

    # Visualization uses x,y,z in [1, 9], matching the Sudoku-style convention.
    sim_volume = full_volume[1:, 1:, 1:]  # (9, 9, 9)
    sim_volume = np.clip(sim_volume, a_min=0.0, a_max=None)

    x_index = slice(None) if x0 == 0 else x0 - 1
    y_index = slice(None) if y0 == 0 else y0 - 1
    z_index = slice(None) if z0 == 0 else z0 - 1

    # --- 1) One heatmap per z-slice.
    slice_fig, slice_axes = plt.subplots(3, 3, figsize=(10, 10))
    slice_axes = slice_axes.ravel()
    global_min = 0.0
    global_max = float(sim_volume.max())
    for z_idx in range(9):
        axis = slice_axes[z_idx]
        xy_for_slice = sim_volume[:, :, z_idx].T  # display as y, x
        image = axis.imshow(
            xy_for_slice,
            origin="lower",
            cmap="viridis",
            vmin=global_min,
            vmax=global_max,
        )
        axis.set_title(f"z = {z_idx + 1}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_xticks(range(0, 9))
        axis.set_yticks(range(0, 9))
        axis.set_xticklabels(range(1, 10))
        axis.set_yticklabels(range(1, 10))
        if isinstance(x_index, int) and isinstance(y_index, int) and z_idx == z_index:
            axis.scatter(
                x_index,
                y_index,
                c="red",
                s=90,
                marker="*",
                edgecolors="white",
                linewidth=0.8,
            )
    slice_fig.colorbar(image, ax=slice_axes.tolist(), shrink=0.55)
    slice_fig.suptitle("S_z(x, y) for each z slice", fontsize=14)

    # Convert figures to image arrays for Gradio output.
    slice_image = fig_to_rgb_array(slice_fig)
    plt.close(slice_fig)

    target_slice = sim_volume[x_index, y_index, z_index]
    target_peak = float(np.max(target_slice))
    if isinstance(x_index, int):
        x_desc = str(x0)
    else:
        x_desc = "any"

    if isinstance(y_index, int):
        y_desc = str(y0)
    else:
        y_desc = "any"

    if isinstance(z_index, int):
        z_desc = str(z0)
    else:
        z_desc = "any"

    summary = (
        f"Target (x0, y0, z0): ({x_desc}, {y_desc}, {z_desc})\n"
        f"Peak at target: {target_peak:.4f}\n"
    )

    return slice_image, summary


def create_app() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("# SSP Heatmap Visualizer (3D -> 2D)")

        with gr.Row():
            with gr.Column(scale=1):
                dim_slider = gr.Slider(
                    minimum=32,
                    maximum=512,
                    step=32,
                    value=256,
                    label="SSP dim",
                )
                seed_slider = gr.Slider(
                    minimum=0,
                    maximum=2000,
                    step=1,
                    value=42,
                    label="Random seed",
                )
                x0_slider = gr.Slider(
                    minimum=0,
                    maximum=9,
                    step=1,
                    value=3,
                    label="Target x0 (0 = ignore)",
                )
                y0_slider = gr.Slider(
                    minimum=0,
                    maximum=9,
                    step=1,
                    value=7,
                    label="Target y0 (0 = ignore)",
                )
                z0_slider = gr.Slider(
                    minimum=0,
                    maximum=9,
                    step=1,
                    value=5,
                    label="Target z0 (0 = ignore)",
                )
                run_btn = gr.Button("Generate")
                summary_box = gr.Textbox(label="Summary", interactive=False, lines=3)
            with gr.Column(scale=2):
                slice_view = gr.Image(label="Per-z heatmaps (9 slices)", type="numpy")

        inputs = [dim_slider, seed_slider, x0_slider, y0_slider, z0_slider]
        outputs = [slice_view, summary_box]

        run_btn.click(fn=build_similarity_views, inputs=inputs, outputs=outputs)

        # Initial render so the app opens with plots visible.
        demo.load(
            fn=build_similarity_views,
            inputs=inputs,
            outputs=outputs,
        )
    return demo


# Gradio CLI (`uv run gradio run ...`) auto-discovers this symbol.
app = create_app()
# Keep `demo` as a familiar alias for `python file.py` execution.
demo = app


if __name__ == "__main__":
    app.launch()
