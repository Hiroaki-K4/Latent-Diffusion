import cv2
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation


def calculate_parameters(diffusion_steps, min_beta, max_beta):
    step = (max_beta - min_beta) / diffusion_steps
    beta_ts = torch.arange(min_beta, max_beta + step, step)

    alpha_ts = 1 - beta_ts
    bar_alpha_ts = torch.cumprod(alpha_ts, dim=0)

    return beta_ts, alpha_ts, bar_alpha_ts


def calculate_data_at_certain_time(x_0, bar_alpha_ts, t):
    eps = torch.randn(size=x_0.shape)
    noised_x_t = (
        torch.sqrt(bar_alpha_ts[t]) * x_0 + torch.sqrt(1 - bar_alpha_ts[t]) * eps
    )

    return noised_x_t, eps


def create_forward_process_animation(x, diffusion_steps, min_beta, max_beta, save_path):
    X = torch.tensor(x / 255.0, dtype=torch.float32)
    beta_ts, alpha_ts, bar_alpha_ts = calculate_parameters(
        diffusion_steps, min_beta, max_beta
    )
    fig, ax = plt.subplots(figsize=(3, 3))
    img_display = ax.imshow(X)
    ax.axis("off")

    def init():
        img_display.set_data(X)
        return (img_display,)

    def update(t):
        noised_x_t, eps = calculate_data_at_certain_time(X, bar_alpha_ts, t)
        noised_image = (noised_x_t.numpy() * 255).clip(0, 255).astype("uint8")
        img_display.set_data(noised_image)
        ax.set_title(f"Forward Process - Step {t}/{diffusion_steps}")
        return (img_display,)

    # Create animation
    anim = FuncAnimation(fig, update, frames=diffusion_steps, init_func=init, blit=True)

    # Save animation as video
    anim.save(save_path, writer="pillow", fps=30)
    plt.close(fig)
    print("Finish saving gif file: ", save_path)


if __name__ == "__main__":
    x = cv2.imread("../resources/brad_pitt.jpg")
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    diffusion_steps = 500
    min_beta = 1e-4
    max_beta = 0.02
    save_path = "../resources/forward_process.gif"
    create_forward_process_animation(x, diffusion_steps, min_beta, max_beta, save_path)
