import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from forward_process import calculate_parameters
from simple_nn import SimpleNN


def sampling(model_path, img_size, diffusion_steps, min_beta, max_beta):
    model = SimpleNN()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        x_init = torch.rand(size=(img_size[0], img_size[1], 3))
        beta_ts, alpha_ts, bar_alpha_ts = calculate_parameters(
            diffusion_steps, min_beta, max_beta
        )
        denoised_x = torch.zeros(
            (diffusion_steps, x_init.shape[0], x_init.shape[1], x_init.shape[2])
        )
        denoised_x[-1] = x_init
        for t in tqdm(range(diffusion_steps - 1, 0, -1)):
            if t > 1:
                z = torch.randn(x_init.shape)
            else:
                z = 0
            ts = torch.full((x_init.shape[0], x_init.shape[1], 1), t)
            mu = (
                1
                / torch.sqrt(alpha_ts[t])
                * (
                    (
                        denoised_x[t]
                        - (1 - alpha_ts[t])
                        / torch.sqrt(1 - bar_alpha_ts[t])
                        * model.forward(denoised_x[t], ts)
                    )
                )
            )
            denoised_x[t - 1] = mu + torch.sqrt(beta_ts[t]) * z

    return denoised_x


def create_sampling_animation(denoised_x, diffusion_steps, save_path):
    fig, ax = plt.subplots(figsize=(3, 3))
    X = (denoised_x[diffusion_steps - 1].numpy() * 255).clip(0, 255).astype("uint8")
    img_display = ax.imshow(X)
    ax.axis("off")

    def init():
        img_display.set_data(X)
        return (img_display,)

    def update(rev_t):
        # Update scatter plot
        t = diffusion_steps - 1 - rev_t
        noised_image = (denoised_x[t].numpy() * 255).clip(0, 255).astype("uint8")
        img_display.set_data(noised_image)
        ax.set_title(f"Sampling - Step {t}/{diffusion_steps}")
        return (img_display,)

    # Create animation
    anim = FuncAnimation(fig, update, frames=diffusion_steps, init_func=init, blit=True)
    # Save animation as video
    anim.save(save_path, writer="pillow", fps=10)
    plt.close(fig)
    print("Finish saving gif file: ", save_path)


if __name__ == "__main__":
    model_path = "diffusion_model.pth"
    img_size = [475, 474]
    diffusion_steps = 500
    min_beta = 1e-4
    max_beta = 0.02
    denoised_x = sampling(model_path, img_size, diffusion_steps, min_beta, max_beta)
    save_path = "../resources/sampling.gif"
    create_sampling_animation(denoised_x, diffusion_steps, save_path)
