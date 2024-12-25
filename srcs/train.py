import cv2
import torch
import torch.nn as nn

from forward_process import calculate_data_at_certain_time, calculate_parameters
from prepare_dataset import create_original_data
from simple_nn import SimpleNN


def train(
    data,
    batch_size,
    device,
    epochs,
    diffusion_steps,
    min_beta,
    max_beta,
    learning_rate,
    output_model_path,
):
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    model = SimpleNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    beta_ts, alpha_ts, bar_alpha_ts = calculate_parameters(
        diffusion_steps, min_beta, max_beta
    )
    for epoch in range(epochs):
        count = 0
        epoch_loss = 0
        for x in data_loader:
            random_time_step = torch.randint(
                0, diffusion_steps, size=[x.shape[0], x.shape[1], x.shape[2], 1]
            )
            noised_x_t, eps = calculate_data_at_certain_time(
                x, bar_alpha_ts, random_time_step
            )
            predicted_eps = model.forward(
                noised_x_t.to(device), random_time_step.to(device)
            )
            loss = loss_fn(predicted_eps, eps.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            count += 1

        print("Epoch {0}, Loss={1}".format(epoch, round(epoch_loss / count, 5)))

    print("Finished training!!")
    torch.save(model.state_dict(), output_model_path)
    print("Saved model: ", output_model_path)


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using {0} device".format(device))

    x = cv2.imread("../resources/brad_pitt.jpg")
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    data = torch.tensor(x / 255.0, dtype=torch.float32)
    data = data.unsqueeze(0)
    batch_size = 128
    epochs = 30
    diffusion_steps = 500
    min_beta = 1e-4
    max_beta = 0.02
    learning_rate = 1e-3
    output_model_path = "diffusion_model.pth"
    train(
        data,
        batch_size,
        device,
        epochs,
        diffusion_steps,
        min_beta,
        max_beta,
        learning_rate,
        output_model_path,
    )
