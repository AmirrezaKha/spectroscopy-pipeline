import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    A simple feedforward autoencoder for unsupervised anomaly detection or denoising.

    Architecture:
    - Encoder: Compresses input data to a lower-dimensional latent space.
    - Decoder: Reconstructs the input from the latent representation.
    """
    def __init__(self, input_dim: int):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features (spectral length).
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Reconstructed output tensor of same shape as input.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(spectra: np.ndarray, epochs: int = 100, lr: float = 1e-3,
                      device: Optional[str] = None) -> Autoencoder:
    """
    Trains an autoencoder on the given spectra.

    Parameters
    ----------
    spectra : np.ndarray
        2D array of FTIR spectra (samples x features).
    
    epochs : int, optional (default=100)
        Number of training epochs.

    lr : float, optional (default=1e-3)
        Learning rate for the Adam optimizer.

    device : str, optional
        Device to run the model on ('cpu' or 'cuda'). If None, auto-detected.

    Returns
    -------
    Autoencoder
        The trained PyTorch autoencoder model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.tensor(spectra, dtype=torch.float32).to(device)

    model = Autoencoder(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, X)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss.item():.6f}")

    return model
