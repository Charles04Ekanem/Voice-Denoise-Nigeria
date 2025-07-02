import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

CONFIG = {
    'seed': 42,
    'n_fft': 512,
    'hop_length': 128,
    'batch_size': 8,
    'lr': 1e-3,
    'epochs': 10,
    'train_split': 0.8,
    'sample_rate': 16000,
    'plot_dir': 'plots',
    'target_length': 16000,
    'hidden_dim': 256,
    'train_noisy': r"C:\Users\CHARLES EKANEM\Documents\ICMEAS\data\custom_noisy",
    'train_clean': r"C:\Users\CHARLES EKANEM\Documents\ICMEAS\data\clean_train",
    'model_dir': 'models_simple',
}

# --- Dataset ---
class VoiceDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, target_length=16000):
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)
        self.target_length = target_length
        self.noisy_files = sorted(list(self.noisy_dir.glob("*.wav")))
        self.clean_files = sorted(list(self.clean_dir.glob("*.wav")))
        assert len(self.noisy_files) == len(self.clean_files), "Mismatch in dataset size."

    def fix_length(self, waveform):
        if waveform.shape[1] < self.target_length:
            pad = self.target_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.target_length]
        return waveform

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy, _ = torchaudio.load(self.noisy_files[idx])
        clean, _ = torchaudio.load(self.clean_files[idx])
        return self.fix_length(noisy), self.fix_length(clean)

# --- Model ---
class DenoisingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cnn = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.bilstm = nn.LSTM(128, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(256, num_heads=4, batch_first=True)
        self.project = nn.Linear(256, hidden_dim)
        self.res_weight = nn.Parameter(torch.tensor(0.1))
        self.res_proj = nn.Linear(256, hidden_dim)
        self.bottleneck = nn.Linear(hidden_dim, 128)
        self.decoder = nn.LSTM(128, 128, batch_first=True)
        self.output = nn.Linear(128, input_dim)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.permute(0, 2, 1)
        x = F.relu(self.cnn(x))
        x = x.permute(0, 2, 1)  # (B, T, 128)

        x, _ = self.bilstm(x)
        attn_out, _ = self.attn(x, x, x)

        res = self.res_proj(x)
        attn_out = attn_out + self.res_weight * res

        x = F.relu(self.bottleneck(attn_out))
        x, _ = self.decoder(x)
        return self.output(x)

# --- Utility Functions ---
def compute_snr(clean, denoised):
    noise = clean - denoised
    signal_power = torch.mean(clean ** 2)
    noise_power = torch.mean(noise ** 2) + 1e-8
    return 10 * torch.log10(signal_power / noise_power)

def plot_waveform(waveform, title, path):
    plt.figure(figsize=(10, 3))
    plt.plot(waveform.squeeze().cpu().numpy())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_spectrogram(waveform, title, path, sample_rate=16000, n_fft=512, hop_length=128):
    plt.figure(figsize=(10, 4))
    spec = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    spec_db = torchaudio.functional.amplitude_to_DB(spec.abs(), multiplier=20, amin=1e-10, db_multiplier=0)
    plt.imshow(spec_db.squeeze().cpu().numpy(), origin='lower', aspect='auto', cmap='viridis')
    plt.title(title)
    plt.colorbar(label='dB')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_training_curves(train_losses, val_snrs, save_dir):
    epochs = range(1, len(train_losses)+1)
    
    plt.figure()
    plt.plot(epochs, train_losses, label="Train MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss over Epochs")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "train_mse_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, val_snrs, label="Validation SNR", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("SNR (dB)")
    plt.title("Validation SNR over Epochs")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "val_snr_curve.png"))
    plt.close()

# --- Main Training ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    os.makedirs(CONFIG['plot_dir'], exist_ok=True)

    dataset = VoiceDataset(CONFIG['train_noisy'], CONFIG['train_clean'], CONFIG['target_length'])
    train_size = int(CONFIG['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    input_dim = CONFIG['n_fft'] // 2 + 1  # 257
    model = DenoisingModel(input_dim=input_dim, hidden_dim=CONFIG['hidden_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss()
    window = torch.hann_window(CONFIG['n_fft']).to(device)
    
    train_losses = []
    val_snrs = []

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        for noisy, clean in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            noisy, clean = noisy.to(device), clean.to(device)
            noisy_spec = torch.stft(noisy.squeeze(1), CONFIG['n_fft'], CONFIG['hop_length'], window=window, return_complex=True)
            clean_spec = torch.stft(clean.squeeze(1), CONFIG['n_fft'], CONFIG['hop_length'], window=window, return_complex=True)

            noisy_mag = torch.abs(noisy_spec).permute(0, 2, 1)
            clean_mag = torch.abs(clean_spec).permute(0, 2, 1)

            output = model(noisy_mag)
            loss = criterion(output, clean_mag)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"\nTrain MSE: {avg_train_loss:.4f}")

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            noisy, clean = next(iter(val_loader))
            noisy, clean = noisy.to(device), clean.to(device)
            noisy_spec = torch.stft(noisy.squeeze(1), CONFIG['n_fft'], CONFIG['hop_length'], window=window, return_complex=True)
            clean_spec = torch.stft(clean.squeeze(1), CONFIG['n_fft'], CONFIG['hop_length'], window=window, return_complex=True)
            noisy_mag = torch.abs(noisy_spec).permute(0, 2, 1)

            pred_mag = model(noisy_mag)
            pred_spec = pred_mag.permute(0, 2, 1) * torch.exp(1j * torch.angle(noisy_spec))
            denoised = torch.istft(pred_spec, CONFIG['n_fft'], CONFIG['hop_length'], window=window)

            snr = compute_snr(clean.squeeze(1), denoised).item()
            val_snrs.append(snr)
            print(f"Validation SNR: {snr:.2f} dB")

            # --- Plotting ---
            plot_waveform(clean.squeeze(1), "Clean", f"{CONFIG['plot_dir']}/clean_waveform_epoch{epoch+1}.png")
            plot_waveform(denoised, "Denoised", f"{CONFIG['plot_dir']}/denoised_waveform_epoch{epoch+1}.png")
            plot_spectrogram(clean.squeeze(1), "Clean Spectrogram", f"{CONFIG['plot_dir']}/clean_spec_epoch{epoch+1}.png")
            plot_spectrogram(noisy.squeeze(1), "Noisy Spectrogram", f"{CONFIG['plot_dir']}/noisy_spec_epoch{epoch+1}.png")
            plot_spectrogram(denoised, "Denoised Spectrogram", f"{CONFIG['plot_dir']}/denoised_spec_epoch{epoch+1}.png")

        # Save model
        torch.save(model.state_dict(), os.path.join(CONFIG['model_dir'], f"epoch{epoch+1}_model.pth"))
    plot_training_curves(train_losses, val_snrs, CONFIG['plot_dir'])
    
if __name__ == "__main__":
    main()