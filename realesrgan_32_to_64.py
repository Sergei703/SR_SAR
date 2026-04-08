import os
import random
import time
from io import BytesIO
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import vgg19
from torch.nn.utils import spectral_norm

from tqdm import tqdm
from smb.SMBConnection import SMBConnection

# ================= CONFIG =================
class Config:
    # SMB соединение
    SMB_SERVER = ""
    SMB_SHARE = ""
    SMB_USER = ""
    SMB_PASSWORD = ""                        
    SMB_DOMAIN = ""

    # Пути внутри шары
    SMB_LR_PATH = "/SR_SAR/patches_32x32"
    SMB_HR_PATH = "/SR_SAR/patches_64x64"

    # Предобученные веса (лежат в той же папке, что и скрипт)
    PRETRAINED = "RealESRGAN_x2plus.pth"

    # Папка для сохранения (создастся автоматически)
    SAVE_DIR = "./ckpt_resrgan_32_to_64"
    
    # Параметры датасета
    DATASET_FRACTION = 1.0
    BATCH_SIZE = 4
    EPOCHS = 200
    
    # Оптимизаторы
    LR_G = 1e-4
    LR_D = 1e-4
    
    # Коэффициенты потерь
    LAMBDA_L1 = 1.0
    LAMBDA_PERCEPTUAL = 1.0
    LAMBDA_GAN = 0.1
    
    # Шаги обновления дискриминатора
    DISCRIMINATOR_UPDATES = 1
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PIN_MEMORY = True
    NUM_WORKERS = 0   # <--- ИСПРАВЛЕНО: 0 workers для избежания конфликтов SMB

    os.makedirs(SAVE_DIR, exist_ok=True)


# ================= SMB DATASET =================
class SMBSRDataset(Dataset):
    def __init__(self, config):
        self.cfg = config
        
        self.conn = SMBConnection(
            config.SMB_USER, config.SMB_PASSWORD,
            "", config.SMB_SERVER,
            use_ntlm_v2=True,
            is_direct_tcp=True
        )
        assert self.conn.connect(config.SMB_SERVER, 445), "Ошибка подключения к SMB"
        
        self.lr_files = []
        for file in self.conn.listPath(config.SMB_SHARE, config.SMB_LR_PATH):
            if file.filename not in ['.', '..'] and file.filename.endswith('.png'):
                self.lr_files.append(file.filename)
        
        self.hr_files = []
        for file in self.conn.listPath(config.SMB_SHARE, config.SMB_HR_PATH):
            if file.filename not in ['.', '..'] and file.filename.endswith('.png'):
                self.hr_files.append(file.filename)
        
        self.lr_files.sort()
        self.hr_files.sort()
        assert len(self.lr_files) == len(self.hr_files), "Количество LR и HR файлов не совпадает"
        
        total = len(self.lr_files)
        subset = int(total * config.DATASET_FRACTION)
        idx = sorted(random.sample(range(total), subset))
        self.lr_files = [self.lr_files[i] for i in idx]
        self.hr_files = [self.hr_files[i] for i in idx]
        
        self.tf = T.Compose([
            T.Grayscale(1),
            T.ToTensor(),
            T.Normalize(mean=(0.5,), std=(0.5,))
        ])
    
    def _read_image(self, share, path, filename, max_retries=3):
        """Читает изображение из SMB с повторными попытками и переподключением"""
        for attempt in range(max_retries):
            try:
                file_path = os.path.join(path, filename).replace('\\', '/')
                file_obj = BytesIO()
                self.conn.retrieveFile(share, file_path, file_obj)
                file_obj.seek(0)
                return Image.open(file_obj)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(0.5)
                # Переподключаемся при ошибке
                try:
                    self.conn.close()
                except:
                    pass
                self.conn = SMBConnection(
                    self.cfg.SMB_USER, self.cfg.SMB_PASSWORD,
                    "", self.cfg.SMB_SERVER,
                    use_ntlm_v2=True,
                    is_direct_tcp=True
                )
                self.conn.connect(self.cfg.SMB_SERVER, 445)
    
    def __len__(self):
        return len(self.lr_files)
    
    def __getitem__(self, idx):
        lr_img = self._read_image(self.cfg.SMB_SHARE, self.cfg.SMB_LR_PATH, self.lr_files[idx])
        hr_img = self._read_image(self.cfg.SMB_SHARE, self.cfg.SMB_HR_PATH, self.hr_files[idx])
        lr = self.tf(lr_img)
        hr = self.tf(hr_img)
        return lr, hr
    
    def __del__(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()


# ================= RRDB BLOCK =================
class RRDB(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, nf, 3, 1, 1),
        )
    def forward(self, x):
        return x + 0.2 * self.block(x)


# ================= GENERATOR (×2) =================
class Generator(nn.Module):
    def __init__(self, num_rrdb=16, nf=64):
        super().__init__()
        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf) for _ in range(num_rrdb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upsample = nn.Sequential(
            nn.Conv2d(nf, nf * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2)
        )
        self.conv_last = nn.Conv2d(nf, 1, 3, 1, 1)
    
    def forward(self, x):
        fea = self.conv_first(x)
        body = self.conv_body(self.body(fea))
        fea = fea + body
        fea = self.upsample(fea)
        out = self.conv_last(fea)
        return torch.tanh(out)


# ================= DISCRIMINATOR (U-Net со спектральной нормой) =================
class UNetDiscriminatorSN(nn.Module):
    def __init__(self, num_in_ch=1, num_feat=64, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)
    
    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)
        
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x4 = x4 + x2
        
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x5 = x5 + x1
        
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x6 = x6 + x0
        
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)
        return out


# ================= PERCEPTUAL LOSS (VGG19) =================
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.feature_layers = feature_layers
        self.slice = nn.ModuleList()
        self.output_layers = []
        
        prev_layer = 0
        for i, layer in enumerate(vgg):
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
            self.slice.append(layer)
            if i in [3, 8, 17, 26]:  # relu1_2, relu2_2, relu3_3, relu4_3
                self.output_layers.append(nn.Sequential(*self.slice[prev_layer:i+1]))
                prev_layer = i + 1
        
        self.output_layers = nn.ModuleList(self.output_layers)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    
    def normalize(self, x):
        x_rgb = x.repeat(1, 3, 1, 1)
        return (x_rgb - self.mean.to(x.device)) / self.std.to(x.device)
    
    def forward(self, x):
        features = []
        x = self.normalize(x)
        for layer in self.output_layers:
            x = layer(x)
            features.append(x)
        return features

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']):
        super().__init__()
        self.vgg = VGGFeatureExtractor(feature_layers).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()
    
    def forward(self, x, gt):
        x_feats = self.vgg(x)
        gt_feats = self.vgg(gt)
        loss = 0
        for x_f, gt_f in zip(x_feats, gt_feats):
            loss += self.criterion(x_f, gt_f)
        return loss / len(x_feats)


# ================= ЗАГРУЗКА ПРЕДОБУЧЕННЫХ ВЕСОВ =================
def load_realesrgan_weights(model, path):
    if not os.path.isfile(path):
        print(f"⚠️ Предобученные веса не найдены: {path}")
        return
    state = torch.load(path, map_location="cpu")
    if "params_ema" in state:
        state = state["params_ema"]
    
    adapted_state = {}
    for key, value in state.items():
        if "conv_first.weight" in key and value.shape[1] == 3:
            adapted_state[key] = value.mean(1, keepdim=True)
        elif "conv_last.weight" in key and value.shape[0] == 3:
            adapted_state[key] = value.mean(0, keepdim=True)
        elif "conv_last.bias" in key and value.shape[0] == 3:
            adapted_state[key] = value.mean(0, keepdim=True)
        else:
            if key in model.state_dict() and model.state_dict()[key].shape == value.shape:
                adapted_state[key] = value
    
    model.load_state_dict(adapted_state, strict=False)
    print("✅ Веса RealESRGAN загружены и адаптированы")


# ================= МЕТРИКИ =================
def psnr(sr, hr):
    mse = F.mse_loss(sr, hr)
    return 10 * torch.log10(1 / (mse + 1e-8))

def enl(img):
    return (img.mean() / (img.std() + 1e-8))**2


# ================= ОБУЧЕНИЕ =================
def train():
    cfg = Config()
    print(f"Устройство: {cfg.DEVICE}")
    print(f"Результаты будут сохранены в: {os.path.abspath(cfg.SAVE_DIR)}")
    
    dataset = SMBSRDataset(cfg)
    loader = DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY
    )
    print(f"Загружено {len(dataset)} пар LR/HR")
    
    G = Generator().to(cfg.DEVICE)
    D = UNetDiscriminatorSN(num_in_ch=1).to(cfg.DEVICE)
    
    load_realesrgan_weights(G, cfg.PRETRAINED)
    
    optim_G = torch.optim.Adam(G.parameters(), lr=cfg.LR_G, betas=(0.9, 0.99))
    optim_D = torch.optim.Adam(D.parameters(), lr=cfg.LR_D, betas=(0.9, 0.99))
    
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(cfg.DEVICE)
    bce_loss = nn.BCEWithLogitsLoss()
    
    G_ema = Generator().to(cfg.DEVICE)
    G_ema.load_state_dict(G.state_dict())
    
    best_psnr = 0.0
    print("Начинаем обучение...")
    
    for epoch in range(1, cfg.EPOCHS + 1):
        epoch_start = time.time()
        G.train()
        D.train()
        
        total_g_loss = 0.0
        total_d_loss = 0.0
        total_psnr = 0.0
        total_enl = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}")
        for lr, hr in pbar:
            lr, hr = lr.to(cfg.DEVICE), hr.to(cfg.DEVICE)
            batch_size = lr.size(0)
            
            # ---- Обучение дискриминатора ----
            for _ in range(cfg.DISCRIMINATOR_UPDATES):
                with torch.no_grad():
                    sr = G(lr)
                real_pred = D(hr)
                real_loss = bce_loss(real_pred, torch.ones_like(real_pred))
                fake_pred = D(sr.detach())
                fake_loss = bce_loss(fake_pred, torch.zeros_like(fake_pred))
                d_loss = (real_loss + fake_loss) / 2
                optim_D.zero_grad()
                d_loss.backward()
                optim_D.step()
            
            # ---- Обучение генератора ----
            sr = G(lr)
            l1 = l1_loss(sr, hr)
            percep = perceptual_loss(sr, hr)
            fake_pred = D(sr)
            gan_loss = bce_loss(fake_pred, torch.ones_like(fake_pred))
            g_loss = (cfg.LAMBDA_L1 * l1 +
                     cfg.LAMBDA_PERCEPTUAL * percep +
                     cfg.LAMBDA_GAN * gan_loss)
            optim_G.zero_grad()
            g_loss.backward()
            optim_G.step()
            
            # Обновление EMA
            with torch.no_grad():
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.data.mul_(0.999).add_(p.data, alpha=0.001)
            
            total_g_loss += g_loss.item() * batch_size
            total_d_loss += d_loss.item() * batch_size
            
            sr_01 = (sr + 1) / 2
            hr_01 = (hr + 1) / 2
            batch_psnr = psnr(sr_01, hr_01).item()
            total_psnr += batch_psnr * batch_size
            total_enl += enl(sr_01).item() * batch_size
            
            pbar.set_postfix({
                'G': f"{g_loss.item():.4f}",
                'D': f"{d_loss.item():.4f}",
                'PSNR': f"{batch_psnr:.2f}"
            })
        
        n = len(dataset)
        avg_g_loss = total_g_loss / n
        avg_d_loss = total_d_loss / n
        avg_psnr = total_psnr / n
        avg_enl = total_enl / n
        epoch_time = time.time() - epoch_start
        
        print(f"\n📊 Epoch {epoch} [{epoch_time:.1f}s]")
        print(f"  G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}")
        print(f"  PSNR: {avg_psnr:.3f} dB | ENL: {avg_enl:.3f}")
        
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(G_ema.state_dict(), os.path.join(cfg.SAVE_DIR, "best_generator.pth"))
            print(f"  ✨ Новый лучший PSNR: {best_psnr:.3f} dB (модель сохранена)")
        
        if epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': G.state_dict(),
                'generator_ema_state_dict': G_ema.state_dict(),
                'discriminator_state_dict': D.state_dict(),
                'optimizer_G_state_dict': optim_G.state_dict(),
                'optimizer_D_state_dict': optim_D.state_dict(),
                'psnr': avg_psnr,
                'enl': avg_enl,
            }
            torch.save(checkpoint, os.path.join(cfg.SAVE_DIR, f"checkpoint_epoch_{epoch}.pth"))
            print(f"  💾 Чекпоинт сохранён (эпоха {epoch})")
    
    print("🎉 Обучение завершено!")
    dataset.__del__()


if __name__ == "__main__":
    train()