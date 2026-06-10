from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_PRIMES = (1, 2_654_435_761, 805_459_861)


def _spatial_hash(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor, T: int) -> torch.Tensor:
    """Instant-NGP spatial hash. Inputs are int32 voxel-corner coordinates."""
    h = (ix * _PRIMES[0]) ^ (iy * _PRIMES[1]) ^ (iz * _PRIMES[2])
    return h.to(torch.long) % T


@dataclass
class HashConfig:
    n_levels: int = 8
    n_features: int = 2
    log2_table_size: int = 13            # T = 2**13 = 8192
    base_resolution: int = 8             # N_min
    finest_resolution: int = 128         # N_max
    bound: float = 1.0                   # domain is [-bound, bound]^3
    hidden: int = 16                     # tiny ReLU layer; 0 = linear decoder

    @property
    def T(self) -> int:
        return 1 << self.log2_table_size

    @property
    def resolutions(self) -> np.ndarray:
        """Geometric progression from N_min to N_max."""
        if self.n_levels == 1:
            return np.array([self.base_resolution], dtype=np.int32)
        b = (self.finest_resolution / self.base_resolution) ** (1.0 / (self.n_levels - 1))
        return np.floor(
            self.base_resolution * (b ** np.arange(self.n_levels))
        ).astype(np.int32)


class MultiResHashSDF(nn.Module):
    """point ∈ [-1,1]^3 → scalar SDF.

    Forward:
        For each level: hash 8 corner ids → look up features → trilinearly interpolate.
        Concat the L·F features and run a single linear layer to scalar SDF.
    """

    def __init__(self, cfg: HashConfig = HashConfig()):
        super().__init__()
        self.cfg = cfg
        T, F, L = cfg.T, cfg.n_features, cfg.n_levels

        # Hash table per level. Small init keeps training stable.
        self.features = nn.Parameter(torch.empty(L, T, F).uniform_(-1e-4, 1e-4))
        # Decoder: linear if cfg.hidden == 0, else one ReLU hidden layer.
        # A 16-wide hidden adds ~270 fp32 params (~1 KB) — negligible vs the hash table.
        in_dim = L * F
        if cfg.hidden > 0:
            self.decoder = nn.Sequential(
                nn.Linear(in_dim, cfg.hidden),
                nn.ReLU(),
                nn.Linear(cfg.hidden, 1),
            )
        else:
            self.decoder = nn.Linear(in_dim, 1)
            nn.init.zeros_(self.decoder.bias)
            nn.init.normal_(self.decoder.weight, std=1e-2)

        self.register_buffer("resolutions",
                             torch.from_numpy(cfg.resolutions).to(torch.int64))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B = x.shape[0]
        u = (x.clamp(-cfg.bound, cfg.bound) + cfg.bound) / (2.0 * cfg.bound)

        feats = []
        for L_i in range(cfg.n_levels):
            N = self.resolutions[L_i].item()
            scaled = u * (N - 1)
            base = torch.floor(scaled).to(torch.int64)
            frac = scaled - base.to(scaled.dtype)

            x0, y0, z0 = base[:, 0], base[:, 1], base[:, 2]
            x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

            T = cfg.T
            h000 = _spatial_hash(x0, y0, z0, T)
            h100 = _spatial_hash(x1, y0, z0, T)
            h010 = _spatial_hash(x0, y1, z0, T)
            h110 = _spatial_hash(x1, y1, z0, T)
            h001 = _spatial_hash(x0, y0, z1, T)
            h101 = _spatial_hash(x1, y0, z1, T)
            h011 = _spatial_hash(x0, y1, z1, T)
            h111 = _spatial_hash(x1, y1, z1, T)

            tbl = self.features[L_i]
            f000, f100 = tbl[h000], tbl[h100]
            f010, f110 = tbl[h010], tbl[h110]
            f001, f101 = tbl[h001], tbl[h101]
            f011, f111 = tbl[h011], tbl[h111]

            tx = frac[:, 0:1]; ty = frac[:, 1:2]; tz = frac[:, 2:3]

            c00 = f000 * (1 - tx) + f100 * tx
            c10 = f010 * (1 - tx) + f110 * tx
            c01 = f001 * (1 - tx) + f101 * tx
            c11 = f011 * (1 - tx) + f111 * tx
            c0  = c00 * (1 - ty) + c10 * ty
            c1  = c01 * (1 - ty) + c11 * ty
            f   = c0  * (1 - tz) + c1  * tz
            feats.append(f)

        h = torch.cat(feats, dim=-1)
        return self.decoder(h).squeeze(-1)

    @property
    def model_size_bytes(self) -> int:
        feat_b = self.features.numel() * 2
        dec_b = sum(p.numel() * 4 for p in self.decoder.parameters())
        return feat_b + dec_b


    def export_npz(self, path: str | Path) -> None:
        """
        Schema:
            features:    fp16  (L, T, F)
            resolutions: int32 (L,)
            cfg:         json-encoded HashConfig
            If cfg.hidden == 0 (linear decoder):
                decoder_W: fp32 (1, L*F)
                decoder_b: fp32 (1,)
            Else (one-hidden-layer ReLU MLP):
                W1: fp32 (hidden, L*F)
                b1: fp32 (hidden,)
                W2: fp32 (1, hidden)
                b2: fp32 (1,)
        """
        import json
        feats = self.features.detach().cpu().to(torch.float16).numpy()
        res = self.resolutions.detach().cpu().to(torch.int32).numpy()
        cfg_json = json.dumps(self.cfg.__dict__)
        out = dict(features=feats, resolutions=res, cfg=np.array(cfg_json))
        if isinstance(self.decoder, nn.Linear):
            out['decoder_W'] = self.decoder.weight.detach().cpu().to(torch.float32).numpy()
            out['decoder_b'] = self.decoder.bias.detach().cpu().to(torch.float32).numpy()
        else:
            l1, _, l2 = self.decoder  # Linear, ReLU, Linear
            out['W1'] = l1.weight.detach().cpu().to(torch.float32).numpy()
            out['b1'] = l1.bias.detach().cpu().to(torch.float32).numpy()
            out['W2'] = l2.weight.detach().cpu().to(torch.float32).numpy()
            out['b2'] = l2.bias.detach().cpu().to(torch.float32).numpy()
        np.savez_compressed(path, **out)

    @staticmethod
    def from_npz(path: str | Path) -> "MultiResHashSDF":
        import json
        z = np.load(path, allow_pickle=False)
        cfg = HashConfig(**json.loads(str(z["cfg"])))
        model = MultiResHashSDF(cfg)
        with torch.no_grad():
            model.features.copy_(torch.from_numpy(z["features"].astype(np.float32)))
            if cfg.hidden == 0:
                model.decoder.weight.copy_(torch.from_numpy(z["decoder_W"]))
                model.decoder.bias.copy_(torch.from_numpy(z["decoder_b"]))
            else:
                l1, _, l2 = model.decoder
                l1.weight.copy_(torch.from_numpy(z["W1"]))
                l1.bias.copy_(torch.from_numpy(z["b1"]))
                l2.weight.copy_(torch.from_numpy(z["W2"]))
                l2.bias.copy_(torch.from_numpy(z["b2"]))
        return model
