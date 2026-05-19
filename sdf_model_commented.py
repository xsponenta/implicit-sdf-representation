"""Анотована копія sdf_model.py — пояснення кожного рядка українською.

Файл повністю функціональний — можна викликати `from sdf_model_commented import ...`
як і оригінал. Все, що додано — це коментарі.
"""

# ─── Імпорти ───────────────────────────────────────────────────────────────
from dataclasses import dataclass             # Декоратор для автогенерації boilerplate класів (init, repr, eq).
from pathlib import Path                       # Об'єктно-орієнтований шлях; замість сирих рядків.

import numpy as np                             # Числова бібліотека — масиви, математичні операції.
import torch                                   # PyTorch — основна бібліотека для deep learning.
import torch.nn as nn                          # Підмодуль з шарами нейромережі (Linear, Sequential, ReLU).

# ─── Константи для просторового хешування ───────────────────────────────────
# Три великі прості числа з оригінальної статті Instant-NGP. Друге — число Кнута
# для хешування (≈ 2^32 · золотий перетин). Підкреслення між тисячами — синтаксичний
# сахар Python, який ігнорується при парсингу але робить число читабельним.
_PRIMES = (1, 2_654_435_761, 805_459_861)


def _spatial_hash(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor, T: int) -> torch.Tensor:
    """Instant-NGP spatial hash. Inputs are int32 voxel-corner coordinates."""
    # XOR трьох добутків координат на прості числа — формула Instant-NGP.
    # `^` — побітовий XOR. PyTorch підтримує його як поелементну операцію над цілими тензорами.
    h = (ix * _PRIMES[0]) ^ (iy * _PRIMES[1]) ^ (iz * _PRIMES[2])
    # Конвертуємо в int64 (бо XOR з великими числами може переповнити int32),
    # потім remainder від ділення на T (розмір таблиці) → індекс у межах [0, T-1].
    return h.to(torch.long) % T


# ─── Конфігурація моделі ────────────────────────────────────────────────────
@dataclass                                     # Декоратор для автогенерації __init__, __repr__, __eq__.
class HashConfig:
    n_levels: int = 8                          # L — кількість рівнів роздільності хеш-сітки.
    n_features: int = 2                        # F — розмір feature-вектора в кожній комірці.
    log2_table_size: int = 13                  # log₂T = 13 → T = 8192 слотів у хеш-таблиці на рівень.
    base_resolution: int = 8                   # N_min — найгрубша роздільність (8³ комірок на рівні 0).
    finest_resolution: int = 128               # N_max — найтонша роздільність (128³ комірок на рівні L-1).
    bound: float = 1.0                         # Домен моделі: точки лежать у [-bound, bound]³.
    hidden: int = 16                           # Ширина прихованого шару декодера. 0 = чисто лінійний декодер.

    @property                                  # Робить .T властивістю, не методом — звертатись через cfg.T.
    def T(self) -> int:
        # Побітовий зсув: 1 << 13 = 2^13 = 8192. Швидший спосіб обчислення степеня двійки.
        return 1 << self.log2_table_size

    @property
    def resolutions(self) -> np.ndarray:
        """Геометрична прогресія резолюцій від N_min до N_max."""
        # Особливий випадок: якщо рівень один, формула геометричної прогресії не визначена.
        if self.n_levels == 1:
            return np.array([self.base_resolution], dtype=np.int32)
        # b — множник прогресії. Для N_min=8, N_max=128, L=8: b = (128/8)^(1/7) ≈ 1.486.
        b = (self.finest_resolution / self.base_resolution) ** (1.0 / (self.n_levels - 1))
        # np.arange(8) = [0,1,2,...,7]; b**arange = [b⁰, b¹, ..., b⁷]; * N_min дає [8, 11.89, 17.67, ...].
        # np.floor округлює вниз → цілі резолюції. .astype(np.int32) — компактне int.
        return np.floor(
            self.base_resolution * (b ** np.arange(self.n_levels))
        ).astype(np.int32)


# ─── Основна модель ─────────────────────────────────────────────────────────
class MultiResHashSDF(nn.Module):
    """Точка ∈ [-1,1]³ → скалярна SDF.

    Forward:
        Для кожного рівня: хеш 8 кутів вокселя → шукаємо feature-вектори
        → трилінійна інтерполяція → конкатенуємо L·F фіч → декодер MLP → SDF.
    """

    def __init__(self, cfg: HashConfig = HashConfig()):
        # Викликаємо конструктор nn.Module — це обов'язково, інакше PyTorch не зможе
        # відстежувати параметри моделі.
        super().__init__()
        self.cfg = cfg                                 # Зберігаємо конфіг для подальшого використання.
        T, F, L = cfg.T, cfg.n_features, cfg.n_levels  # Розпакування для скорочення подальших рядків.

        # Створюємо хеш-таблиці. Форма (L, T, F) = (8, 8192, 2) → загалом 131 072 чисел.
        # uniform_(-1e-4, 1e-4) — заповнення дрібними випадковими числами. Підкреслення в кінці
        # методу означає "in-place" модифікація. Маленька ініціалізація = модель починає з
        # майже-нульових передбачень, що стабілізує тренування.
        # nn.Parameter — спецтип, що автоматично включається до model.parameters() (отримує градієнти).
        self.features = nn.Parameter(torch.empty(L, T, F).uniform_(-1e-4, 1e-4))

        # Декодер: лінійний якщо cfg.hidden==0, інакше один прихований шар з ReLU.
        # Маленький MLP додає ~270 fp32 параметрів (~1 КБ) — мізерно проти хеш-таблиці.
        in_dim = L * F                                  # Вхід декодера = довжина конкатенованого вектора фіч.
        if cfg.hidden > 0:
            # nn.Sequential — це список модулів, що застосовуються один за одним.
            self.decoder = nn.Sequential(
                nn.Linear(in_dim, cfg.hidden),          # 16 → 16 повнозв'язний шар.
                nn.ReLU(),                              # Нелінійність ReLU(x) = max(0, x).
                nn.Linear(cfg.hidden, 1),               # 16 → 1 фінальний вихідний шар.
            )
        else:
            # Чисто лінійний декодер — одне множення матриця-вектор.
            self.decoder = nn.Linear(in_dim, 1)
            nn.init.zeros_(self.decoder.bias)           # Зсув = 0 — обережна ініціалізація.
            nn.init.normal_(self.decoder.weight, std=1e-2)  # Ваги з N(0, 0.01²) — невеликі.

        # register_buffer — як nn.Parameter, але БЕЗ градієнтів. Buffer переноситься з моделлю
        # між пристроями (CPU/GPU) і зберігається у state_dict, але не оновлюється оптимізатором.
        # Резолюції — це константи моделі, не тренуються.
        self.register_buffer("resolutions",
                             torch.from_numpy(cfg.resolutions).to(torch.int64))

    @property                                          # cfg.n_params без дужок (як атрибут).
    def n_params(self) -> int:
        # Сума numel() для кожного параметра моделі. numel() = total number of elements.
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B = x.shape[0]                                  # Розмір батчу — кількість точок (форма x: (B, 3)).

        # КРОК 1: обрізаємо координати до [-bound, bound] і відображаємо в [0, 1].
        # clamp обмежує значення; (x + bound) зсуває; / (2 * bound) масштабує.
        u = (x.clamp(-cfg.bound, cfg.bound) + cfg.bound) / (2.0 * cfg.bound)

        feats = []                                      # Список для збору feature-векторів з кожного рівня.

        # КРОК 2: цикл по L=8 рівнях.
        for L_i in range(cfg.n_levels):
            # .item() конвертує одноелементний тензор у Python-int. Це нам треба, щоб
            # використати в подальших обчисленнях зі скалярами.
            N = self.resolutions[L_i].item()
            # Масштабуємо координати [0,1] до [0, N-1] — індекси сітки на цьому рівні.
            # (N-1), а не N, бо у нас N комірок але N+1 вершин: точка u=0 → індекс 0, u=1 → N-1.
            scaled = u * (N - 1)
            # Округлення вниз → індекс "нижнього-лівого-задньго" кута вокселя, що містить точку.
            base = torch.floor(scaled).to(torch.int64)
            # Дробова частина в [0, 1) — параметр інтерполяції в межах одиничного вокселя.
            frac = scaled - base.to(scaled.dtype)

            # Розділяємо base на 3 координати (нижні кути) і додаємо 1 для верхніх.
            x0, y0, z0 = base[:, 0], base[:, 1], base[:, 2]
            x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

            # КРОК 3: хешуємо 8 кутів вокселя. Іменування hXYZ: X∈{0,1} для x0/x1, аналогічно Y,Z.
            T = cfg.T
            h000 = _spatial_hash(x0, y0, z0, T)         # (x0, y0, z0) → індекс у хеш-таблиці.
            h100 = _spatial_hash(x1, y0, z0, T)         # (x1, y0, z0).
            h010 = _spatial_hash(x0, y1, z0, T)         # (x0, y1, z0).
            h110 = _spatial_hash(x1, y1, z0, T)         # (x1, y1, z0).
            h001 = _spatial_hash(x0, y0, z1, T)         # (x0, y0, z1).
            h101 = _spatial_hash(x1, y0, z1, T)         # (x1, y0, z1).
            h011 = _spatial_hash(x0, y1, z1, T)         # (x0, y1, z1).
            h111 = _spatial_hash(x1, y1, z1, T)         # (x1, y1, z1).

            # КРОК 4: дістаємо 8 feature-векторів з хеш-таблиці поточного рівня.
            tbl = self.features[L_i]                    # Таблиця рівня L_i, форма (T, F) = (8192, 2).
            # tbl[hXYZ] — fancy indexing: бере рядок tbl за кожним індексом у hXYZ.
            # Результат — тензор форми (B, F): для кожної точки батчу — її feature-вектор.
            f000, f100 = tbl[h000], tbl[h100]
            f010, f110 = tbl[h010], tbl[h110]
            f001, f101 = tbl[h001], tbl[h101]
            f011, f111 = tbl[h011], tbl[h111]

            # КРОК 5: трилінійна інтерполяція. frac[:, 0:1] зберігає вимір розміру 1
            # для broadcasting з тензорами форми (B, F) — інакше форми не співпадуть.
            tx = frac[:, 0:1]; ty = frac[:, 1:2]; tz = frac[:, 2:3]

            # Прохід 1 — інтерполяція по X. 4 ребра вокселя дають 4 проміжні значення.
            c00 = f000 * (1 - tx) + f100 * tx           # Нижнє ребро по y=0, z=0.
            c10 = f010 * (1 - tx) + f110 * tx           # Верхнє ребро по y=1, z=0.
            c01 = f001 * (1 - tx) + f101 * tx           # Нижнє ребро по y=0, z=1.
            c11 = f011 * (1 - tx) + f111 * tx           # Верхнє ребро по y=1, z=1.
            # Прохід 2 — інтерполяція по Y. Об'єднує 4 в 2.
            c0  = c00 * (1 - ty) + c10 * ty             # Грань z=0.
            c1  = c01 * (1 - ty) + c11 * ty             # Грань z=1.
            # Прохід 3 — інтерполяція по Z. Об'єднує 2 в 1.
            f   = c0  * (1 - tz) + c1  * tz             # Фінальний feature-вектор форми (B, F).
            feats.append(f)

        # КРОК 6: конкатенація по останній осі. 8 тензорів (B, F) → один тензор (B, L*F) = (B, 16).
        h = torch.cat(feats, dim=-1)
        # КРОК 7: декодер видає скаляр на точку, потім squeeze(-1) прибирає вимір розміру 1.
        return self.decoder(h).squeeze(-1)

    @property
    def model_size_bytes(self) -> int:
        """Передбачуваний розмір при експорті у .npz формат."""
        # features в fp16 → 2 байти на число.
        feat_b = self.features.numel() * 2
        # декодер в fp32 → 4 байти на число.
        dec_b = sum(p.numel() * 4 for p in self.decoder.parameters())
        return feat_b + dec_b


    def export_npz(self, path: str | Path) -> None:
        """
        Зберігаємо модель у .npz файл, який потім зчитує numba-інференс.

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
        import json                                    # Локальний імпорт — потрібен лише тут.
        # .detach() — від'єднує від обчислювального графа (нам не потрібен gradient).
        # .cpu() — переносить з GPU/MPS на CPU (можна було б тренувати на іншому пристрої).
        # .to(torch.float16) — конвертуємо до 16-бітних чисел (економимо 2× пам'яті).
        # .numpy() — переводимо в numpy-масив для зберігання.
        feats = self.features.detach().cpu().to(torch.float16).numpy()
        res = self.resolutions.detach().cpu().to(torch.int32).numpy()
        # __dict__ повертає словник полів dataclass.
        cfg_json = json.dumps(self.cfg.__dict__)
        # np.array(cfg_json) обгортає рядок як 0-вимірний numpy-масив для збереження.
        out = dict(features=feats, resolutions=res, cfg=np.array(cfg_json))
        # Дві гілки залежно від типу декодера.
        if isinstance(self.decoder, nn.Linear):
            # Лінійний — одна матриця W і вектор b (fp32 для точності).
            out['decoder_W'] = self.decoder.weight.detach().cpu().to(torch.float32).numpy()
            out['decoder_b'] = self.decoder.bias.detach().cpu().to(torch.float32).numpy()
        else:
            # MLP — розпаковуємо Sequential на 3 модулі (Linear, ReLU, Linear).
            # _ — ReLU, не потребує збереження (немає параметрів).
            l1, _, l2 = self.decoder
            out['W1'] = l1.weight.detach().cpu().to(torch.float32).numpy()
            out['b1'] = l1.bias.detach().cpu().to(torch.float32).numpy()
            out['W2'] = l2.weight.detach().cpu().to(torch.float32).numpy()
            out['b2'] = l2.bias.detach().cpu().to(torch.float32).numpy()
        # savez_compressed зберігає словник у стиснутий .npz файл. **out розпаковує як kwargs.
        np.savez_compressed(path, **out)

    @staticmethod                                      # Викликається через MultiResHashSDF.from_npz(...).
    def from_npz(path: str | Path) -> "MultiResHashSDF":
        """Завантажуємо модель з .npz файлу."""
        import json
        # allow_pickle=False — безпеково; ми не зберігали жодних Python-об'єктів.
        z = np.load(path, allow_pickle=False)
        # Парсимо JSON-рядок з конфігом, потім конструюємо HashConfig зі словника.
        cfg = HashConfig(**json.loads(str(z["cfg"])))
        # Створюємо порожню модель з тією ж конфігурацією.
        model = MultiResHashSDF(cfg)
        # Контекст без обчислення градієнтів — економимо пам'ять і час при копіюванні ваг.
        with torch.no_grad():
            # .astype(np.float32) — конвертуємо назад до fp32 для тренування.
            # .copy_(...) — in-place копія в існуючий тензор (відмічається підкресленням).
            model.features.copy_(torch.from_numpy(z["features"].astype(np.float32)))
            # Знову дві гілки залежно від cfg.hidden.
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
