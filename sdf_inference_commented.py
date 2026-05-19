"""Анотована копія sdf_inference.py — пояснення кожного рядка українською.

Файл повністю функціональний — імпортується і використовується як оригінал.
"""

# ─── Імпорти ───────────────────────────────────────────────────────────────
import json                                    # Для парсингу cfg-поля з .npz.
from pathlib import Path                       # Об'єктно-орієнтовані шляхи (str | Path).
from typing import Optional                    # Анотація типу для опціональних параметрів.

import numpy as np                             # Числові масиви.
import numba                                   # JIT-компілятор Python → нативний код.
from numba import njit, prange                 # njit — JIT-декоратор; prange — паралельний range.

# ─── Константи хешування ────────────────────────────────────────────────────
# Ті самі прості числа, що в sdf_model.py. np.int64(...) — явна типізація:
# numba дуже строго типізує, тому для гарантії правильного типу при множенні
# в njit-функції обгортаємо їх у int64.
_P0 = np.int64(1)
_P1 = np.int64(2_654_435_761)
_P2 = np.int64(805_459_861)


# ─── Хеш-функція ────────────────────────────────────────────────────────────
# njit — Numba скомпілює цю функцію при першому виклику в нативний CPU-код.
# cache=True — компільований код кешується на диск (не перекомпілюється при наступному запуску).
# fastmath=True — дозволяє компілятору переставляти операції для швидкості (за рахунок
#                 мікроскопічних відхилень в молодших цифрах — неважливо для нас).
# inline='always' — підказка завжди підставляти тіло цієї функції замість виклику.
@njit(cache=True, fastmath=True, inline='always')
def _hash3(ix: int, iy: int, iz: int, T: int) -> int:
    # Той самий XOR трьох добутків, що в torch-моделі.
    h = (ix * _P0) ^ (iy * _P1) ^ (iz * _P2)
    # h & (T - 1) — побітове AND. Працює як modulo тільки якщо T — степінь двійки.
    # T = 8192 = 2^13, тому T - 1 = 8191 (13 одиничок у двійковому). AND з цим маскує
    # лише 13 найменших значущих бітів. Це ~5× швидше за modulo на сучасних CPU.
    return h & (T - 1)


# ─── Forward pass для однієї точки ──────────────────────────────────────────
# Без parallel=True, бо це функція-одинарник; паралелізм — на рівні batch.
@njit(cache=True, fastmath=True)
def _forward_one(x: float, y: float, z: float,
                 features: np.ndarray,        # (L, T, F) float32 — таблиці фіч.
                 res: np.ndarray,             # (L,) int64 — резолюції рівнів.
                 W1: np.ndarray, b1: np.ndarray,   # Ваги прихованого шару декодера.
                 W2: np.ndarray, b2: np.ndarray,   # Ваги вихідного шару.
                 has_hidden: bool) -> float:
    # Розпакування розмірів. Numba знає типи з контексту.
    L = features.shape[0]
    T = features.shape[1]
    F = features.shape[2]
    in_dim = L * F                                 # Вхідний вимір декодера = 16.

    # КРОК 1: clamp + map до [0, 1]. Скалярні if-else, бо numba не має .clamp() для скалярів.
    if x < -1.0: x = -1.0
    elif x > 1.0: x = 1.0
    if y < -1.0: y = -1.0
    elif y > 1.0: y = 1.0
    if z < -1.0: z = -1.0
    elif z > 1.0: z = 1.0
    # Множення на 0.5 концептуально швидше за / 2.0 (некоторі архітектури).
    ux = (x + 1.0) * 0.5
    uy = (y + 1.0) * 0.5
    uz = (z + 1.0) * 0.5

    # Локальний буфер для конкатенованих фіч. numba алокує на стеку (якщо розмір відомий) — дуже швидко.
    h_buf = np.empty(in_dim, dtype=np.float32)

    # КРОК 2: цикл по L=8 рівнях.
    for Li in range(L):
        N = res[Li]                                # Резолюція поточного рівня.
        # Масштабуємо до координат сітки.
        sx = ux * (N - 1.0)
        sy = uy * (N - 1.0)
        sz = uz * (N - 1.0)
        # int64(sx) — конверсія в ціле з округленням вниз (для додатних чисел = floor).
        ix = np.int64(sx); iy = np.int64(sy); iz = np.int64(sz)
        # Дробова частина в [0, 1) — параметр інтерполяції.
        fx = sx - ix; fy = sy - iy; fz = sz - iz

        # КРОК 3: хешуємо 8 кутів вокселя.
        h000 = _hash3(ix,     iy,     iz,     T)
        h100 = _hash3(ix + 1, iy,     iz,     T)
        h010 = _hash3(ix,     iy + 1, iz,     T)
        h110 = _hash3(ix + 1, iy + 1, iz,     T)
        h001 = _hash3(ix,     iy,     iz + 1, T)
        h101 = _hash3(ix + 1, iy,     iz + 1, T)
        h011 = _hash3(ix,     iy + 1, iz + 1, T)
        h111 = _hash3(ix + 1, iy + 1, iz + 1, T)

        # КРОК 4-5: дістаємо фічі + трилінійна інтерполяція по кожному виміру F.
        # Попередньо обчислюємо (1 - frac) щоб не рахувати тричі (1 - fx).
        omfx = 1.0 - fx; omfy = 1.0 - fy; omfz = 1.0 - fz
        # Внутрішній цикл по F-вимірах feature-вектора (F=2).
        for Fi in range(F):
            # 8 значень з 8 кутів.
            f000 = features[Li, h000, Fi]
            f100 = features[Li, h100, Fi]
            f010 = features[Li, h010, Fi]
            f110 = features[Li, h110, Fi]
            f001 = features[Li, h001, Fi]
            f101 = features[Li, h101, Fi]
            f011 = features[Li, h011, Fi]
            f111 = features[Li, h111, Fi]

            # Прохід 1 — по X. 4 проміжні значення.
            c00 = f000 * omfx + f100 * fx
            c10 = f010 * omfx + f110 * fx
            c01 = f001 * omfx + f101 * fx
            c11 = f011 * omfx + f111 * fx
            # Прохід 2 — по Y. 2 проміжні.
            c0  = c00 * omfy + c10 * fy
            c1  = c01 * omfy + c11 * fy
            # Прохід 3 — по Z. 1 фінальне значення. Пишемо одразу в правильну позицію буфера.
            h_buf[Li * F + Fi] = c0 * omfz + c1 * fz

    # КРОК 6: декодер.
    if has_hidden:                                 # MLP з прихованим шаром.
        hidden = W1.shape[0]                       # Динамічно з розмірності матриці W1.
        # Починаємо з фінального зсуву.
        out = b2[0]
        # Цикл по hidden нейронах прихованого шару.
        for j in range(hidden):
            acc = b1[j]                            # Накопичувач: починаємо з b1[j].
            # Внутрішній цикл — добуток W1[j,:] на h_buf.
            for i in range(in_dim):
                acc += W1[j, i] * h_buf[i]
            # ReLU: якщо acc від'ємне, ставимо 0.
            if acc < 0.0:
                acc = 0.0
            # Додаємо до виходу зі вагою W2[0, j].
            out += W2[0, j] * acc
        return out
    else:                                          # Чисто лінійний декодер.
        out = b2[0]                                # Зсув як стартова точка.
        for i in range(in_dim):
            out += W2[0, i] * h_buf[i]             # Скалярний добуток W2 · h_buf.
        return out


# ─── Batched forward pass ───────────────────────────────────────────────────
# parallel=True — Numba може розпаралелити функцію.
@njit(cache=True, fastmath=True, parallel=True)
def _forward_batch(pts: np.ndarray,            # (B, 3) float32 — пакет точок.
                   features: np.ndarray,
                   res: np.ndarray,
                   W1: np.ndarray, b1: np.ndarray,
                   W2: np.ndarray, b2: np.ndarray,
                   has_hidden: bool,
                   out: np.ndarray) -> None:    # Вихід пишеться сюди (in-place, без алокації).
    B = pts.shape[0]
    # prange — паралельний range. Numba автоматично розбиває [0, B) на чанки і запускає на CPU-потоках.
    for i in prange(B):
        # Викликаємо _forward_one для i-ої точки і пишемо результат в out[i].
        out[i] = _forward_one(pts[i, 0], pts[i, 1], pts[i, 2],
                              features, res, W1, b1, W2, b2, has_hidden)


# ─── Бенчмарк-функція для одиничного запиту ─────────────────────────────────
@njit(cache=True, fastmath=True)
def _forward_loop_bench(x: float, y: float, z: float,
                        n_iters: int,
                        features: np.ndarray, res: np.ndarray,
                        W1: np.ndarray, b1: np.ndarray,
                        W2: np.ndarray, b2: np.ndarray,
                        has_hidden: bool) -> float:
    # Накопичуємо в acc, інакше компілятор викине цикл як "мертвий код" (його результат не використовується).
    acc = 0.0
    for _ in range(n_iters):
        # Викликаємо _forward_one з однаковими аргументами n_iters разів.
        acc += _forward_one(x, y, z, features, res, W1, b1, W2, b2, has_hidden)
    return acc


# ─── Зручний клас-обгортка для inference ────────────────────────────────────
class SDFInference:

    def __init__(self, npz_path: str | Path):
        # Завантажуємо .npz файл (експортований sdf_model.py через export_npz).
        z = np.load(npz_path, allow_pickle=False)
        # Парсимо cfg (JSON-рядок) у словник.
        cfg = json.loads(str(z['cfg']))
        self.cfg = cfg
        # Прапорець: чи модель з прихованим шаром (hidden > 0) або чисто лінійна.
        self.has_hidden = cfg.get('hidden', 0) > 0

        # fp16 → fp32. На CPU нативна швидкість fp32, тому конвертуємо при завантаженні
        # (а не на льоту в inference-кернелі).
        self.features = z['features'].astype(np.float32)
        self.res = z['resolutions'].astype(np.int64)

        # Завантажуємо ваги декодера. Дві гілки за has_hidden.
        if self.has_hidden:
            self.W1 = z['W1'].astype(np.float32)
            self.b1 = z['b1'].astype(np.float32)
            self.W2 = z['W2'].astype(np.float32)
            self.b2 = z['b2'].astype(np.float32)
        else:
            # Для лінійного декодера W1/b1 — dummy (numba потрібно щось передати).
            self.W1 = np.zeros((1, 1), dtype=np.float32)
            self.b1 = np.zeros((1,), dtype=np.float32)
            # Реальні ваги лінійного декодера зберігаємо в "вихідних" слотах.
            self.W2 = z['decoder_W'].astype(np.float32)
            self.b2 = z['decoder_b'].astype(np.float32)

    def query_one(self, x: float, y: float, z: float) -> float:
        """Один запит, повертає скалярне SDF."""
        # float(...) — numba може повернути numpy-скаляр, нам потрібен Python-float.
        return float(_forward_one(x, y, z, self.features, self.res,
                                  self.W1, self.b1, self.W2, self.b2,
                                  self.has_hidden))

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        """Пакетний запит. pts форми (B, 3), повертає (B,)."""
        # ascontiguousarray — гарантує безперервну C-style пам'ять (numba не любить non-contiguous).
        pts = np.ascontiguousarray(pts, dtype=np.float32)
        # Алокуємо вихідний масив (без ініціалізації — швидше).
        out = np.empty(pts.shape[0], dtype=np.float32)
        # Виклик паралельного forward — пише в out in-place.
        _forward_batch(pts, self.features, self.res,
                       self.W1, self.b1, self.W2, self.b2,
                       self.has_hidden, out)
        return out

    def warmup(self) -> None:
        """Спрацьовує numba-JIT компіляцію. Перший виклик повільний (~1 сек), наступні швидкі."""
        # Виклик з простими входами — компілятор згенерує машинний код для всіх викликуваних функцій.
        _ = self.query_one(0.0, 0.0, 0.0)
        _ = self(np.zeros((4, 3), dtype=np.float32))

    def bench_single_point(self, n_iters: int = 100_000) -> float:
        """Повертає наносекунди на запит у скомпільованому циклі (без Python-overhead)."""
        import time
        # Прогрів — 100 ітерацій, щоб тригернути JIT-компіляцію _forward_loop_bench.
        _forward_loop_bench(0.1, 0.2, 0.3, 100, self.features, self.res,
                            self.W1, self.b1, self.W2, self.b2, self.has_hidden)
        # Основний таймінг.
        t0 = time.perf_counter()                       # Найточніший таймер Python.
        _forward_loop_bench(0.1, 0.2, 0.3, n_iters, self.features, self.res,
                            self.W1, self.b1, self.W2, self.b2, self.has_hidden)
        dt = time.perf_counter() - t0                  # Час в секундах.
        # Ділимо на n_iters → секунди на ітерацію. * 1e9 → наносекунди.
        return dt / n_iters * 1e9
