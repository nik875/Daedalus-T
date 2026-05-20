"""
Daedalus attack reimplemented in PyTorch targeting YOLO26's NMS-free one2one head.
Optimizer: mSAM (micro-batch Sharpness-Aware Minimization).

Two attack modes:
  Daedalus       — digital full-image perturbation (equivalent to l2_yolov3.py)
  DaedalusPoster — physical poster patch with Expectation over Transformations (EOT)

Key implementation notes:
  - YOLO26's one2one head detaches its feature inputs in Detect.forward to
    prevent gradient interference during model training.  We patch this out
    so gradients flow back to the input for the attack.
  - Loss: mean((sigmoid(one2one_scores) - 1)^2) across all 8400 slots.
    No w*h term — without NMS there is nothing to exploit with tiny boxes.
  - mSAM ascent uses per-transform (per-EOT-sample) normalised gradients
    averaged together, so no single augmentation dominates the ascent
    direction.  This is where mSAM differs meaningfully from standard SAM.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import kornia.augmentation as KA
from tqdm import tqdm
from skimage import io
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect


# ---------------------------------------------------------------------------
# Patch YOLO26 Detect head
# ---------------------------------------------------------------------------
# The one2one head detaches its feature inputs by design (prevents gradient
# interference between heads during model training). For the attack we need
# gradients to flow from the one2one scores all the way back to the image.

_original_detect_forward = Detect.forward


def _patched_detect_forward(self, x):
    preds = self.forward_head(x, **self.one2many)
    if self.end2end:
        one2one = self.forward_head(x, **self.one2one)  # no detach
        preds = {"one2many": preds, "one2one": one2one}
    if self.training:
        return preds
    y = self._inference(preds["one2one"] if self.end2end else preds)
    if self.end2end:
        y = self.postprocess(y.permute(0, 2, 1))
    return y if self.export else (y, preds)


Detect.forward = _patched_detect_forward


# ---------------------------------------------------------------------------
# mSAM optimizer
# ---------------------------------------------------------------------------

class mSAM:
    """
    Micro-batch Sharpness-Aware Minimization.

    Each optimisation step requires two forward+backward passes:
      1. ascent_step(): perturb params in the direction of steepest gradient
                        ascent, scaled to L2 ball of radius rho.
      2. descent_step(): compute gradient at the perturbed point, restore
                         original params, apply base_optimizer update.

    With batch_size=1 this is identical to standard SAM.  With larger batches
    (e.g. EOT), ascent is computed per-example so each example gets its own
    perturbation direction rather than following the mean gradient.
    """

    def __init__(self, params, base_optimizer_cls, rho=0.05, **base_kwargs):
        self.params = list(params)
        self.base_optimizer = base_optimizer_cls(self.params, **base_kwargs)
        self.rho = rho
        self._saved_e = {}

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        grads = [p.grad for p in self.params if p.grad is not None]
        if not grads:
            return torch.tensor(1e-12)
        return torch.norm(torch.stack([g.norm(2) for g in grads]))

    def ascent_step(self):
        """Perturb params toward steepest ascent (SAM neighbourhood)."""
        norm = self._grad_norm() + 1e-12
        for p in self.params:
            if p.grad is None:
                continue
            e_w = (self.rho / norm) * p.grad.detach()
            p.data.add_(e_w)
            self._saved_e[id(p)] = e_w

    def descent_step(self):
        """Restore params then apply base optimizer using current gradients."""
        for p in self.params:
            buf = self._saved_e.pop(id(p), None)
            if buf is not None:
                p.data.sub_(buf)
        self.base_optimizer.step()
        self.base_optimizer.zero_grad()


# ---------------------------------------------------------------------------
# Daedalus attack
# ---------------------------------------------------------------------------

# Defaults
CONFIDENCE       = 0.3
LEARNING_RATE    = 3e-3
BINARY_SEARCH_STEPS = 5
MAX_ITERATIONS   = 10000
ABORT_EARLY      = True
INITIAL_CONST    = 2.0
SAM_RHO          = 0.025
IMAGE_SIZE       = 640
MODEL_PATH       = "yolo26n.pt"
SAVE_PATH        = "adv_examples/yolo26/"


class Daedalus:
    """
    Daedalus adversarial example generator targeting YOLO26's one2one head.

    The attack finds a minimal-L2 perturbation that drives all 8400 raw
    pre-filter classification scores toward 1.0, causing the model's top-300
    selection to return 300 high-confidence spurious detections.
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        confidence=CONFIDENCE,
        learning_rate=LEARNING_RATE,
        binary_search_steps=BINARY_SEARCH_STEPS,
        max_iterations=MAX_ITERATIONS,
        abort_early=ABORT_EARLY,
        initial_const=INITIAL_CONST,
        rho=SAM_RHO,
        device=None,
    ):
        self.confidence = confidence
        self.lr = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.rho = rho
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        yolo = YOLO(model_path)
        self.model = yolo.model.to(self.device)
        # eval() gives stable BatchNorm (uses pretrained running stats) so the
        # model behaves as it does in deployment.  The Detect head must stay in
        # train() so its forward() returns the {"one2many","one2one"} dict we need.
        self.model.eval()
        self.model.model[-1].train()
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scores(self, newimgs):
        """Return sigmoid class scores from the one2one head: (N, 80, 8400)."""
        out = self.model(newimgs)
        return torch.sigmoid(out["one2one"]["scores"])

    def _adv_loss(self, scores):
        """Mean squared push toward confidence=1 across all slots and classes."""
        return torch.mean((scores - 1.0) ** 2)

    def _l2_dist(self, newimgs, orig):
        """Per-image L2 distortion between adversarial and original: (N,)."""
        return torch.sum((newimgs - orig) ** 2, dim=[1, 2, 3])

    def _to_img(self, w_orig, delta):
        """tanh reparameterisation keeps pixels in [0, 1]."""
        return torch.tanh(w_orig + delta) * 0.5 + 0.5

    def _check_success(self, adv_loss, init_adv_loss):
        return adv_loss <= init_adv_loss * (1.0 - self.confidence)

    # ------------------------------------------------------------------
    # Core attack
    # ------------------------------------------------------------------

    def _attack_single(self, img_np):
        """
        Attack one image.

        img_np: (H, W, 3) float32 in [0, 1]
        returns: (H, W, 3) float32 adversarial image
        """
        orig = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )  # (1, 3, H, W)

        # arctanh-space representation of the original image
        w_orig = torch.arctanh((orig * 2.0 - 1.0).clamp(-0.999999, 0.999999))

        best_l2 = float("inf")
        best_adv = orig.clone()

        lower_bound = 0.0
        upper_bound = 1e10
        const = self.initial_const

        for outer in range(self.binary_search_steps):
            print(f"  [search {outer + 1}/{self.binary_search_steps}] const={const:.5g}")

            delta = torch.zeros_like(w_orig, requires_grad=True)
            optimizer = mSAM([delta], torch.optim.Adam, rho=self.rho, lr=self.lr)

            init_adv_loss = None
            prev_loss = None
            step_adv_loss = None

            for it in range(self.max_iterations):
                # --- SAM ascent: first forward pass ---
                optimizer.zero_grad()
                newimgs = self._to_img(w_orig, delta)
                scores = self._scores(newimgs)
                adv_loss = self._adv_loss(scores)
                l2 = self._l2_dist(newimgs, orig)
                loss = const * adv_loss + l2.mean()
                loss.backward()
                optimizer.ascent_step()

                # --- SAM descent: second forward pass at perturbed point ---
                optimizer.zero_grad()
                newimgs = self._to_img(w_orig, delta)
                scores = self._scores(newimgs)
                adv_loss = self._adv_loss(scores)
                l2 = self._l2_dist(newimgs, orig)
                loss = const * adv_loss + l2.mean()
                loss.backward()
                optimizer.descent_step()

                step_adv_loss = adv_loss.item()
                step_l2 = l2.mean().item()
                step_loss = loss.item()

                if init_adv_loss is None:
                    init_adv_loss = step_adv_loss
                    prev_loss = step_loss * 1.1

                if it % (self.max_iterations // 10) == 0:
                    print(
                        f"    iter {it:5d} | loss={step_loss:.4f} | "
                        f"adv={step_adv_loss:.4f} | l2={step_l2:.4f}"
                    )

                # early abort if not making progress
                if self.abort_early and it % (self.max_iterations // 10) == 0:
                    if step_loss > prev_loss * 0.9999:
                        print("    early stop")
                        break
                    prev_loss = step_loss

                # track best successful result
                if (
                    self._check_success(step_adv_loss, init_adv_loss)
                    and step_l2 < best_l2
                ):
                    best_l2 = step_l2
                    best_adv = newimgs.detach().clone()

            # adjust const for next binary search step
            if self._check_success(step_adv_loss, init_adv_loss):
                upper_bound = min(upper_bound, const)
                const = (lower_bound + upper_bound) / 2.0 if upper_bound < 1e9 else const / 2.0
            else:
                lower_bound = max(lower_bound, const)
                const = (lower_bound + upper_bound) / 2.0 if upper_bound < 1e9 else const * 10.0

        return best_adv.squeeze(0).permute(1, 2, 0).cpu().numpy()

    def attack(self, imgs, save_path=SAVE_PATH):
        """
        Attack a list of images and save results.

        imgs: list of (H, W, 3) float32 numpy arrays in [0, 1]
        returns: (N, H, W, 3) numpy array of adversarial images
        """
        os.makedirs(save_path, exist_ok=True)
        results = []
        distortions = []

        for i, img in enumerate(imgs):
            print(f"\n=== Image {i + 1}/{len(imgs)} ===")
            adv = self._attack_single(img)
            l2 = float(np.sum((adv - img) ** 2))
            print(f"  Final L2 distortion: {l2:.4f}")

            results.append(adv)
            distortions.append(l2)
            io.imsave(
                os.path.join(save_path, f"adv_{i:04d}_l2={l2:.3f}.png"),
                (adv * 255).clip(0, 255).astype(np.uint8),
            )

        results = np.array(results)
        np.savez(
            os.path.join(save_path, "batch.npz"),
            X_adv=results,
            distortions=np.array(distortions),
        )
        return results


# ---------------------------------------------------------------------------
# EOT compositing via kornia
# ---------------------------------------------------------------------------

# Shared augmentation module — instantiated once, reused every call.
# RandomAffine with same_on_batch=False gives each image its own random
# scale + rotation + translation in a single differentiable op.
_eot_aug = KA.RandomAffine(
    degrees=10,
    translate=(0.35, 0.35),
    p=1.0,
    same_on_batch=False,
)


def apply_patch_eot(patch, backgrounds, eot_num, scale_range=(0.15, 0.25)):
    """
    Composite the patch onto each background under EOT augmentation.

    patch       : (1, 3, Ph, Pw)  — adversarial poster being optimised
    backgrounds : (N, 3, H, W)    — scene images (detached, not optimised)
    eot_num     : random transforms to sample per background image
    scale_range : (min, max) patch width as a fraction of scene width

    Returns
    -------
    composites : (N * eot_num, 3, H, W)
    masks      : (N * eot_num, 1, H, W)

    Gradient flow: composites → p_aug → p_canvas → F.interpolate → patch ✓
    Using kornia's differentiable RandomAffine avoids in-place assignment,
    which was silently breaking the gradient chain in the previous version.
    """
    device = patch.device
    N, _, H, W = backgrounds.shape

    composite_list, mask_list = [], []

    for _ in range(eot_num):
        # Random patch size for this transform (controls apparent poster size)
        scale = torch.empty(1).uniform_(*scale_range).item()
        ts = max(8, int(scale * W))

        # Resize patch to target size, expand to batch
        p_batch = F.interpolate(
            patch.expand(N, -1, -1, -1),
            size=(ts, ts), mode="bilinear", align_corners=False,
        )  # (N, 3, ts, ts)

        # Pad patch and a matching binary mask to full canvas (patch at top-left)
        p_canvas = F.pad(p_batch,   (0, W - ts, 0, H - ts))  # (N, 3, H, W)
        m_canvas = F.pad(
            torch.ones(N, 1, ts, ts, device=device),
            (0, W - ts, 0, H - ts),
        )  # (N, 1, H, W)

        # Apply same random affine to patch and mask so they stay aligned
        p_aug = _eot_aug(p_canvas)
        m_aug = _eot_aug(m_canvas, params=_eot_aug._params).clamp(0.0, 1.0)

        # Alpha-composite patch over background
        composites = backgrounds.detach() * (1.0 - m_aug) + p_aug * m_aug
        composite_list.append(composites)
        mask_list.append(m_aug)

    return torch.cat(composite_list, dim=0), torch.cat(mask_list, dim=0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SceneDataset(torch.utils.data.Dataset):
    """Loads all images from a directory, resized to a fixed size."""

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, image_dir, size=IMAGE_SIZE):
        self.size = size
        self.paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(image_dir)
            for f in files
            if os.path.splitext(f)[1].lower() in self.EXTS
        ]
        if not self.paths:
            raise FileNotFoundError(f"No images found in {image_dir}")
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


# ---------------------------------------------------------------------------
# DaedalusPoster — universal physical poster attack with EOT
# ---------------------------------------------------------------------------

PATCH_SIZE        = 100    # poster patch size in pixels (before EOT scaling)
EOT_NUM           = 10     # random transforms per batch step
ADV_LOSS_WEIGHT   = 1.0
EPOCHS            = 10
BATCH_SIZE        = 8
POSTER_SAVE_PATH  = "adv_examples/yolo26_poster/"


class DaedalusPoster:
    """
    Universal physical poster attack for YOLO26.

    Trains a single patch over a full image dataset so it is adversarial
    across arbitrary background scenes, not just a fixed set.  Each training
    step composites the patch onto a mini-batch of backgrounds under EOT
    random transforms (scale, rotation, translation) and pushes all 8400
    one2one candidates toward confidence=1.

    mSAM: the ascent step averages per-EOT-transform normalised gradients so
    no single augmentation dominates the perturbation direction.
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        patch_size=PATCH_SIZE,
        eot_num=EOT_NUM,
        adv_loss_weight=ADV_LOSS_WEIGHT,
        learning_rate=LEARNING_RATE,
        rho=SAM_RHO,
        scale_range=(0.15, 0.25),
        device=None,
    ):
        self.patch_size = patch_size
        self.eot_num = eot_num
        self.adv_loss_weight = adv_loss_weight
        self.lr = learning_rate
        self.rho = rho
        self.scale_range = scale_range
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        yolo = YOLO(model_path)
        self.model = yolo.model.to(self.device)
        # eval() gives stable BatchNorm (uses pretrained running stats) so the
        # model behaves as it does in deployment.  The Detect head must stay in
        # train() so its forward() returns the {"one2many","one2one"} dict we need.
        self.model.eval()
        self.model.model[-1].train()
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scores(self, imgs):
        """Sigmoid one2one class scores: (B, 80, 8400)."""
        return torch.sigmoid(self.model(imgs)["one2one"]["scores"])

    def _adv_loss(self, scores):
        """
        Push the top-300 scoring slots toward confidence 1.

        Focusing on top-K avoids diluting gradients across the ~672K slots
        that are dead regardless of the patch.  300 matches YOLO26's output
        budget so the loss directly targets the slots that will appear in the
        final detection output.
        """
        topk = scores.reshape(scores.shape[0], -1).topk(300, dim=1).values
        return torch.mean((topk - 1.0) ** 2)

    def _top_score(self, scores):
        """Mean of top-300 scores — the same quantity optimised by _adv_loss."""
        return scores.reshape(scores.shape[0], -1).topk(300, dim=1).values.mean().item()

    # ------------------------------------------------------------------
    # mSAM ascent — micro-batch: one gradient per EOT transform
    # ------------------------------------------------------------------

    def _msam_ascent(self, patch_w, backgrounds):
        """
        Perturb patch_w toward the mean of per-transform normalised gradients.

        Each EOT transform is a separate 'micro-batch example': its gradient
        is normalised independently before averaging, so rare-but-important
        viewpoints are not drowned out by the majority.

        Returns the ascent vector so the caller can restore patch_w before
        the descent step.
        """
        e_ws = []
        for _ in range(self.eot_num):
            if patch_w.grad is not None:
                patch_w.grad.zero_()

            patch = torch.sigmoid(patch_w)
            composites, _ = apply_patch_eot(
                patch, backgrounds, eot_num=1, scale_range=self.scale_range
            )
            loss = self.adv_loss_weight * self._adv_loss(self._scores(composites))
            loss.backward()

            g = patch_w.grad.detach().clone()
            e_ws.append(self.rho / (g.norm(2) + 1e-12) * g)

        e_w = torch.stack(e_ws).mean(0)
        patch_w.data.add_(e_w)
        return e_w

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _step(self, patch_w, optimizer, backgrounds):
        # Ascent
        optimizer.zero_grad()
        e_w = self._msam_ascent(patch_w, backgrounds)

        # Descent at perturbed point
        optimizer.zero_grad()
        patch = torch.sigmoid(patch_w)
        composites, _ = apply_patch_eot(
            patch, backgrounds, eot_num=self.eot_num, scale_range=self.scale_range
        )
        scores = self._scores(composites)
        adv_loss = self._adv_loss(scores)
        loss = self.adv_loss_weight * adv_loss
        loss.backward()

        grad_norm = patch_w.grad.norm().item() if patch_w.grad is not None else 0.0
        top_score = self._top_score(scores.detach())

        patch_w.data.sub_(e_w)          # restore before Adam step
        optimizer.base_optimizer.step()
        optimizer.zero_grad()

        return adv_loss.item(), grad_norm, top_score

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        image_dir,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        save_path=POSTER_SAVE_PATH,
        checkpoint_every=1,
    ):
        """
        Train the universal poster patch over a full image dataset.

        image_dir        : directory of background scene images (e.g. COCO val2017)
        epochs           : number of full passes over the dataset
        batch_size       : backgrounds per gradient step
        save_path        : directory for patch checkpoints and composites
        checkpoint_every : save patch image every N epochs

        Returns the final patch as (Ph, Pw, 3) float32 [0, 1].
        """
        os.makedirs(save_path, exist_ok=True)

        dataset = SceneDataset(image_dir)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=(self.device == "cuda"), drop_last=True,
        )

        # Patch in sigmoid-space: sigmoid(0) = 0.5, a neutral grey start
        patch_w = torch.zeros(
            1, 3, self.patch_size, self.patch_size,
            device=self.device, requires_grad=True,
        )
        optimizer = mSAM([patch_w], torch.optim.AdamW, rho=self.rho, lr=self.lr)
        total_steps = epochs * len(loader)
        warmup_steps = max(1, int(total_steps * 0.1))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer.base_optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer.base_optimizer,
                    start_factor=0.1, end_factor=1.0, total_iters=warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer.base_optimizer,
                    T_max=total_steps - warmup_steps, eta_min=self.lr * 0.01,
                ),
            ],
            milestones=[warmup_steps],
        )

        print(
            f"\n=== Training universal poster patch ({self.patch_size}×{self.patch_size}) ===\n"
            f"    dataset : {len(dataset)} images  |  batch : {batch_size}"
            f"  |  EOT : {self.eot_num}  |  epochs : {epochs}\n"
        )

        for epoch in range(1, epochs + 1):
            epoch_adv, epoch_grad, epoch_top, n_steps = 0.0, 0.0, 0.0, 0

            pbar = tqdm(loader, desc=f"epoch {epoch:3d}/{epochs}", leave=True, ascii=True)
            for backgrounds in pbar:
                backgrounds = backgrounds.to(self.device)
                adv, grad_norm, top_score = self._step(patch_w, optimizer, backgrounds)
                scheduler.step()
                epoch_adv  += adv
                epoch_grad += grad_norm
                epoch_top  += top_score
                n_steps    += 1
                pbar.set_postfix(
                    adv=f"{adv:.4f}",
                    top300=f"{top_score:.4f}",
                    grad=f"{grad_norm:.2e}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            epoch_adv  /= n_steps
            epoch_grad /= n_steps
            epoch_top  /= n_steps
            print(
                f"  -> epoch {epoch:3d} summary | "
                f"adv={epoch_adv:.4f} top300={epoch_top:.4f} grad={epoch_grad:.2e}"
            )

            if epoch % checkpoint_every == 0:
                self._save(patch_w, save_path, tag=f"epoch{epoch:03d}")

        patch = self._save(patch_w, save_path, tag="final")
        return patch

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def _save(self, patch_w, save_path, tag=""):
        patch_np = (
            torch.sigmoid(patch_w).squeeze(0).permute(1, 2, 0)
            .detach().cpu().numpy()
        )
        fname = f"patch_{tag}.png" if tag else "patch.png"
        io.imsave(
            os.path.join(save_path, fname),
            (patch_np * 255).clip(0, 255).astype(np.uint8),
        )
        np.save(os.path.join(save_path, fname.replace(".png", ".npy")), patch_np)
        return patch_np


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(path, size=IMAGE_SIZE):
    """Load, resize to model input size, and normalise to [0, 1]."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["digital", "poster"], default="poster")
    parser.add_argument("--image-dir", default="../Datasets/COCO/val2017/")
    # digital mode only
    parser.add_argument("--num-images", type=int, default=10)
    # poster mode only
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    args = parser.parse_args()

    if args.mode == "digital":
        imgs = []
        for root, _, files in os.walk(args.image_dir):
            for f in sorted(files):
                if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                imgs.append(preprocess(os.path.join(root, f)))
                if len(imgs) >= args.num_images:
                    break
            if len(imgs) >= args.num_images:
                break
        if not imgs:
            raise FileNotFoundError(f"No images found in {args.image_dir}")
        Daedalus().attack(imgs)
    else:
        DaedalusPoster().train(
            args.image_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
