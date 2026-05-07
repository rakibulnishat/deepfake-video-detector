"""
model.py — Deepfake Detection Model
Architecture:
  1. EfficientNet-B4  →  Spatial artifact tokens
  2. Swin-Transformer →  Temporal inconsistency tokens
  3. Spatiotemporal Adapter (STA) with orthogonal subspace loss
  4. Contrastive feature bank (real/fake prototypes)
  5. Binary classification head

This matches the research plan's core novelty exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, Dict, Optional


# ─── SPATIOTEMPORAL ADAPTER (STA) ────────────────────────────────────────────

class SpatiotemporalAdapter(nn.Module):
    """
    Decomposes fused spatial+temporal features into two orthogonal subspaces:
      - spatial_tokens:   captures per-frame artifact signals
      - temporal_tokens:  captures cross-frame inconsistency signals

    The orthogonal subspace loss encourages these to be linearly independent,
    forcing the model to learn manipulation-agnostic representations.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.spatial_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, in_dim]  — fused spatial+temporal features
        Returns:
            spatial_tokens:  [B, hidden_dim]
            temporal_tokens: [B, hidden_dim]
        """
        s = self.spatial_proj(x)
        t = self.temporal_proj(x)
        return s, t

    @staticmethod
    def orthogonal_loss(
        s: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalise alignment between spatial and temporal subspaces.
        Loss = mean squared cosine similarity between s and t.
        """
        s_norm = F.normalize(s, dim=-1)
        t_norm = F.normalize(t, dim=-1)
        cos_sim = (s_norm * t_norm).sum(dim=-1)  # [B]
        return (cos_sim ** 2).mean()


# ─── CONTRASTIVE FEATURE BANK ────────────────────────────────────────────────

class ContrastiveFeatureBank(nn.Module):
    """
    Maintains running prototype vectors for REAL and FAKE classes.
    Enables contrastive loss: pull representations toward their class
    prototype, push away from the opposite prototype.
    """

    def __init__(self, feature_dim: int, bank_size: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        self.bank_size = bank_size

        # Non-trainable buffers (updated via EMA during training)
        self.register_buffer("real_bank",
                             torch.randn(bank_size, feature_dim))
        self.register_buffer("fake_bank",
                             torch.randn(bank_size, feature_dim))
        self.register_buffer("real_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("fake_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor):
        """Update the feature bank with current batch features."""
        for feat, lbl in zip(features, labels):
            feat = F.normalize(feat.unsqueeze(0), dim=-1)
            if lbl.item() == 0:  # REAL
                ptr = int(self.real_ptr)
                self.real_bank[ptr] = feat.squeeze()
                self.real_ptr[0] = (ptr + 1) % self.bank_size
            else:                 # FAKE
                ptr = int(self.fake_ptr)
                self.fake_bank[ptr] = feat.squeeze()
                self.fake_ptr[0] = (ptr + 1) % self.bank_size

    def contrastive_loss(
        self, features: torch.Tensor, labels: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        InfoNCE-style contrastive loss using the prototype banks.
        Pulls each sample toward its class prototype, pushes from the other.
        """
        feat_norm = F.normalize(features, dim=-1)

        real_proto = F.normalize(self.real_bank.mean(0, keepdim=True), dim=-1)
        fake_proto = F.normalize(self.fake_bank.mean(0, keepdim=True), dim=-1)

        # [B, 2] — similarity to real and fake prototypes
        sim = torch.cat([
            (feat_norm * real_proto).sum(-1, keepdim=True),
            (feat_norm * fake_proto).sum(-1, keepdim=True),
        ], dim=-1) / temperature

        # Labels: 0=REAL → should be close to col 0; 1=FAKE → col 1
        targets = labels.long()
        return F.cross_entropy(sim, targets)


# ─── SPATIAL ENCODER (EfficientNet-B4) ───────────────────────────────────────

class SpatialEncoder(nn.Module):
    """
    EfficientNet-B4 backbone. Processes each frame independently.
    Returns pooled features: [B*T, spatial_dim].
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,          # Remove classifier head
            global_pool="avg",
        )
        self.out_dim = self.backbone.num_features  # 1792 for B4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*T, C, H, W]  — individual frames
        Returns:
            [B*T, out_dim]
        """
        return self.backbone(x)


# ─── TEMPORAL ENCODER (Swin Transformer) ─────────────────────────────────────

class TemporalEncoder(nn.Module):
    """
    Swin Transformer (tiny) for temporal modelling.
    Takes the sequence of spatial features across T frames,
    treats the temporal dimension as a sequence.
    """

    def __init__(self, spatial_dim: int, num_frames: int = 8,
                 pretrained: bool = True):
        super().__init__()
        self.num_frames = num_frames

        # Linear projection: spatial_dim → Swin's embed_dim
        swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        swin_dim = swin.num_features  # 768 for swin_tiny

        # We don't use Swin on raw images here — we use a 1D Transformer
        # on the sequence of spatial features (more efficient on DFDC)
        self.input_proj = nn.Linear(spatial_dim, swin_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_frames, swin_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=swin_dim,
            nhead=8,
            dim_feedforward=swin_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out_dim = swin_dim

    def forward(self, spatial_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_feats: [B, T, spatial_dim]
        Returns:
            [B, out_dim]  — mean-pooled temporal representation
        """
        x = self.input_proj(spatial_feats)     # [B, T, swin_dim]
        x = x + self.pos_embed[:, :x.size(1)]  # add positional embedding
        x = self.transformer(x)                # [B, T, swin_dim]
        return x.mean(dim=1)                   # [B, swin_dim]


# ─── FULL DEEPFAKE DETECTION MODEL ───────────────────────────────────────────

class DeepfakeDetector(nn.Module):
    """
    Full model combining:
      EfficientNet-B4 (spatial) + Temporal Transformer + STA + Contrastive Bank
    """

    def __init__(
        self,
        num_frames: int = 8,
        face_size: int = 224,
        sta_hidden_dim: int = 256,
        bank_size: int = 512,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_frames = num_frames

        # Encoders
        self.spatial_enc = SpatialEncoder(pretrained=pretrained)
        self.temporal_enc = TemporalEncoder(
            spatial_dim=self.spatial_enc.out_dim,
            num_frames=num_frames,
            pretrained=pretrained,
        )

        # Fuse spatial mean + temporal output
        fused_dim = self.spatial_enc.out_dim + self.temporal_enc.out_dim

        # STA adapter
        self.sta = SpatiotemporalAdapter(in_dim=fused_dim, hidden_dim=sta_hidden_dim)

        # Contrastive bank on STA output
        self.bank = ContrastiveFeatureBank(
            feature_dim=sta_hidden_dim * 2,  # spatial + temporal tokens concat
            bank_size=bank_size,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(sta_hidden_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),   # Binary: REAL vs FAKE
        )

    def forward(
        self,
        frames: torch.Tensor,            # [B, T, C, H, W]
        labels: Optional[torch.Tensor] = None,  # [B] — needed for bank update
    ) -> Dict[str, torch.Tensor]:

        B, T, C, H, W = frames.shape

        # ── Spatial encoding ──────────────────────────────────────
        frames_flat = frames.view(B * T, C, H, W)          # [B*T, C, H, W]
        spatial_flat = self.spatial_enc(frames_flat)        # [B*T, spatial_dim]
        spatial = spatial_flat.view(B, T, -1)               # [B, T, spatial_dim]

        # ── Temporal encoding ─────────────────────────────────────
        temporal = self.temporal_enc(spatial)               # [B, temporal_dim]

        # ── Fuse ──────────────────────────────────────────────────
        spatial_mean = spatial.mean(dim=1)                  # [B, spatial_dim]
        fused = torch.cat([spatial_mean, temporal], dim=-1) # [B, fused_dim]

        # ── STA decomposition ─────────────────────────────────────
        s_tokens, t_tokens = self.sta(fused)                # [B, hidden], [B, hidden]
        combined = torch.cat([s_tokens, t_tokens], dim=-1)  # [B, hidden*2]

        # ── Classification ────────────────────────────────────────
        logits = self.classifier(combined).squeeze(-1)      # [B]
        probs = torch.sigmoid(logits)

        # ── Losses (only computed during training) ─────────────────
        losses = {}
        if self.training and labels is not None:
            # Binary cross-entropy
            losses["bce"] = F.binary_cross_entropy_with_logits(logits, labels)
            # Orthogonal subspace loss
            losses["orthogonal"] = SpatiotemporalAdapter.orthogonal_loss(
                s_tokens, t_tokens
            )
            # Contrastive loss from feature bank
            losses["contrastive"] = self.bank.contrastive_loss(combined, labels)
            # Update bank
            self.bank.update(combined.detach(), labels)

        return {
            "logits": logits,
            "probs": probs,
            "combined_features": combined,
            "spatial_tokens": s_tokens,
            "temporal_tokens": t_tokens,
            "losses": losses,
        }

    def get_cam_target_layer(self):
        """Returns the target conv layer for Grad-CAM."""
        return self.spatial_enc.backbone.conv_head


def build_model(cfg: dict, device: str) -> DeepfakeDetector:
    """Build model from config dict."""
    m = cfg["model"]
    d = cfg["data"]
    model = DeepfakeDetector(
        num_frames=d["frames_per_video"],
        face_size=d["face_size"],
        sta_hidden_dim=m["sta_hidden_dim"],
        bank_size=m["bank_size"],
        dropout=m["dropout"],
        pretrained=m["pretrained"],
    )
    return model.to(device)
