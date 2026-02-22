import torch
import torch.nn as nn
import torch.nn.functional as F

class UIC(nn.Module):
    def __init__(self, D, d=64, hidden_qk=64, patch_size=16, stride=8, dropout=0.1, topk=8, use_topk=True):
        super().__init__()
        self.D = D
        self.d = d
        self.patch_size = patch_size
        self.stride = stride
        self.eps = 1e-8
        self.topk = topk
        self.use_topk = use_topk

        self.var_encoder = nn.Sequential(
            nn.Linear(D, max(D // 2, d), bias=True),
            nn.LeakyReLU(),
            nn.Linear(max(D // 2, d), d, bias=True)
        )

        self.q_lin = nn.Linear(d, hidden_qk, bias=False)
        self.k_lin = nn.Linear(d, hidden_qk, bias=False)
        self.v_lin = nn.Linear(d, hidden_qk, bias=False)

        self.after_agg = nn.Sequential(
            nn.Linear(hidden_qk, d),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

        self.var_decoder = nn.Linear(d, D, bias=True)

    def forward(self, X):
        """
        X: (B, L, D)
        return: P_in (B, L, D)
        """
        B, L, D = X.shape
        assert D == self.D

        # (B, L, d)
        X_e = self.var_encoder(X)

        # patching: (B, d, L) -> unfold -> (B, d, h, e) -> (B, h, e, d)
        X_e_t = X_e.permute(0, 2, 1)
        patches = X_e_t.unfold(dimension=2, size=self.patch_size, step=self.stride)
        patches = patches.permute(0, 2, 3, 1).contiguous()  # (B, h, e, d)
        _, h, e, d = patches.shape

        # (B*h, e, d)
        patches_flat = patches.view(B * h, e, d)

        #  (B*h, d)
        per_patch = patches_flat.mean(dim=1)

        # q,k: (B*h, m)
        q = self.q_lin(per_patch)
        k = self.k_lin(per_patch)
        v = self.v_lin(per_patch)

        # (B*h, m, m)
        qn = q / (q.norm(dim=-1, keepdim=True) + self.eps)
        kn = k / (k.norm(dim=-1, keepdim=True) + self.eps)
        M = torch.bmm(qn.unsqueeze(2), kn.unsqueeze(1)) - torch.bmm(kn.unsqueeze(2), qn.unsqueeze(1))
        A = F.relu(torch.tanh(M))  # (B*h, m, m)


        if self.use_topk and (self.topk is not None) and (self.topk > 0) and (self.topk < A.size(-1)):
            k_keep = min(self.topk, A.size(-1))
            vals, idx = torch.topk(A, k=k_keep, dim=-1)  # (B*h, m, k)
            mask = torch.zeros_like(A, dtype=torch.bool)
            mask.scatter_(-1, idx, True)
            A = A * mask.to(A.dtype)

        A = A / A.sum(dim=-1, keepdim=True).clamp(min=self.eps)

        # v: (B*h, m) -> (B*h, m, 1)
        v_col = v.unsqueeze(-1)
        Av = torch.bmm(A, v_col).squeeze(-1)  # (B*h, m)

        X_A = Av  # (B*h, m)

        z = self.after_agg(X_A)  # (B*h, d)

        #  (B*h, e, d) -> (B, h, e, d)
        X_patch_time = z.unsqueeze(1).expand(-1, e, -1).contiguous()
        X_patch_time = X_patch_time.view(B, h, e, d)

        # (B, L, d)
        out = torch.zeros(B, L, d, device=X.device, dtype=X.dtype)
        counts = torch.zeros(B, L, d, device=X.device, dtype=X.dtype)

        for idx in range(h):
            start = idx * self.stride
            end = min(start + self.patch_size, L)
            seg = X_patch_time[:, idx, : (end - start), :]
            out[:, start:end, :] += seg
            counts[:, start:end, :] += 1.0

        out = out / counts.clamp(min=1.0)

        # (B, L, D)
        P_in = self.var_decoder(out)
        return P_in
