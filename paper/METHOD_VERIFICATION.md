# RA-SAMO Method Verification: Equations ↔ Code Mapping

## 1. Cox Partial Likelihood (Eq. 6)

**Paper**: $\mathcal{L}_{\text{cox}} = -\frac{1}{|\mathcal{E}|} \sum_{i: \delta_i = 1} (r_i - \log \sum_{j: t_j \geq t_i} e^{r_j})$

**Code**: `experiments/samo/objectives.py:5-17` → `cox_efron_free(risk, time, event)`
- Uses Breslow approximation (not Efron despite function name)
- Normalizes by number of events: `.mean()` on event subset
- Risk set computed via masked `logsumexp`: `time[j] >= time[i]`
- Verified: ✅ Exact match

## 2. Pairwise Ranking Loss (Eq. 7)

**Paper**: $\mathcal{L}_{\text{rank}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \log(1 + e^{(r_j - r_i)/\tau})$

**Code**: `experiments/samo/objectives.py:19-32` → `ranking_loss(risk, time, event, temperature)`
- `comparable_pairs()` computes mask: event[i]=1 AND time[i] < time[j]
- Uses `F.softplus(margin[mask]).mean()` — softplus = log(1+exp(x))
- Temperature parameter divides margin
- Verified: ✅ Exact match

## 3. EMA Loss Weight Controller (Eq. 8-9)

**Paper**: Bounded EMA-based weight update with clip and warmup

**Code**: `experiments/samo/objectives.py:34-77` → `LossWeightController`
- `ema=0.9`, `min_weight=0.02`, `max_weight=0.3`, `max_change=0.02`, `warmup=2`
- Computes ratio: `smooth[k] / baseline[k]` for each objective
- Target: `0.1 * ratio / center` (center = mean of ratios)
- Clipped delta: `max(-0.02, min(0.02, target - current))`
- Final clip to [0.02, 0.3]
- Verified: ✅ Exact match

## 4. Gradient Projection (Eq. 10-11)

**Paper**: $\hat{\mathbf{g}}_m^{\text{sec}} = \mathbf{g}_m^{\text{sec}} - \frac{\langle \mathbf{g}_m^{\text{sec}}, \mathbf{g}_m^{\text{pri}} \rangle}{\|\mathbf{g}_m^{\text{pri}}\|^2} \mathbf{g}_m^{\text{pri}}$

**Code**: `experiments/samo/representation_gradients.py:40-46` → `_project(aux, primary)`
- Conflict detection: `_dot(aux, primary) < 0 and denom > eps`
- Projection: `a - coeff * p` where `coeff = dot(aux, pri) / dot(pri, pri)`
- Returns (projected_grad, conflict_flag)
- Verified: ✅ Exact match (standard PCGrad-style projection)

## 5. Combined Gradient Replacement (Eq. 12)

**Paper**: $\mathbf{g}_m = \mathbf{g}_m^{\text{pri}} + w_{\text{aux}} \hat{\mathbf{g}}_m^{\text{aux}} + w_{\text{rank}} \hat{\mathbf{g}}_m^{\text{rank}}$

**Code**: `experiments/samo/representation_gradients.py:72-80`
- Before projection: `original = [p + w_aux * a + w_rank * q]`
- After projection: `combined = [p + w_aux * proj_a + w_rank * proj_r]`
- Computes relative change: `||combined - original|| / ||original||`
- Verified: ✅ Exact match

## 6. prepare_branch_gradients (Algorithm 1, lines 3-12)

**Code**: `experiments/samo/representation_gradients.py:49-91`
- Iterates over ("image", image_loss, w_img) and ("gene", gene_loss, w_gene)
- `_grads(loss, params)` uses `torch.autograd.grad(retain_graph=True)`
- Records cosine similarities and projection flags in `BranchDiagnostics`
- Verified: ✅ Matches algorithm pseudocode

## 7. apply_branch_gradients (Algorithm 1, line 15)

**Code**: `experiments/samo/representation_gradients.py:94-99`
- Replaces `param.grad = grad.clone()` for each encoder parameter
- Returns diagnostics dict
- Verified: ✅ Matches algorithm pseudocode

## 8. Cross-Modal Attention Fusion (Eq. 3)

**Code**: `models/fusion.py` → `CrossModalAttention`
- Projects both modalities to `FUSED_DIM=256`
- Attention: `W_a (64×256)` → `tanh` → `u (64)` → softmax
- Output: weighted sum of projected representations
- Verified: ✅ Exact match

## 9. Encoder Architectures

**Image encoder** (Eq. 1): `models/encoders.py` → `SlideEmbeddingEncoder`
- `Linear(d_img, 512)` → BN → ReLU → Dropout(0.3)
- Verified: ✅

**Gene encoder** (Eq. 2): `models/encoders.py` → `GeneEncoder`
- `Linear(G, 512)` → BN → ReLU → Dropout → `Linear(512, 256)` → BN → ReLU → Dropout → `Linear(256, 128)` → BN → ReLU → Dropout
- Verified: ✅

## 10. HAM10000 Adaptation

**Image encoder**: `experiments/ham10000/models.py` → `ResNet18Encoder`
- Pretrained ResNet-18, output dim 512 (removes final FC)
- Verified: ✅

**Metadata encoder**: `experiments/ham10000/models.py` → `MetadataEncoder`
- 3-layer MLP: input → 64 → 64 → 64
- Verified: ✅

**Gene encoder alias**: `HAM10000ModelWithAux.gene_encoder` property → `self._meta_encoder`
- Required for `prepare_branch_gradients` which accesses `model.gene_encoder`
- Verified: ✅

**Loss adaptation**: `experiments/ham10000/objectives.py` → `classification_loss_parts`
- Returns (fusion_ce, img_aux_ce, meta_aux_ce, zero_ranking)
- Ranking always `fusion_ce * 0.0` (zero tensor with grad connection)
- Verified: ✅
