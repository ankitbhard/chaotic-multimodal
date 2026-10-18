# PCRF — Perturbation-Calibrated Reliability-Guided Fusion

## Mathematical Formulation

### Encoder Representations

Given patient $i$ with histopathology slide embedding $x_I^{(i)}$ and gene expression
profile $x_G^{(i)}$:

$$z_I^{(i)} = f_I(x_I^{(i)}) \in \mathbb{R}^{512}$$
$$z_G^{(i)} = f_G(x_G^{(i)}) \in \mathbb{R}^{128}$$

where $f_I$ is a SlideEmbeddingEncoder (Linear + BN + ReLU + Dropout) and
$f_G$ is a GeneEncoder (3-layer MLP: n_genes -> 512 -> 256 -> 128).

### Reliability Estimation

Per-modality reliability is estimated by small networks:

$$q_I^{(i)} = R_I(z_I^{(i)}) \in \mathbb{R}$$
$$q_G^{(i)} = R_G(z_G^{(i)}) \in \mathbb{R}$$

where $R_m: \mathbb{R}^{d_m} \to \mathbb{R}$ is a 2-layer MLP (d_m -> 64 -> 1).

### Reliability-Guided Fusion

Fusion weights are computed via softmax over reliability logits:

$$[\alpha_I^{(i)}, \alpha_G^{(i)}] = \text{softmax}([q_I^{(i)}, q_G^{(i)}])$$

The fused representation is:

$$z_F^{(i)} = \alpha_I^{(i)} \cdot P_I(z_I^{(i)}) + \alpha_G^{(i)} \cdot P_G(z_G^{(i)})$$

where $P_m: \mathbb{R}^{d_m} \to \mathbb{R}^{256}$ are linear projection layers.

With explicit modality masks $m \in \{0,1\}^2$:

$$q'_m = q_m + (m_k - 1) \cdot 10^9$$

This forces the softmax weight of missing modalities to ~0.

### Survival Prediction

$$h^{(i)} = \text{ReLU}(W_s \cdot z_F^{(i)} + b_s)$$
$$\hat{y}^{(i)} = W_h \cdot h^{(i)} + b_h \in \mathbb{R}^2$$

Risk score: $r^{(i)} = \hat{y}^{(i)}_1$ (second logit).

### Auxiliary Unimodal Cox Heads

$$r_I^{(i)} = g_I(z_I^{(i)}) \in \mathbb{R}$$
$$r_G^{(i)} = g_G(z_G^{(i)}) \in \mathbb{R}$$

where $g_m: \mathbb{R}^{d_m} \to \mathbb{R}$ is a 2-layer MLP (d_m -> 64 -> 1).

### Loss Function

$$\mathcal{L} = \mathcal{L}_{\text{Cox}}^{\text{joint}} + \lambda_{\text{aux}} (\mathcal{L}_{\text{Cox}}^{\text{img}} + \mathcal{L}_{\text{Cox}}^{\text{gene}}) + \gamma \mathcal{L}_{\text{rel}}$$

where $\mathcal{L}_{\text{Cox}}$ is the negative partial log-likelihood:

$$\mathcal{L}_{\text{Cox}} = -\frac{1}{|\mathcal{D}|} \sum_{i: \delta_i=1} \left[ r_i - \log \sum_{j: t_j \geq t_i} \exp(r_j) \right]$$

Default hyperparameters: $\lambda_{\text{aux}} = 0.1$, $\gamma = 0.01$.

### Perturbation-Calibrated Reliability Supervision

For modality $m$, the perturbation sensitivity is:

$$d_m^{(i)} = \frac{1}{K} \sum_{k=1}^K \left[ g_m(z_m^{(i)} + \delta_k) - g_m(z_m^{(i)}) \right]^2$$

where $\delta_k \sim \mathcal{N}(0, \sigma^2 I)$ and $K=5$ perturbation samples.

Sensitivity is normalized within each training batch:

$$\tilde{d}_m^{(i)} = \frac{d_m^{(i)} - \mu_m}{\sigma_m + \epsilon}$$

Reliability targets (lower sensitivity -> higher reliability):

$$t_m^{(i)} = \sigma(-\tilde{d}_m^{(i)})$$

where $\sigma$ is the sigmoid function.

The reliability loss:

$$\mathcal{L}_{\text{rel}} = \sum_{m \in \{I,G\}} \text{BCE}(\sigma(q_m), t_m^{\text{detach}})$$

**Key properties:**
1. Targets $t_m$ are fully detached — no gradient flows through the sensitivity computation
2. Aux heads are trained with their own Cox loss (on detached encoder features when aux_loss=False)
3. Normalization uses training batch statistics only
4. The reliability loss only updates the reliability networks $R_m$ and propagates through encoders

### Method Variants

| Variant | Aux Cox loss in total | Reliability loss | Architecture |
|---------|----------------------|------------------|--------------|
| adamw | No | No | JointModel (attention fusion) |
| aux_cox | Yes (through encoders) | No | JointModel + AuxCoxHead |
| mod_dropout | No | No | JointModel + training dropout |
| learned_gating | No | No | PCRFModel (same capacity) |
| pcrf | No (detached) | Yes | PCRFModel |
| pcrf_aux | Yes (through encoders) | Yes | PCRFModel |

The learned_gating baseline has **identical** parameter count to PCRF/PCRF_aux,
ensuring that any improvement from reliability supervision is not due to extra capacity.

### Parameter Counts (BRCA, n_genes~1000)

| Component | Parameters |
|-----------|-----------|
| SlideEmbeddingEncoder | ~525K |
| GeneEncoder | ~676K |
| Projection (img + gene) | ~164K |
| Reliability nets | ~41K |
| Shared + survival head | ~66K |
| Aux Cox heads | ~41K |
| **Total (PCRF)** | **~1.51M** |
| JointModel (baseline) | ~1.45M |
