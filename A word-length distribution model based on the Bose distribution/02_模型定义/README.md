# 02_模型定义

词长分布拟合的全部 11 个候选模型（概率质量函数 PMF）。

## 模型清单

| model_id | 中文名 | 参数 | 说明 |
|----------|--------|------|------|
| `shifted_poisson` | 1-位移泊松 | λ | P(ℓ)=Pois(ℓ-1;λ) |
| `gamma` | Gamma | α, β | 连续 Gamma，率参数 β |
| `poisson_mixture` | 混合泊松 | π, λ₁, λ₂ | 两成分泊松混合 |
| `zipf_alekseev` | Zipf-Alekseev | a, b | P(ℓ)∝(ℓ+a)⁻ᵇ |
| `negative_binomial` | 负二项 | r, p | NB(ℓ-1; r, p) |
| `hyper_poisson` | Hyper-Poisson | λ, b | g(k)=λᵏ/Γ(k+b) |
| `log_normal` | 对数正态 | μ, σ | 对数正态 PDF |
| `bose` | 玻色子模型 | d, β, μ̃, scale | 词例全积分 / 词型低温近似 |
| `ehp` | 指数超泊松 | λ, b, α | P(ℓ)=F_HP(ℓ-1)^α−F_HP(ℓ-2)^α |
| `generalized_gamma` | 广义伽马(Stacy) | a, b, p | Stacy 分布 |
| `sichel` | 西切尔 | θ, σ², γ | 逆高斯-泊松混合 |

## 玻色子模型（核心）

- **词例分布**（全数值积分）：`P(ℓ) ∝ ℓ^(d/2) · B(d/2, β(ℓ-1)+μ̃)`
- **词型分布**（低温近似）：`P(ℓ) ∝ ℓ^(d/2) · exp(-βℓ+μ̃)`
- `B(ν,z)` 为广义玻色积分，实现见 `bose_integral()`（提取自原始实验脚本）。

## 主要函数

- `bose_integral(nu, z)`：广义玻色积分（含小 z 近似）
- `pmf_<model>(ell, *params)`：各模型归一化 PMF
- `bose_model(ell, d, beta, mu_tilde, scale, dist_type)`：玻色子预测计数
- `MODEL_REGISTRY`：模型注册表（shape 参数名、是否拟合 scale）
- `predict(model_id, ell, params, total_count, dist_type)`：统一预测接口

## 用法示例

```python
from models import predict, MODEL_REGISTRY
import numpy as np

ell = np.arange(1, 13, dtype=float)
# 玻色子词例分布
y = predict("bose", ell,
            {"d": 3.5, "beta": 1.4, "mu_tilde": 1.4, "scale": 1e5},
            dist_type="token")
```
