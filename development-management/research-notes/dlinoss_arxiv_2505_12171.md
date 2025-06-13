# Learning to Dissipate Energy in Oscillatory State-Space Models

**arXiv:2505.12171v1 [cs.LG] 17 May 2025**

**Authors:**
- Jared Boyer (MIT CSAIL, jaredb@mit.edu)
- T. Konstantin Rusch (MIT CSAIL)
- Daniela Rus (MIT CSAIL)

---

## Abstract

State-space models (SSMs) are a class of networks for sequence learning that benefit from fixed state size and linear complexity with respect to sequence length, contrasting the quadratic scaling of typical attention mechanisms. Inspired from observations in neuroscience, Linear Oscillatory State-Space models (LinOSS) are a recently proposed class of SSMs constructed from layers of discretized forced harmonic oscillators. Although these models perform competitively, leveraging fast parallel scans over diagonal recurrent matrices and achieving state-of-the-art performance on tasks with sequence length up to 50k, LinOSS models rely on rigid energy dissipation ("forgetting") mechanisms that are inherently coupled to the timescale of state evolution. As forgetting is a crucial mechanism for long-range reasoning, we demonstrate the representational limitations of these models and introduce Damped Linear Oscillatory State-Space models (D-LinOSS), a more general class of oscillatory SSMs that learn to dissipate latent state energy on multiple timescales. We analyze the spectral distribution of the model’s recurrent matrices and prove that the SSM layers exhibit stable dynamics under simple, flexible parameterizations. D-LinOSS consistently outperforms previous LinOSS methods on long-range learning tasks, without introducing additional complexity, and simultaneously reduces the hyperparameter search space by 50%.

---

## 1. Introduction

State-space models (SSMs) [Gu et al. 2021, Smith et al. 2023, Gu and Dao 2023, Hasani et al. 2022, Rusch and Rus 2025] have emerged as a powerful deep learning architecture for sequence modeling, demonstrating strong performances across various domains, including natural language processing [Gu and Dao 2023], audio generation [Goel et al. 2022], reinforcement learning [Lu et al. 2024], and scientific and engineering applications [Hu et al. 2024].

Despite the abundance of neural network architectures for sequence modeling, SSMs have gained significant attention due to their fundamental advantages over both Recurrent Neural Networks (RNNs) and Transformer architectures based on self-attention mechanisms [Vaswani 2017]. Built upon layers of sequence-to-sequence transformations defined by linear dynamical systems, SSMs integrate principles from control theory with modern deep learning techniques, making them highly effective across multiple modalities. While recent SSM architectures are often formulated as linear RNNs [Orvieto et al. 2023], they introduce notable improvements over their predecessors, offering enhanced speed, accuracy, and the ability to capture long-range dependencies more effectively.

In this work, we focus on and further extend the recently introduced linear oscillatory state-space model (LinOSS) [Rusch and Rus 2025]. LinOSS is based on a system of second-order ordinary differential equations (ODEs) that model forced harmonic oscillators and is discretized using fast associative parallel scans. The structure of the underlying oscillatory dynamics allows LinOSS to learn long-range interactions over arbitrary time scales without enforcing any constraints on the SSM state matrix. However, as we will subsequently show, LinOSS inherently couples frequency and damping, reducing latent state energy dissipation to a single scale and limiting the model’s expressive power. To overcome this, we propose Damped Linear Oscillatory State-Space Models (D-LinOSS), which enhance the LinOSS architecture by incorporating learnable damping across all scales.

Our approach constructs a deep state space model capable of capturing a wide range of temporal relationships by expanding the expressivity of individual SSM layers. Unlike previous versions of LinOSS that were constrained to a limited subset of oscillatory systems, our method allows each layer to independently learn a wider range of stable oscillatory dynamics, collectively leading to a more powerful sequence model. Our full contributions are:

- We conduct a rigorous spectral analysis of the proposed D-LinOSS model, highlighting the representational improvements enabled by learnable damping.
- We validate the theoretical expressivity improvements through a synthetic experiment of learning exponential decay.
- We derive a stable parameterization of D-LinOSS and introduce an initialization procedure to generate arbitrary eigenvalue distributions in the recurrent matrix. We perform ablations comparing different initialization techniques.
- We provide extensive empirical evaluation, showing that D-LinOSS on average outperforms state-of-the-art models across eight different challenging real-world sequential datasets. At the same time, D-LinOSS reduces the hyperparameter space of previous LinOSS models by 50% by eliminating the need for multiple discretization schemes.
- To support reproducibility and further research, we release our code and experiments at [github.com/jaredbmit/damped-linoss](https://github.com/jaredbmit/damped-linoss).

---

## 2. Background

### 2.1 Underlying continuous-time dynamics

D-LinOSS layers are constructed from a system of damped, forced harmonic oscillators, represented in the following state-space formulation:

$$
x''(t) = -A x(t) - G x'(t) + B u(t),\\
y(t) = C x(t) + D u(t)
$$

The system input $u : [0, T] \to \mathbb{R}^p$, the system state $x : [0, T] \to \mathbb{R}^m$, and the system output $y : [0, T] \to \mathbb{R}^q$ are all vector-valued functions of continuous time $t$. The parameters $A$ and $G$ are restricted to diagonal matrices with non-negative entries, meaning (1) is an uncoupled second-order system. The feed-forward operation $D u(t)$ will be omitted for the rest of the paper for concision.

The goal of a D-LinOSS layer is to model complex sequence-to-sequence transformations $u \mapsto y$ by learning parameters $A, G, B, C, D$. $A$ controls the natural frequency of the system’s oscillation and $G$ defines the damping, i.e., the energy dissipation of the latent state. The underlying dynamical system of previous LinOSS models is (1) subject to $G = 0$; thus, D-LinOSS is constructed from a more general oscillatory state space layer with learnable damping. The additional $m$ learnable parameters per layer are a negligible contribution to model size and have no impact on speed.

### 2.2 Discretization

To approximately solve the ODE system in (1), we first rewrite it as an equivalent first-order system by introducing the auxiliary state variable $z(t) \in \mathbb{R}^m$. The full state $[z, x]^\top$ is denoted $w \in \mathbb{R}^{2m}$. A discretization scheme is then applied to the first-order system, mapping the parameters $A, G, B, C$ to discrete-time counterparts $M, F, H$ in the system:

$$
z'(t) = -A x(t) - G z(t) + B u(t),\\
x'(t) = z(t),\\
y(t) = C x(t)
$$

Discretize:

$$
w_{k+1} = M w_k + F u_{k+1},\\
y_{k+1} = H w_k
$$

Unlike standard first-order SSMs, LinOSS explicitly models the acceleration and velocity of the system state, resulting in smoother outputs due to the twice-integrated structure. However, this structure necessitates the use of special discretization schemes to maintain system stability.

Most SSMs discretize the underlying continuous-time dynamics using zero-order hold or the bilinear method. However, due to the second-order nature of LinOSS, both implicit-explicit (IMEX) and fully implicit (IM) methods are leveraged to ensure stability. These integrators endow different “forgetting” behaviors into the discrete-time systems; the IM integrator introduces an energy dissipation term whereas the IMEX integrator completely preserves energy over time. The selection of discretization technique is a binary hyperparameter in the original LinOSS model and gives rise to two flavors of LinOSS (LinOSS-IM and LinOSS-IMEX) that exhibit different dynamical behaviors.

We extend the use of the structure-preserving implicit-explicit method to the D-LinOSS layer, as using the implicit method would introduce additional dissipative terms that are both unnecessary and uncontrollable for the learning process. Applying the IMEX discretization to System (2) yields:

$$
z_{k+1} = z_k + \Delta t (-A x_k - G z_{k+1} + B u_{k+1}),\\
x_{k+1} = x_k + \Delta t z_{k+1}
$$

or in matrix form:

$$
\begin{bmatrix} I + \Delta t G & 0 \\ -\Delta t I & I \end{bmatrix} \begin{bmatrix} z_{k+1} \\ x_{k+1} \end{bmatrix} = \begin{bmatrix} I - \Delta t A & 0 \\ 0 & I \end{bmatrix} \begin{bmatrix} z_k \\ x_k \end{bmatrix} + \begin{bmatrix} \Delta t B u_k \\ 0 \end{bmatrix}
$$

Inverting the left hand side block matrix and re-arranging terms, we arrive at the final discrete-time formulation, parameterized by $M \in \mathbb{R}^{2m \times 2m}$, $F \in \mathbb{R}^{2m \times p}$, and $H \in \mathbb{R}^{q \times 2m}$:

$$
M := \begin{bmatrix} S^{-1} & -\Delta t S^{-1} A \\ \Delta t S^{-1} & I - \Delta t^2 S^{-1} A \end{bmatrix}, \\
F := \begin{bmatrix} \Delta t S^{-1} B \\ \Delta t^2 S^{-1} B \end{bmatrix}, \\
H := [0, C]
$$

Here, the Schur complement is the diagonal matrix $S = I + \Delta t G$ and $M$ and $F$ are block matrices composed of diagonal sub-matrices.

### 2.3 Diagonal equivalence

For general SSMs, evaluating the recurrence $w_{k+1} = M w_k + F u_{k+1}$ with a dense transition matrix $M$ is computationally expensive, which can be prohibitive for both training and inference on long sequences. However, when $M$ is diagonalizable, the system can be rewritten equivalently:

$$
\tilde{w}_{k+1} = \Lambda \tilde{w}_k + \tilde{F} u_{k+1}, \\
y_{k+1} = \tilde{H} \tilde{w}_k
$$

where $\Lambda$ is diagonal and the change of variables follows:

$$
M = V \Lambda V^{-1}, \quad \tilde{w} = V^{-1} w, \quad \tilde{F} = V^{-1} F, \quad \tilde{H} = H V
$$

In this formulation, the recurrence $\Lambda \tilde{w}_k$ becomes a vector dot-product, reducing the computational cost to $O(m)$ per step. In practice, many SSMs are learned directly in this diagonalized space, avoiding the cost of explicitly computing $V$ or $V^{-1}$.

In the case of D-LinOSS, the recurrent matrix $M$ is composed of block matrices which are diagonal (see (5)), so computing $M w_k$ already requires only $O(m)$ operations and diagonalization is not strictly necessary for efficient implementation.

### 2.4 Associative parallel scans

Many modern SSM architectures rely on associative parallel scan algorithms [Kogge and Stone 1973, Blelloch 1990] to efficiently compute recurrent operations across long sequences. The key idea is to exploit the associativity of the recurrence operator to parallelize what would otherwise be a sequential computation.

Given a binary associative operator $\bullet$, satisfying $(a \bullet b) \bullet c = a \bullet (b \bullet c)$, the cumulative product over a sequence $[a, b, c, ...]$:

$$
[a, a \bullet b, a \bullet b \bullet c, ...]
$$

can be computed in $O(\log N)$ sequential steps instead of $O(N)$, where $N$ is the sequence length. This transformation is commonly referred to as an operator scan.

For SSMs, associative scans enable efficient computation of the recurrence $w_{k+1} = M w_k + F u_k$ when $M$ is time-invariant, acting as a key building block for scaling SSMs to long contexts.

---

## 3. Theoretical properties

Spectral analysis provides a powerful lens to examine the stability and dynamical behavior of SSMs. In the absence of bounding nonlinearities like tanh, the eigenvalues of the recurrent matrix $M$ fully govern how latent states evolve across time. In particular, eigenvalues with near unit norm retain energy across long time horizons, while those closer to zero rapidly dissipate energy.

In the previous LinOSS-IM and LinOSS-IMEX models, which are based on a system of harmonic oscillators, the internal system spectra are rigidly defined by the selection of discretization technique, tightly coupling frequency and damping. As shown in Figure 1, this reduces latent state energy dissipation to a single scale when normalizing frequency, limiting the range of expressible dynamics.

For D-LinOSS, the spectrum of $M$ instead arises from the structure of damped harmonic oscillators, introducing a new tunable mechanism that decouples damping from frequency. Unlike the preceding models, D-LinOSS layers can represent all stable second-order systems, yielding a broader range of expressible dynamics and thus a more powerful sequence model. This is depicted in Figure 1, where the scale of energy dissipation can be arbitrarily selected regardless of oscillation frequency.

These notions are formalized in this section, where we characterize the eigenvalues of D-LinOSS, derive stability conditions, and compare the resulting spectral range to that of previous LinOSS models. In particular, we rigorously show that the set of reachable, stable eigenvalue configurations in D-LinOSS is the full complex unit disk, where that of LinOSS has zero measure in $\mathbb{C}$.

### 3.1 Spectral analysis and stability

**Proposition 3.1.** The eigenvalues of the D-LinOSS recurrent matrix $M \in \mathbb{R}^{2m \times 2m}$ are:

$$
\lambda_{i1,2} = \frac{1 + \frac{\Delta t_i}{2} G_i - \frac{\Delta t_i^2}{2} A_i}{1 + \Delta t_i G_i} \pm \frac{\Delta t_i}{2} \frac{\sqrt{(G_i - \Delta t_i A_i)^2 - 4A_i}}{1 + \Delta t_i G_i}
$$

where pairs of eigenvalues are denoted as $\lambda_{i1,2}$ and $i = 1, 2, ..., m$.

**Proposition 3.2.** Assume $G_i, A_i$ are non-negative, and $\Delta t_i \in (0, 1]$. If the following is satisfied:

$$
(G_i - \Delta t_i A_i)^2 \leq 4A_i
$$

then $\lambda_{i1,2}$ come in complex conjugate pairs $\lambda_i, \lambda_i^*$ with the following magnitude:

$$
|\lambda_i| = \frac{1}{\sqrt{1 + \Delta t_i G_i}} \leq 1
$$

Define $S_i$ to be the set of all $(G_i, A_i)$ that satisfy the above condition.

**Proposition 3.3.** The mapping $\Phi : S_i \to \mathbb{C}_{|z| \leq 1}$ defined by $(G_i, A_i) \mapsto \lambda_i$ is bijective.

Compared to D-LinOSS, the preceding LinOSS-IM and LinOSS-IMEX are limited in the set of reachable eigenvalues. The eigenvalues of these two models are:

$$
\lambda^{IMEX}_{i1,2} = \frac{1}{1 + \Delta t_i^2 A_i} \pm j \frac{\Delta t_i \sqrt{A_i}}{1 + \Delta t_i^2 A_i}, \\
\lambda^{IM}_{i1,2} = \frac{1}{2}(2 - \Delta t_i^2 A_i) \pm \frac{j}{2} \sqrt{\Delta t_i^2 A_i (4 - \Delta t_i^2 A_i)}
$$

Both forms impose a rigid relationship between oscillation frequency and damping, constraining the set of reachable spectra. The set of stable eigenvalues reachable under these parameterizations occupies zero area within the unit disk.

---

## 4. Results

### 4.1 UEA time-series classification

Test accuracies averaged over five different seeds on UEA time-series classification datasets:

| Model         | Worms        | SCP1         | SCP2         | Ethanol      | Heartbeat    | Motor        | Avg  |
|---------------|--------------|--------------|--------------|--------------|--------------|--------------|------|
| NRDE          | 83.9 ± 7.3   | 80.9 ± 2.5   | 53.7 ± 6.9   | 25.3 ± 1.8   | 72.9 ± 4.8   | 47.0 ± 5.7   | 60.6 |
| NCDE          | 75.0 ± 3.9   | 79.8 ± 5.6   | 53.0 ± 2.8   | 29.9 ± 6.5   | 73.9 ± 2.6   | 49.5 ± 2.8   | 60.2 |
| Log-NCDE      | 85.6 ± 5.1   | 83.1 ± 2.8   | 53.7 ± 4.1   | 34.4 ± 6.4   | 75.2 ± 4.6   | 53.7 ± 5.3   | 64.3 |
| LRU           | 87.8 ± 2.8   | 82.6 ± 3.4   | 51.2 ± 3.6   | 21.5 ± 2.1   | 78.4 ± 6.7   | 48.4 ± 5.0   | 61.7 |
| S5            | 81.1 ± 3.7   | 89.9 ± 4.6   | 50.5 ± 2.6   | 24.1 ± 4.3   | 77.7 ± 5.5   | 47.7 ± 5.5   | 61.8 |
| S6            | 85.0 ± 16.1  | 82.8 ± 2.7   | 49.9 ± 9.4   | 26.4 ± 6.4   | 76.5 ± 8.3   | 51.3 ± 4.7   | 62.0 |
| Mamba         | 70.9 ± 15.8  | 80.7 ± 1.4   | 48.2 ± 3.9   | 27.9 ± 4.5   | 76.2 ± 3.8   | 47.7 ± 4.5   | 58.6 |
| LinOSS-IMEX   | 80.0 ± 2.7   | 87.5 ± 4.0   | 58.9 ± 8.1   | 29.9 ± 1.0   | 75.5 ± 4.3   | 57.9 ± 5.3   | 65.0 |
| LinOSS-IM     | 95.0 ± 4.4   | 87.8 ± 2.6   | 58.2 ± 6.9   | 29.9 ± 0.6   | 75.8 ± 3.7   | 60.0 ± 7.5   | 67.8 |
| D-LinOSS      | 93.9 ± 3.2   | 88.9 ± 3.0   | 58.6 ± 2.3   | 29.9 ± 0.6   | 75.8 ± 4.9   | 61.1 ± 2.0   | 68.0 |

### 4.2 PPG-DaLiA time-series regression

| Model         | MSE ×10⁻²    |
|---------------|--------------|
| NRDE          | 9.90 ± 0.97  |
| NCDE          | 13.54 ± 0.69 |
| Log-NCDE      | 9.56 ± 0.59  |
| LRU           | 12.17 ± 0.49 |
| S5            | 12.63 ± 1.25 |
| S6            | 12.88 ± 2.05 |
| Mamba         | 10.65 ± 2.20 |
| LinOSS-IMEX   | 7.5 ± 0.46   |
| LinOSS-IM     | 6.4 ± 0.23   |
| D-LinOSS      | 6.16 ± 0.73  |

### 4.3 Weather time-series forecasting

| Model         | Mean Absolute Error |
|---------------|---------------------|
| Informer      | 0.731               |
| Informer†     | 0.741               |
| LogTrans      | 0.773               |
| Reformer      | 1.575               |
| LSTMa         | 1.109               |
| LSTnet        | 0.757               |
| S4            | 0.578               |
| LinOSS-IMEX   | 0.508               |
| LinOSS-IM     | 0.528               |
| D-LinOSS      | 0.486               |

---

## 5. Related Work

State Space Models (SSMs) were originally introduced as a powerful deep learning framework for sequential data in [Gu et al. 2021]. Early models [Gu et al. 2022, Nguyen et al. 2022, Goel et al. 2022] primarily leveraged Fast Fourier Transform (FFT) and HiPPO [Gu et al. 2020] parameterizations to efficiently solve linear recurrences. Over time, SSMs have undergone continuous refinement. More recently, most SSM architectures utilize diagonal state matrices combined with fast associative parallel scans, which has been used in the context of RNNs before [Martin and Cundy 2017, Kaul 2020]. This approach was first introduced to SSMs in [Smith et al. 2023], which still relied on HiPPO matrices for initializing SSM weights, but has since been simplified to random weight initialization, as demonstrated in [Orvieto et al. 2023]. In addition, while our proposed D-LinOSS model and all aforementioned models are based on linear time-invariant (LTI) systems, there is increasing interest in SSMs based on time-varying systems [Gu and Dao 2023, Hasani et al. 2022, Merrill et al. 2024].

The most closely related model to our proposed D-LinOSS is the original LinOSS, introduced in [Rusch and Rus 2025]. While LinOSS was the first SSM built on oscillatory dynamics, several other deep learning models also incorporate oscillatory behavior. These include recurrent architectures like coupled oscillatory RNNs (coRNNs) [Rusch and Mishra 2021a] and UnICORNNs [Rusch and Mishra 2021b], as well as graph-based models such as Graph Coupled Oscillator Networks (GraphCON) [Rusch et al. 2022].

---

## 6. Discussion and conclusion

We introduced D-LinOSS, an extension of the LinOSS model that incorporates learnable damping across all temporal scales. Through spectral analysis, we showed that existing LinOSS variants are rigidly defined by their discretization scheme and can only express a narrow set of dynamical behaviors. In contrast, D-LinOSS captures the full range of stable, damped oscillatory dynamics. This expanded expressivity yields a 10–30× improvement on a synthetic regression task, and leads to consistent performance gains across eight real-world benchmarks. D-LinOSS outperforms all baselines considered in this work, including Transformer-based models, LSTM variants, other modern SSMs, and previous versions of LinOSS. Additionally, D-LinOSS reduces the LinOSS hyperparameter search space by 50% without adding any computational overhead. These results establish D-LinOSS as an efficient and powerful extension to the family of deep state space models.

While D-LinOSS demonstrates strong empirical results as a general sequence model, it is based on layers of LTI dynamical systems, which are fundamentally limited in their ability to capture certain contextual dependencies, such as the selective copying task [Gu and Dao 2023, Jing et al. 2019]. Building on the growing interest in time-varying SSMs sparked by [Gu and Dao 2023], we aim to explore future work on selective variants of LinOSS that integrate the efficiency and expressiveness of LinOSS-type models with the powerful selectivity mechanism enabled by time-varying dynamics. As D-LinOSS is inherently well-suited to represent temporal relationships with oscillatory structure, we aim to explore applications to domains where such patterns are fundamental. In particular, climate science, seismic monitoring, and astrophysics data all exhibit complex patterns governed by oscillatory behavior. Moving forward, we believe that D-LinOSS will play an increasingly central role in advancing machine-learning based approaches in domains grounded in the physical sciences.

---

## Acknowledgments

This work was supported in part by the Postdoc.Mobility grant P500PT-217915 from the Swiss National Science Foundation, the Schmidt AI2050 program (grant G-22-63172), and the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

The authors acknowledge the MIT SuperCloud [Reuther et al. 2018] and Lincoln Laboratory Supercomputing Center for providing HPC resources that have contributed to the research results reported within this paper.

---

## References

*The full reference list is included in the original arXiv paper. For brevity, see the arXiv:2505.12171 PDF or the references section above for all cited works.*

---

## Supplementary Material

### A. Theoretical properties

#### A.1 Derivation of the eigenvalues of D-LinOSS

(See main text for full derivation and equations.)

#### A.2 Proof of the stability criterion

(See main text for full derivation and equations.)

#### A.3 Spectral image of D-LinOSS

(See main text for full derivation and equations.)

#### A.4 Proof of the set measure of LinOSS eigenvalues

(See main text for full derivation and equations.)

#### A.5 Universality of LinOSS

(See main text for full theorem statement and discussion.)

### B. Experiments and results

#### B.1 Parameterization

(See main text for parameterization details and equations.)

#### B.2 Regression experiment

(See main text for experiment setup and results.)

#### B.3 Hyperparameters

(See main text for hyperparameter grid and best settings.)

#### B.4 Compute requirements

(See main text for compute resource table and discussion.)

#### B.5 Initialization techniques

(See main text for initialization study and results.)

---

*This Markdown file is a direct conversion of the content from arXiv:2505.12171 for research reference in the LinossRust project. For figures, equations, and full references, consult the original PDF.*
