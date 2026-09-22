# AdsDrift EquiformerV3 局部等变生成器

本目录是 `test_18` 生成器，继承 `test_16`，只将块内聚合改为“自注意力残差 → 交叉注意力残差 → FFN”。初始化与参数数量不变。固定层继续作为 query/key/value 并更新特征；其坐标仍固定。模型入口为 `generator.py`；构造条件协议和编码器位于相邻的 `../condition/`。从 EquiformerV3 固定下来的底层算子整理在 `core/`。

## 1. 输入与输出

给定可分解条件

$$
c=(S_{\rm prim},hkl,s_{\rm term},b,M,n_{\rm layer},v_{\rm vac},\varepsilon,A),
$$

以及同一条件下的随机初始结构库

$$
R_0\in\mathbb R^{B\times G\times N\times3},
$$

生成器一次前向得到

$$
\widehat R=G_\theta(R_0;Z,t,L,c).
$$

$Z$、$t$、$L$ 是已经构造出的完整 slab 图；$c$ 显式记录原胞、晶面、终止、超胞矩阵、层数、真空、应变和吸附物。完整图负责局部物理，$c$ 负责把同一材料背后的控制变量暴露给网络；没有 system-ID embedding。

条件编码器在原胞分数坐标上建立最小镜像周期图，使用距离、相对表面法向投影和面内距离进行消息传递。原胞池化表示与晶格度量、实际 slab 度量、约化 Miller 指数、面间距、终止相位、超胞矩阵及其物理度量等不变量拼接。原胞原子换序或整体旋转不会改变条件标量。

编码器输出初始角色偏置 $B_t(c)$，以及每层每种角色的缩放和平移 $(\gamma_t^{(l)},\beta_t^{(l)})$：

$$
X_{i,0}^{(0)}=E_Z(Z_i)+E_t(t_i)+B_{t_i}(c),
$$

$$
\widetilde X_{i,0}^{(l)}=X_{i,0}^{(l)}
\left[1+s_c\tanh\gamma_{t_i}^{(l)}(c)\right]
+s_c\beta_{t_i}^{(l)}(c).
$$

只调制 $l=0$ 标量，再把结果送入 slab self、adsorbate self 和 interface cross attention；$l>0$ 的张量变换规则不变，所以坐标输出仍保持 SE(3) 等变。

模型只改变 role 2 和 role 3 的坐标：

$$
\widehat R_i=R_{0,i},\qquad t_i=1.
$$

## 2. 周期局部图

对源原子 $j$、目标原子 $i$ 和周期平移 $n\in\mathbb Z^3$，定义有向边向量

$$
r_{ji}^{(n)}=R_{0,j}+n^\top L-R_{0,i},
\qquad d_{ji}^{(n)}=\|r_{ji}^{(n)}\|_2.
$$

默认只枚举 $n_a,n_b\in\{-1,0,1\}$、$n_c=0$，每种关系对每个 query 最多保留24条最近边。三类边为

$$
\mathcal E_{SS}=\{j\in S_{\rm all}\to i\in S_{\rm all}:d_{ji}<r_S\},
$$

$$
\mathcal E_{AA}=\{j\in A\to i\in A:j\ne i,\ d_{ji}<r_A\},
$$

$$
\mathcal E_{SA}=\{S_{\rm all}\to A\}\cup\{A\to S_{\rm all}\},
\qquad d_{ji}<r_C.
$$

默认 $r_S=6.0$ Å、$r_A=4.5$ Å、$r_C=6.0$ Å。完整 slab 包括固定层和可弛豫层；固定层参与 slab self-attention 和 adsorbate → slab cross-attention 的查询，仍受同样的距离截断和邻居数限制。

## 3. SO(3) 输入表示

每个节点的特征为

$$
X_i\in\mathbb R^{(l_{\max}+1)^2\times C},
\qquad C=128,\quad l_{\max}=3,\quad m_{\max}=2.
$$

标量通道初始化为

$$
X_{i,l=0}^{(0)}=E_Z(Z_i)+E_t(t_i)+B_{t_i}(c).
$$

局部边距离通过64个 Gaussian radial basis 展开：

$$
g_k(d)=\exp[-\gamma(d-\mu_k)^2].
$$

径向网络生成沿边局部坐标系的 $m=0$ 系数，再通过 Wigner-D 逆旋转产生球谐边嵌入：

$$
X_i^{(0)}\leftarrow X_i^{(0)}+
\frac1{\sqrt{K}}
\sum_{(j\to i)\in\mathcal E_{\rm all}}
D^{(l)}(Q_{ji})^{-1}\rho_l(g(d_{ji}),E_Z^{\rm src}(Z_j),E_Z^{\rm dst}(Z_i)).
$$

$Q_{ji}$ 把边方向对齐到局部轴。因为绝对坐标不进入标量 MLP，网络对整体平移不变；球谐/Wigner 路径保证高阶特征随旋转按对应不可约表示变换。此初始化沿用 `test_16`，几何边嵌入包括 $l=0$ 到 $l_{\max}$ 的分量，全原子图仍会在第一层之前引入界面环境信息。本版没有隔离初始化。

## 4. EquiformerV3 图注意力

对关系 $q\in\{SS,AA,SA\}$，传入节点特征前先做等变归一化和已有条件调制 $C_k$（只改变 $l=0$）：

$$
\bar X=C_k\!\left(\operatorname{EqNorm}(X)\right).
$$

每条边把源和目标表示旋转到边对齐坐标系，结合元素嵌入及径向权重：

$$
M_{ji}^{q}=\operatorname{SO2Conv}_2\!\left(
\sigma_{S^2}\!\left[
\operatorname{SO2Conv}_1
\left(D(Q_{ji})\,\Psi(\bar X_j,\bar X_i;d_{ji})\right)
\right]\right).
$$

`use_add_merge=true` 时，$\Psi$ 对源、目标使用不同的径向通道权重后相加，避免拼接带来的约两倍 SO(2) 输入计算。$\sigma_{S^2}$ 使用 EquiformerV3 的 SwiGLU-$S^2$ 激活。

注意力权重是旋转不变量：

$$
\alpha_{ji,h}^{q}=
\operatorname{softmax}_{j\in\mathcal N_q(i)}
\left(a_h^\top u_{ji,h}^{q}\right).
$$

多头 value 加权、旋转回全局坐标并按目标节点聚合：

$$
A_i^q=\sum_{j\in\mathcal N_q(i)}
D(Q_{ji})^{-1}\left(\alpha_{ji}^q M_{ji}^q\right).
$$

三个分支参数互不共享；两个交叉方向共享同一套 interface 参数。令 $m=\mathbf1_{S_{\rm all}\cup A}$，单层先进行自注意力残差更新：

$$
U = X + \frac{m}{\sqrt2}\odot
\left[A^{SS}(\bar X)+A^{AA}(\bar X)\right].
$$

$$
\bar U=C_k(N_{\rm attn}(U)),\qquad
V=U+\frac{m}{\sqrt2}\odot A^{SA}(\bar U).
$$

交叉注意力读取的是更新后的 $U$，不是旧的 $X$；其后才计算 FFN：

$$
X^+=V+m\odot\operatorname{FFN}_{\mathrm{SwiGLU}-S^2}
\!\left(C_k(N_{\rm ffn}(V))\right).
$$

两次 $N_{\rm attn}$ 复用同一组原有归一化参数，$C_k$ 也复用原有 FiLM；各注意力残差仍保留原系数 $1/\sqrt2$，没有增加参数或截断梯度。两条 self 路径的目标集合不重叠。

默认串联4层。固定节点的 $X_i$ 与其他有效节点一起接受注意力和 FFN 残差；self 更新后的固定层表示也会被本层 cross 读取，并继续进入下一层。padding 特征不更新。这里更新的是隐藏表示，不是中间坐标；输出头仍然只在末尾一次性修改可弛豫表面和吸附物坐标。

## 5. 平滑截断

距离注意力使用五阶多项式包络。令 $x=d/r_c$：

$$
e(d)=
\begin{cases}
1+ax^p+bx^{p+1}+cx^{p+2},&x<1,\\
0,&x\ge1,
\end{cases}
$$

其中 $p=5$，$a=-(p+1)(p+2)/2$，$b=p(p+2)$，$c=-p(p+1)/2$。包络既参与 softmax 的指数权重，也乘入边嵌入，使相互作用在截断位置平滑衰减。

## 6. 从 l=1 读取坐标位移

末层等变归一化后取

$$
V_i=X_{i,l=1}\in\mathbb R^{3\times C}.
$$

三个输出头仅对通道做共享线性组合：

$$
v_{i,k}=\sum_{c=1}^{C}w_cV_{i,kc},\qquad k\in\{x,y,z\}.
$$

同一组 $w_c$ 用于三个空间分量，因此不会破坏 $l=1$ 的旋转变换规律。表面位移为

$$
\Delta R_i^{S}=s_S v_i^S,\qquad i\in S_{\rm mov}.
$$

吸附物分解为整体和平移之外的内部变化：

$$
\Delta c=s_c\frac1{|A|}\sum_{i\in A}v_i^c,
$$

$$
\Delta r_i=s_r\left(v_i^r-\frac1{|A|}\sum_{j\in A}v_j^r\right),
\qquad \sum_{i\in A}\Delta r_i=0.
$$

最终

$$
\widehat R_i=
\operatorname{wrap}_{ab}(\bar R_0+\Delta c)
+(R_{0,i}-\bar R_0)+\Delta r_i,
\qquad i\in A.
$$

只包裹吸附物中心，不逐原子包裹，从而避免把跨周期分子撕开。默认 $s_S=0.5$ Å、$s_c=3.0$ Å、$s_r=1.0$ Å。三个头零初始化，所以未训练模型严格从恒等映射开始。

## 7. 当前边界

- 这是一步条件生成器，不是力场，也不做多步弛豫。
- 当前没有全局模式 token 或盆地路由；多模态性完全来自不同 $R_0$ 及其局部交互。
- 吸附物构象通过逐原子 $l=1$ 内部残差改变，没有显式键长/键角约束；几何有效性仍由数据、Drifting 特征与结构检查共同约束。
- 旧 CatFlow checkpoint 与本模型结构不兼容，加载时应明确报错。

## 8. 默认规模

三套固定体系配置位于 `../../config/`，它们共享完全相同的模型参数：4层、128通道、8头、$l_{\max}=3$、$m_{\max}=2$、FFN 256、每种关系最多24邻居；只允许数据体系和输出目录不同。
