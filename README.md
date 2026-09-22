# AdsDrift：条件吸附结构的一步生成模型

> **项目状态 / Project status**
>
> AdsDrift 是作者开发的研究原型，目前原始开发已暂停，现按 Apache-2.0
> 许可证公开，欢迎新的维护者、复现工作与后续研究从这里继续。本仓库按现状
> 提供，不代表成熟的软件发布，也不承诺持续维护。
>
> This repository is an as-is research handoff. The original development is
> paused, and new maintainers, reproductions, and research extensions are
> welcome.

AdsDrift 面向催化剂筛选：给定一个固定的表面—吸附物体系，从覆盖晶面的随机初始结构出发，一次生成器前向计算得到一组接近低能终点的候选结构。训练目标是该体系下的低能构型分布；生成结果可以用于后续去重、能量排序和短弛豫。

## 开源范围

仓库包含当前模型架构、训练与生成代码、配置样例和数据制备工具。以下内容
**不随仓库分发**：训练数据、AdsDrift 训练 checkpoint、运行结果，以及
MACE-MH-1 权重。MACE-MH-1 需要从其
[官方发布页](https://huggingface.co/mace-foundations/mace-mh-1) 单独获取并遵守
上游 ASL 许可证；可通过环境变量 `ADSDRIFT_MACE_CHECKPOINT` 指定本地路径。

希望继续项目的贡献者可以先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)；第三方代码
来源与许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## MACE-MH-1 在 AdsDrift 中的作用

AdsDrift 采用 **MACE-MH-1 作为冻结的特征编码器**。训练时使用
`oc20_usemppbe` head，从 MACE-MH-1 的两个 interaction layer 提取吸附物与
可移动表面原子的表示，并在该特征空间中构造 Drifting 分布学习目标。
MACE-MH-1 的参数始终冻结，但生成结构到 MACE 特征的坐标计算图保持可微，
因此特征损失的梯度能够传回 AdsDrift 生成器。

MACE-MH-1 在这里不是待训练的 AdsDrift 生成器，也不是用于最终结构排序的
能量标签器。训练完成后，仅执行 AdsDrift 生成器的一步结构生成时不需要运行
MACE。仓库不分发 MACE-MH-1 权重；请从
[MACE-MH-1 官方模型页](https://huggingface.co/mace-foundations/mace-mh-1)
单独获取，并遵守其上游许可证。

> **English:** AdsDrift uses MACE-MH-1 as a frozen, differentiable feature
> encoder during training. Features from the `oc20_usemppbe` head define the
> Drifting objective, while gradients with respect to generated coordinates
> are propagated back into the AdsDrift generator. MACE-MH-1 weights are not
> redistributed here, and generator-only inference does not run MACE.

本文描述 `test_18`：继承 `test_16`，将每个块的并行三分支聚合改为
“slab/吸附物自注意力残差 → 双向交叉注意力残差 → FFN”。交叉注意力
读取该层自注意力更新后的特征。初始化（包括全图几何嵌入及晶体条件）
保持不变，固定层特征更新而坐标不动；继续采用冻结 MACE 特征空间和
不隔离的联合 Drifting 坐标梯度（2026-09-12 恢复）。参数数量不变。

从新的干净晶体/slab 和吸附物直接生成 PDB（含真空层、吸附物库，不用 MACE/MLFF 或正样本 bank），使用 [generate/generate_pdb.py](generate/generate_pdb.py)；运行参数可通过该脚本的 `--help` 查看。

## 1. 工程目录与调用关系

```text
AdsDrift/
├── README.md                         # 统一的架构、数据与运行说明
├── config/                          # 三套固定体系训练配置
├── data/prepare_dataset.py           # 数据筛选、终点导出、训练索引
├── generate/                        # 新晶体＋吸附物→生成器→PDB，含真空和分子库
├── plot/plot_train.py                # 六面板训练仪表盘
├── model/
│   ├── __init__.py                   # Python 包声明与生成器导出
│   ├── model.py                      # 完整模型组合、训练/推理入口
│   ├── drifting/
│   │   ├── loss.py                   # 联合核、漂移场、损失和指标
│   │   ├── mace_features.py          # 冻结 MACE、固定配置与可微特征提取
│   │   ├── mace_pt/                  # 本地外部权重目录（Git 忽略）
│   │   └── distributed_objective.py  # 多卡生成样本联合计算漂移
│   ├── generator/
│   │   ├── generator.py               # 三分支局部等变生成器
│   │   └── core/                      # 合并后的 AdsDrift 等变核心算子
│   ├── condition/
│   │   ├── schema.py                  # 构造因子的严格数据协议
│   │   └── encoder.py                 # 周期原胞图和构造因子编码器
│   ├── initialize/
│   │   ├── generate_random_inputs.py # R0 采样、覆盖检查与绘图
│   │   ├── inputs_to_structures.py   # NPZ 与 ASE 结构转换
│   │   └── r0_resampling.py          # 按体系和 epoch 在线重新采样
│   └── utils/
│       ├── data_loader.py                       # 条件数据加载、模式采样、batch
│       ├── train.py                             # 单卡训练、日志和 checkpoint
│       ├── multi_gpu_training.py                # 多卡训练
│       ├── generate_structures.py               # checkpoint 一步生成与结构导出
│       ├── structure_validation.py              # 周期几何、结构去重和异常检查
│       └── screening_evaluation_metrics.py      # 筛选曲线与覆盖指标
```

`tests/` 包含条件置换不变性、构造因子敏感性、旋转等变性和反向传播测试。

绘制训练曲线（在 `AdsDrift/plot` 目录下运行）：

```bash
python plot_train.py /path/to/test_or_run_directory
```

读取指定目录的 `metrics.jsonl`，只生成或更新同目录的 `training_dashboard.png`。六个面板依次为原始漂移损失 `mean(V²)`、模式吸引指标、归一化优化损失、结构位移、梯度与学习率、训练速度。默认使用 51 条记录的滑动中位数，可通过 `--window` 调整。若目录已有 `training_metrics_summary.json` 中的 oracle 位移统计，则显示参考线；否则省略参考线。不加载 checkpoint，也不修改统计文件。

训练的数据流为：

```text
条件 c + 随机噪声 ──初始化──> R0 ──Gθ──> 生成坐标 R̂
                                          │
                                      冻结 MACE φ
                                          │
                                     生成特征 φ(R̂)
                                          │
同体系低能终点 ──预先提取──> 正样本特征 ────┤
                                          ↓
                               联合距离与 Drifting 漂移场
                                          ↓
                         stop-gradient 特征目标与联合特征损失
                                          ↓
              一次 ∇R L → 平移/旋转/内部投影加权 → 更新生成器 θ
```

`model.py` 中的 `AdsorptionDriftingObjective` 组合生成器、冻结特征编码器和漂移损失。
`build_model(generator_config, drifting_config, coordinate_gradient_balancing_config)`
接收生成器、漂移与梯度平衡参数，并自动加载包内固定的 MACE 编码器。

共享联合核仍同时使用四个特征分支来计算样本权重与模式吸引场：

$$
\{h_{\rm ads}^{(0)},h_{\rm ads}^{(1)},
  h_{\rm surf}^{(0)},h_{\rm surf}^{(1)}\}
\longrightarrow (L_{\rm ads},L_{\rm surf}).
$$

随后只替换损失对坐标的 VJP：

$$
\frac{\partial L}{\partial R_i}=
\begin{cases}
\partial L_{\rm ads}/\partial R_i,&i\in\text{adsorbate},\\
\partial L_{\rm surf}/\partial R_i,&i\in\text{movable surface},\\
0,&i\in\text{fixed/padding}.
\end{cases}
$$

实现只执行一次 MACE 前向；两个分支分别反向得到坐标 VJP，再通过数值等于
原始 loss 的直通代理标量传给生成器。记录中额外保存两个分支 loss、两条被
屏蔽交叉梯度的范数与比例。

## 2. 条件、学习变量和张量形状

test_18 沿用 test_6，把条件写成构造因子的乘积，而不是不可拆分的“体系编号”：

$$
c=(S_{\rm prim},hkl,s_{\rm term},b,M,n_{\rm layer},v_{\rm vac},\varepsilon,A),
$$

其中 $S_{\rm prim}=(Z_{\rm prim},F_{\rm prim},L_{\rm prim})$ 是标准原胞元素、分数坐标和晶格；$hkl$ 是相对于常规标准胞 $L_{\rm conv}$ 定义的约化 Miller 指数；$s_{\rm term}$ 与 $b$ 分别表示终止位移和上下表面；$M$ 是从原胞到取向、扩展后体相晶格的三维非奇异整数矩阵；其后依次为 slab 层数、真空、应变和吸附物身份。`condition_factors.json` 同时保存 $L_{\rm prim}$ 与 $L_{\rm conv}$，避免把常规胞误称为原胞，也避免在错误的倒易基底上解释 Miller 指数。这里“独立”表示这些变量在接口中可分别控制，并不假设它们在数据统计上相互独立。

完整 slab 的 $(Z,t,L,R_0)$ 仍作为局部物理图输入，但不再承担全部条件表示：

$$
c\xrightarrow{\text{surface construction}}S_{\rm slab},\qquad
(S_{\rm slab},R_0,c)\xrightarrow{G_\theta}\widehat R.
$$

模型当前仍只生成位置、朝向和表面弛豫，不生成元素、原胞、Miller 指数或超胞矩阵；这些是推理时给定、将来可组合扫描的条件。

| 输入 | 形状 | 含义 |
| --- | --- | --- |
| `atomic_numbers` | [B,N] | 各原子的原子序数 |
| `roles` | [B,N] | padding=0、固定层=1、可移动层=2、吸附物=3 |
| `atom_mask` | [B,N] | 有效原子位置 |
| `cell` | [B,3,3] | 完整晶胞向量，单位 Å |
| `r0_positions` | [B,G,N,3] | 每个条件的 G 个随机初始结构，单位 Å |

每个体系目录还必须有 `condition_factors.json`，显式保存原胞、晶面、终止、超胞矩阵、层数、真空、应变和吸附物元素序列。字段示例见 `data/condition_factors.example.json`。旧数据缺少该文件会立即报错；test_18 不从最终 slab 反推这些变量，因为这种反推对重构、合金和不同终止通常不唯一。

OC20 的 tag 0/1/2 分别映射为内部 role 1/2/3。当前训练加载器规定 B=1，G 默认100，可在 YAML 中设置。

生成器内部将 B 和 G 合并成 B×G 个独立结构。注意力发生在每个结构的原子之间，100个候选不会在生成器内部互相注意；它们在后面的漂移损失中共同参与分布比较。

## 3. 随机初始化 R0

初始化读取参考轨迹的第0帧。参考表面的所有原子保持原位，吸附物整体平移和旋转，得到覆盖晶面的有效初始结构。

每个候选使用七维高斯提议噪声：

$$
z\sim\mathcal N(0,I_7),\qquad u_k=\Phi(z_k),\quad k=1,2,3.
$$

前三维经高斯累积分布函数变成均匀变量，用于晶面内分数坐标和吸附高度。默认采用等面积分层：G=100时覆盖10×10区域，每个区域接受一个随机位点；高度用100个分层区间，并随机分配给横向区域。高度范围默认1.2–3.0 Å，定义为吸附物最低原子减去表面最高原子的法向高度。

后四维归一化为单位四元数，产生均匀 SO(3) 旋转。初始化保持分子内部相对构象，仅随机化整体朝向；单原子吸附物不受旋转影响。目前没有额外采样分子扭转角。

采样还检查跨周期接触、真空间距和横向去重，默认最小横向间距为0.1 Å。经过分层和拒绝采样后，最终 R0 是受几何约束的分布，不是独立高斯原子坐标。

`r0_resampling.py` 根据体系 ID、基础随机种子和 epoch 生成确定性新种子，从保存的参考结构与采样元数据重建 R0。它校验参考文件哈希、原子顺序和晶胞。在线模式下每轮得到新的100个输入，便于学习对新初始化的映射。

## 4. 生成器 Gθ

完整逐步公式、局部边定义和输出约束见 [生成器详细 README](model/generator/README.md)。

### 4.1 SO(3) 原子表示

每个节点保存从 $l=0$ 到 $l=l_{\max}$ 的不可约表示：

$$
X_i\in\mathbb R^{(l_{\max}+1)^2\times C},\qquad
X_{i,0,:}=E_Z(Z_i)+E_{\mathrm{role}}(t_i)+B_{t_i}(c).
$$

初始化还叠加由全原子统一周期图的距离径向基和 Wigner-D 旋转生成的几何边嵌入，包括其标量分量。此初始化沿用旧版，因此第一层前仍含有界面几何信息，本版不隔离初始化。网络不嵌入绝对笛卡尔坐标，因此整体平移不改变预测位移。

### 4.2 三分支局部等变块

每层包含三套不共享参数的 EquiformerV3 注意力：

```text
完整 slab ──K/V──> 完整 slab 查询         （slab self-attention）
吸附物     ──K/V──> 吸附物查询             （adsorbate self-attention）
完整 slab  ──K/V──> 吸附物查询 ─┐
吸附物     ──K/V──> 完整 slab 查询 ─┴──   （共享的 interface cross-attention）
```

固定层同时作为 query、key/value，所有有效原子都参与 attention 和 FFN 特征残差更新，padding 不更新。更新后的固定层特征继续传入下一层；其坐标仍由输出掩码固定。四层均使用平滑半径截断、SO(2) 图注意力、Wigner-D 旋转及 SwiGLU-$S^2$ 前馈层。默认 $C=128$、$l_{\max}=3$、$m_{\max}=2$；计算只发生在半径截断后的周期边上，不再进行全局 $N^2$ 注意力，也暂不加入全局模式路由。每层只在 $l=0$ 标量通道施加按 fixed/surface/adsorbate 区分的 FiLM 条件调制，因此不改变高阶张量的旋转规则。

同一层内先并行计算 SS 和 AA，再完成其残差更新，之后重新归一化并施加已有 FiLM，供 SA 读取。最后更新 SA 残差，再计算 FFN。两次注意力归一化共享同一组已有参数；SS、AA、SA 各自的残差系数仍为 $1/\sqrt2$。因此参数数量、初始化权重、邻居图和单次坐标读出均不变，只有特征更新顺序改变。

### 4.3 三个几何输出头

| 原子部分 | 输出 | 约束 |
| --- | --- | --- |
| 固定层 | 原输入坐标 | 完全不移动 |
| 可移动表面层 | 每原子三维残差 | 从该原子的 $l=1$ 特征读取，尺度0.5 Å |
| 吸附物整体 | 一个三维中心位移 | 吸附物 $l=1$ 输出均值，尺度3 Å |
| 吸附物内部 | 每原子三维残差 | 减去残差均值，尺度1 Å |

对吸附物，输出可以写为：

$$
\hat R_i=\operatorname{wrap}_{ab}(\bar R_0+\Delta c)
 +(R_{0,i}-\bar R_0)+\Delta r_i,
\qquad\sum_{i\in\mathrm{ads}}\Delta r_i=0.
$$

只把分子中心包裹回晶面内的周期晶胞，保持分子连续，不对每个原子单独包裹。中心位移和内部位移分开，避免重复表达整体平移自由度。上述尺度是输出乘数，不是硬性位移上限。

三个输出头只在通道维做共享线性投影，不混合 $l=1$ 的三个空间分量，因此坐标残差随结构共同旋转。输出头零初始化：初始输出等于 R0 的周期等价表示。单原子吸附物的内部残差恒为零。生成器一次前向就产生最终候选坐标；训练完成后只生成结构时，不需要运行 MACE。

## 5. 冻结 MACE 特征空间

训练用 MACE-MH-1，head 为 `oc20_usemppbe`。读取两个 interaction 的512通道特征。底层网络参数冻结并保持 eval 模式；生成坐标到特征的计算图保留，使梯度可以传回生成器。

| 特征分支 | 单结构形状 | 读取方式 |
| --- | --- | --- |
| `scalar_ads` | [4,512] | 两层吸附物标量的均值和标准差 |
| `message_l1_ads` | [4,512] | 两层 l=1 消息模长的均值和标准差 |
| `scalar_movable` | [2,M,512] | 两层可移动表面原子的标量，保留原子身份 |
| `message_l1_movable` | [2,M,512] | 两层可移动表面原子的 l=1 消息模长 |

M 为可移动表面原子数。l=1 处理使用三个分量的欧氏范数，当前没有保留其完整方向向量。吸附物通过统计池化形成固定尺寸表示，可移动层按体系内固定原子顺序保留特征。

损失最多使用4+4+2+2=12组特征。单原子吸附物的标准差组退化，当前会按特征尺度阈值跳过；此前单 O 体系实际使用8组。固定层虽没有独立的损失读出，其原子仍在 MACE 图和生成器中提供环境信息。

正样本特征由数据制备阶段预先保存；生成样本特征每步在线计算。所有正样本分支必须对应同一个终点索引，避免混合不同构型的吸附物与表面特征。

## 6. Drifting 损失

每个体系独立计算：生成样本为负样本集合，正样本来自相同体系的低能终点。不同催化体系之间不做正负样本匹配。

### 6.1 联合距离

第 g 组特征按本批生成/正样本的平均距离估计尺度 s_g，再除以通道数平方根，使各组对联合距离具有可比权重：

$$
\tilde f_g=f_g/s_g,\qquad
j=\frac{1}{\sqrt Q}\operatorname{concat}_g
       \left(\tilde f_g/\sqrt{d_g}\right).
$$

Q 为有效特征组数，d_g 为该组展平后的维度。尺度和联合核权重停止梯度；生成特征本身仍参与反向传播。多个可移动原子带来的维度增加不会直接按通道数放大其权重。

### 6.2 吸引与排斥

对每个温度 τ，用联合特征的欧氏距离构建 logits：

$$
\ell_{ij}=-\|j_i-j_j\|_2/\tau,\qquad
A_{ij}=\sqrt{\operatorname{softmax}_{\rm row}(\ell)_{ij}
                  \operatorname{softmax}_{\rm col}(\ell)_{ij}}.
$$

列集合包含正样本和生成样本，生成样本对自身的负样本权重设为零。权重经过正负质量交叉归一化，得到共享的 W⁺、W⁻；每组使用同一套权重计算：

$$
V_g^{(\tau)}=W^+\tilde F_g^+-W^-\tilde F_g^-.
$$

正样本提供吸引，其他生成样本提供排斥。排斥可以帮助维持多样性，但不能保证有限训练中每个模式都有相同生成频率。

### 6.3 实际优化目标与日志

各温度漂移场先按自身 RMS 归一化，再相加构造停止梯度的目标：

$$
T_g=\operatorname{sg}\!\left(\tilde F_g^-
 +\sum_\tau\frac{V_g^{(\tau)}}{\operatorname{RMS}(V_g^{(\tau)})}\right),
\qquad
L=\sum_g\operatorname{mean}\!\left[(\tilde F_g^--T_g)^2\right].
$$

用于观察收敛的 `raw_drift_loss` 是漂移 RMS 归一化之前的
\(\operatorname{mean}(V^2)\)，再对特征组和温度取平均；这里的 V 已在经过特征尺度归一化的空间中计算。
`loss` 是上述归一化后的优化目标，两条曲线的量级和走势不必一致。

三档无量纲核温度直接由 YAML 的 `drifting.temperatures: [0.02, 0.05, 0.2]` 设置，采用归档报告记录的数据集标定结果；训练不再读取 `temperature_calibration.json`，也不再使用 `temperatures: auto`。整个目标不包含直接的坐标配对监督、能量 MAE 或力 MAE。

### 6.4 吸附物坐标梯度投影与加权

联合特征损失仍包含吸附物与表面的全部交叉作用，但先只对隔离坐标叶节点
求一次 MACE 坐标 VJP：

$$
g=\partial_R(L_{\rm ads}+L_{\rm surf}).
$$

吸附物梯度正交分解为平移、无穷小刚体旋转和内部形变：

$$
g_{\rm ads}=g_t+g_r+g_i.
$$

默认YAML梯度能量比例为平移20%、旋转20%、内部形变60%。每个非退化
分量按自身范数重标度，组合后保持该候选原始MACE吸附物梯度总能量；
表面梯度不变。
加权场停止梯度并通过线性代理损失送入生成器，因此只需要一次 MACE 一阶
坐标梯度，不计算 Hessian。启动记录使用
`coordinate_gradient_routing=mace_projected_balanced_first_order`，日志
`coordinate_routing_enabled=1`，并保存加权前后的分量比例与缩放倍数。

## 7. 训练数据与采样单位

完整预处理数据未包含在本仓库中。原开发环境使用的数据布局为：

```text
dataset/drift_oc20dense_0p50eV_v1
```

该版本从973个审计体系选出905个至少有两个低能模式的体系，包含15,207个模式、31,147个终点及90,500个预生成 R0。筛选窗口为各体系接受终点中观测最低能量以上0.50 eV，模式定义为审计结果中的 `symmetry_rmsd_0.1A`。

每个 batch 仅使用一个体系，默认生成100个候选。每个 epoch 随机遍历全部已配置体系一次，因此完整905体系配置为905个 batch/epoch，单体系配置为1个 batch/epoch。

默认每个模式抽一个终点，模式内有多个成员时随机选择成员；所有四个特征分支使用相同的抽样索引。
这让训练中的正样本模式具有均等采样机会，但生成频率不是热力学占据概率。当前目标是覆盖各低能模式，不要求输出数量严格均分。

在线重采样需在单卡配置中显式开启：

```yaml
data:
  r0_per_system: 100
  positives_per_mode: 1
  max_positive_modes: null
  online_r0_resampling:
    enabled: true
    base_seed: 20260907
    max_attempts_per_sample: 100
    max_bank_retries: 4
```

三份固定训练配置均已开启这段设置；若关闭，则使用数据集缓存的 R0。当前多卡运行器沿用固定 R0 的分片实现，在线重采样路径在单卡运行器中。

## 8. 数据制备入口

在原有结构/特征 bank 完成后，还需为每个体系写入可信构造因子。例如：

```bash
python -m AdsDrift.data.prepare_condition_factors \
  --system-directory /path/to/systems/0_1190_0 \
  --primitive-structure /path/to/bulk-primitive.cif \
  --conventional-structure /path/to/bulk-conventional.cif \
  --miller 1 1 1 --termination-shift 0.25 --top \
  --supercell-matrix 3 0 0 0 3 0 0 0 1 \
  --slab-layers 4 --vacuum 15
```

原胞、终止位移和超胞矩阵必须来自数据生成记录或可信来源，不能用最终 slab 的外观猜测。脚本会检查原胞周期性、Miller 指数、整数矩阵和吸附物原子顺序。

`data/prepare_dataset.py` 已合并原来的三个数据脚本。其子命令可以独立执行：

| 子命令 | 作用 | 主要输入 |
| --- | --- | --- |
| `index` | 生成体系索引、种子和计数 | audit-root、trajectory-root、tag-mapping |
| `trajectories` | 每个模式复制一条完整代表轨迹及干净表面 | audit-root、trajectory-root、mapping-root |
| `positives` | 单体系每个模式导出一个代表终帧 | system-dir、audit-json、tag-mapping |

公共参数 `--window-ev` 默认0.5，`--cluster-key` 默认上述对称性聚类键。接受旧拼写 `--window` 与 `--window-eV`。
`index` 可用 `--expected-systems 905` 检查全量计数；`trajectories` 可用 `--expected-samples 15274` 检查代表轨迹总数。对子集省略这些检查。

`index` 仅建立索引，manifest 保持 `complete: false`；它不生成 R0 或提取 MACE 特征。代表终点导出也不展开所有模式成员。完整训练 bank 仍需初始化及特征制备步骤。本仓库包含 `model/initialize/` 中的初始化工具；历史实验目录和未整理的特征导出脚本没有纳入本次开源快照。

```bash
python model/AdsDrift/data/prepare_dataset.py index \
  --audit-root analysis/oc20_dense_positive_audit_20260905/full_results_0p50eV \
  --trajectory-root data/oc20_dense_extracted/trajs \
  --tag-mapping data/oc20_dense_mappings_extracted/oc20dense_tags.pkl \
  --output-root dataset/drift_oc20dense_0p50eV_v2 \
  --minimum-modes 2 --random-count 100 --expected-systems 905

python model/AdsDrift/data/prepare_dataset.py positives \
  --system-dir test_1/model/0_1190_0 \
  --audit-json analysis/oc20_dense_positive_audit_20260905/local_results/systems/0_1190_0.json \
  --tag-mapping analysis/oc20_dense_positive_audit_20260905/mappings/oc20dense_tags.pkl \
  --output-dir model/AdsDrift/data/positive_0p50eV
```

索引和终点导出拒绝覆盖已有结果。原始轨迹导出支持相同输入配置下续跑；冲突配置及同尺寸但内容不同的文件会报错。

## 9. 运行方式

顶层 Python 包名为 `AdsDrift`，内部导入统一使用 `from AdsDrift... import ...`。

克隆仓库后，先安装依赖，并把仓库的父目录加入 Python 搜索路径：

```bash
git clone https://github.com/Prome-theus1/AdsDrift.git
cd AdsDrift
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$(dirname "$PWD")${PYTHONPATH:+:$PYTHONPATH}"
```

之后可在仓库父目录使用 `python -m AdsDrift...`；配置参数需指向实际配置文件。模型文件和数据文件的相对路径以运行时工作目录解析。GPU 训练通常需要先按目标 CUDA 环境安装匹配的 PyTorch，再安装其余依赖。

```bash
# 统一入口
python -m AdsDrift.model.model --help

# 单卡训练（需要数据集与冻结 MACE 权重）
python -m AdsDrift.model.model train \
  --config AdsDrift/config/train_single_0_1190_0.yaml

# 多卡训练
torchrun --nproc_per_node=2 -m AdsDrift.model.model train-distributed \
  --config AdsDrift/config/train_single_0_1190_0.yaml

# 已有条件 bank 的一步生成
python -m AdsDrift.model.model sample \
  --checkpoint /path/to/epoch_5000.pt \
  --system-directory dataset/drift_oc20dense_0p50eV_v1/systems/0_1190_0 \
  --output-directory model/AdsDrift/runs/inference_0_1190_0

# 默认本地示例的随机初始化
python -m AdsDrift.model.initialize.generate_random_inputs --num-samples 100

# 结构转换的参数帮助
python -m AdsDrift.model.initialize.inputs_to_structures --help
```

`model/model.py` 也支持直接文件执行。单卡训练、分布式训练和采样的全部参数由对应子命令的 `--help` 给出。
当前 `sample` 命令复用完整条件 bank 加载器，因此其输入目录仍要求正样本元数据与特征文件齐全，虽然生成器前向本身不使用这些正样本。

训练相对路径以运行时工作目录解析。MACE 权重不随仓库分发；默认兼容路径为 `model/drifting/mace_pt/macemh1model.pt`，也可设置 `ADSDRIFT_MACE_CHECKPOINT=/path/to/mace-mh-1.model`。head、SHA256、microbatch=8、activation checkpointing 和数值阈值统一由代码中的 `MACEFeatureConfig` 管理。checkpoint 会记录实际 MACE 配置以便追溯。

运行结果输出到
`training.run_directory` 指定的目录，默认 `model/AdsDrift/runs/production`；`--run-directory` 可覆盖该设置。不再使用 `paths.run_directory`。默认配置使用TF32、MACE microbatch=8和activation checkpointing。

本次整理保持生成器参数名称、张量形状与保存的 `generator` state dict 格式不变，已有 checkpoint 可由新路径的生成器加载；历史配置里的旧文件路径需要指向实际位置。

学习率预热使用 `training.warmup_epochs`，默认 10，设为 0 可关闭预热。内部按 `warmup_epochs × 每个 epoch 的优化步数` 换算；当前每个体系对应一次更新，所以优化步数为体系数量。单卡与多卡均采用此规则，多卡不再除以卡数。预热之后仍使用原有余弦衰减；若总训练轮数不超过预热轮数，则全程处于预热阶段。旧配置的 `warmup_steps` 应替换为 `warmup_epochs`。

## 10. 验证与当前边界

历史实验依据和部分复现材料不在本次开源快照中。仓库保留当前模型实现、配置与数据制备入口；新的维护者应优先补充可公开的小型测试数据和端到端复现流程。

输出分布接近参考终点，并不意味着生成器已给出严格驻点或数学上的全局最低能结构。验证还需要关注可移动表面误差、碰撞、残余力、能量及未见体系表现。能量和力的独立验证属于后续评估；当前训练阶段仅优化上述冻结特征空间的分布目标。

## 许可证

AdsDrift 原创代码按 [Apache License 2.0](LICENSE) 许可。仓库内由 EquiformerV3、FAIR Chemistry 和 e3nn 改编的部分仍保留其 MIT 许可与署名，详见 [NOTICE](NOTICE) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。外部模型权重和数据不受本仓库许可证覆盖。
