# SpatialAST 开发文档

> **硬件约束**：4 × NVIDIA H100 80GB (NVLink)  
> **预计总开发周期**：8–10 周  
> **参考架构**：architecture.md v2.0

---

## 目录

1. [环境与基础设施](#1-环境与基础设施)
2. [代码库结构与模块划分](#2-代码库结构与模块划分)
3. [开发阶段规划](#3-开发阶段规划)
4. [Phase 0：环境搭建与依赖安装](#4-phase-0环境搭建与依赖安装)
5. [Phase 1：核心数据层](#5-phase-1核心数据层)
6. [Phase 2：数据管线](#6-phase-2数据管线)
7. [Phase 3：模型实现](#7-phase-3模型实现)
8. [Phase 4：扩散引擎](#8-phase-4扩散引擎)
9. [Phase 5：训练系统](#9-phase-5训练系统)
10. [Phase 6：推理与评估](#10-phase-6推理与评估)
11. [Phase 7：消融实验与调优](#11-phase-7消融实验与调优)
12. [4×H100 显存与计算预算分析](#12-4h100-显存与计算预算分析)
13. [分布式训练配置](#13-分布式训练配置)
14. [实验管理与复现](#14-实验管理与复现)
15. [测试策略](#15-测试策略)
16. [风险清单与缓解方案](#16-风险清单与缓解方案)

---

## 1. 环境与基础设施

### 1.1 硬件规格

```
┌──────────────────────────────────────────────────────────┐
│  计算节点                                                 │
│                                                          │
│  GPU:   4 × NVIDIA H100 80GB (SXM5, NVLink 4.0)         │
│         ├── 单卡显存: 80 GB HBM3                          │
│         ├── 总显存:  320 GB                               │
│         ├── 单卡 FP16/BF16 峰值: ~990 TFLOPS             │
│         └── NVLink 带宽: 900 GB/s (双向)                  │
│                                                          │
│  CPU:   建议 ≥64 核 (数据预处理并行)                       │
│  RAM:   建议 ≥512 GB (全量数据集预加载)                    │
│  存储:  ≥2 TB NVMe SSD                                   │
│         ├── 原始数据:   ~5 GB                             │
│         ├── 预处理缓存: ~20 GB                            │
│         └── Checkpoints: ~50 GB (每个 ~1.5 GB × 35个)    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 1.2 软件栈

| 层级 | 软件 | 版本要求 |
|------|------|---------|
| OS | Ubuntu 22.04 LTS | kernel ≥5.15 |
| 驱动 | NVIDIA Driver | ≥535 |
| CUDA | CUDA Toolkit | 12.1+ |
| Python | CPython | 3.10 / 3.11 |
| 深度学习 | PyTorch | 2.2+ (CUDA 12.1 build) |
| SSM | mamba-ssm | ≥1.2.0 |
| 注意力 | flash-attn | ≥2.5.0 |
| 分布式 | PyTorch FSDP 或 DeepSpeed | 内置 / 0.14+ |
| 数据 | HuggingFace Datasets | ≥2.16 |
| 配置 | Hydra + OmegaConf | 1.3+ |
| 实验追踪 | Weights & Biases | ≥0.16 |
| 测试 | pytest + hypothesis | 最新稳定版 |

---

## 2. 代码库结构与模块划分

```
spatial_ast/
│
├── configs/                        # --- 配置文件 ---
│   ├── model/
│   │   ├── base.yaml               #   d_model=768, 18 blocks
│   │   ├── small.yaml              #   d_model=512, 12 blocks (调试用)
│   │   └── large.yaml              #   d_model=1024, 24 blocks (可选扩展)
│   ├── train/
│   │   ├── default.yaml            #   4×H100, bs=128, 200k steps
│   │   └── debug.yaml              #   1 GPU, bs=4, 100 steps
│   ├── diffusion/
│   │   └── default.yaml            #   T=1000, 层次化破坏
│   └── eval/
│       └── default.yaml            #   评估采样配置
│
├── core/                           # --- 核心数据结构 (纯 Python, 无 GPU 依赖) ---
│   ├── __init__.py
│   ├── types.py                    #   NodeType, NodeSpec, NodeRegistry
│   ├── ast_node.py                 #   ASTNode 数据类
│   ├── tokenizer.py                #   词表定义 + 编码/解码
│   ├── serializer.py               #   AST ↔ Token 序列
│   ├── grammar.py                  #   CFG 规则 + 父子兼容矩阵
│   └── geometry.py                 #   GeometryDescriptor 提取
│
├── data/                           # --- 数据管线 ---
│   ├── __init__.py
│   ├── deepcad_parser.py           #   DeepCAD JSON → 内部表示
│   ├── decompiler.py               #   DeepCAD 指令 → AST
│   ├── meta_annotator.py           #   AST → TokenMeta (depth/parent/sibling/geom)
│   ├── dataset.py                  #   SpatialASTDataset (torch Dataset)
│   ├── augmentation.py             #   坐标平移/缩放/镜像/旋转
│   └── statistics.py               #   数据集分布统计 + 可视化
│
├── diffusion/                      # --- 层次化破坏扩散 ---
│   ├── __init__.py
│   ├── schedule.py                 #   corruption_probability(d, t)
│   ├── corruption.py               #   SubtreeCorruptor + 批量破坏
│   └── loss.py                     #   compute_loss (CE + L2 + 辅助损失)
│
├── model/                          # --- 神经网络模块 ---
│   ├── __init__.py
│   ├── embedding.py                #   SpatialASTEmbedding (7 路嵌入)
│   ├── attention.py                #   MHA + CrossAttention (FlashAttn)
│   ├── gc_mamba.py                 #   GeometryConditionedMamba (双向)
│   ├── dwconv.py                   #   DepthwiseConv1d
│   ├── glf.py                      #   GlobalLocalGatedFusion
│   ├── adaln.py                    #   AdaLayerNorm
│   ├── ffn.py                      #   SwiGLU FFN
│   ├── hybrid_block.py             #   HybridBlock (组装以上子层)
│   ├── condition.py                #   ConditionEncoder (T5 + ViT)
│   ├── timestep.py                 #   TimestepEncoder (sinusoidal + MLP)
│   └── denoiser.py                 #   SpatialASTDenoiser (完整模型)
│
├── compiler/                       # --- AST ↔ DeepCAD 编译 ---
│   ├── __init__.py
│   ├── ir.py                       #   IRInstruction 中间表示
│   ├── emitter.py                  #   AST → IR
│   ├── backend.py                  #   IR → DeepCAD 指令
│   ├── quantize.py                 #   Q8 量化/反量化
│   └── validator.py                #   编译前语义检查
│
├── decoder/                        # --- 约束解码 ---
│   ├── __init__.py
│   ├── grammar_mask.py             #   GrammarMask (合法 token 过滤)
│   ├── bracket_balancer.py         #   括号平衡修复
│   ├── geometry_checker.py         #   Loop 闭合检查/修复
│   └── pipeline.py                 #   ConstraintDecoderPipeline (串联)
│
├── training/                       # --- 训练系统 ---
│   ├── __init__.py
│   ├── trainer.py                  #   Trainer (训练循环)
│   ├── distributed.py              #   FSDP/DDP 初始化
│   ├── evaluator.py                #   Evaluator (指标计算)
│   ├── callbacks.py                #   CheckpointCallback, WandbLogger, EarlyStopping
│   └── profiler.py                 #   GPU 显存/计算 profiling
│
├── inference/                      # --- 推理 ---
│   ├── __init__.py
│   ├── sampler.py                  #   SpatialASTSampler (迭代去噪)
│   ├── edit.py                     #   子树级局部编辑 API
│   └── export.py                   #   ONNX / TorchScript 导出
│
├── scripts/                        # --- 入口脚本 ---
│   ├── preprocess.py               #   批量 DeepCAD → AST 转换
│   ├── train.py                    #   torchrun 训练入口
│   ├── eval.py                     #   评估入口
│   ├── generate.py                 #   生成入口
│   ├── profile_memory.py           #   显存 profiling 脚本
│   └── visualize_ast.py            #   AST 树可视化
│
├── tests/                          # --- 测试 ---
│   ├── unit/
│   │   ├── test_types.py
│   │   ├── test_serializer.py
│   │   ├── test_tokenizer.py
│   │   ├── test_grammar.py
|   |   ├── other unit tests...
│   │   └── test_quantize.py
│   ├── integration/
│   │   ├── test_decompile_compile.py    # 往返一致性
│   │   ├── test_corruption.py
│   │   ├── test_model_forward.py
│   │   └── test_constraint_decoder.py
│   └── smoke/
│       ├── test_overfit_one.py           # 单样本过拟合
│       └── test_distributed.py           # 多 GPU 冒烟测试
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

### 2.1 模块依赖图 (开发顺序)

```
Layer 0 (无依赖):  core/types.py, core/ast_node.py, core/tokenizer.py
                         │
Layer 1:           core/serializer.py, core/grammar.py, core/geometry.py
                         │
Layer 2:           compiler/*, data/decompiler.py, data/meta_annotator.py
                         │
Layer 3:           data/dataset.py, data/augmentation.py
                         │
Layer 4:           model/* (所有子模块)
                         │
Layer 5:           diffusion/*, decoder/*
                         │
Layer 6:           training/*, inference/*
                         │
Layer 7:           scripts/* (入口点)
```

---

## 3. 开发阶段规划

```
Week    1     2     3     4     5     6     7     8     9     10
       ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
P0 ██                                                          环境搭建
P1 ████████                                                    核心数据层
P2      ██████████                                             数据管线
P3            ████████████                                     模型实现
P4                  ████████                                   扩散引擎
P5                        ██████████████                       训练系统
P6                                    ████████                 推理/评估
P7                                          ████████████       消融/调优
```

| 阶段 | 周期 | 产出 | 通过标准 |
|------|------|------|---------|
| **P0** | 第 1 周前 3 天 | 环境就绪, 依赖安装, 单卡/多卡冒烟测试通过 | `nvidia-smi` 4卡可见, `mamba-ssm` import 成功, FlashAttn 编译通过 |
| **P1** | 第 1–2 周 | `core/` 全部模块 + 单元测试 | 序列化往返无损, 语法校验 100% 覆盖 |
| **P2** | 第 2–3 周 | `data/` + `compiler/` 全部模块 | 170k AST 数据集生成完毕, 往返编译一致率 >99% |
| **P3** | 第 3–5 周 | `model/` 全部子模块 + 单样本过拟合测试 | 前向传播形状正确, 单样本 loss→0, 显存 ≤30 GB/卡 |
| **P4** | 第 4–5 周 | `diffusion/` + `decoder/` | 层次化破坏可视化正确, 约束解码输出 100% 合法 AST |
| **P5** | 第 5–7 周 | 完整训练管线, 4×H100 分布式训练 | 200k steps 内 loss 收敛, 编译成功率 >80% |
| **P6** | 第 7–8 周 | 推理管线 + 全量评估 | CD/MMD/COV/JSD 达到可报告水平 |
| **P7** | 第 8–10 周 | 消融实验表 + 论文数据 | 完成所有消融对比 |

---

## 4. Phase 0：环境搭建与依赖安装

### 4.1 Conda 环境

```bash
conda create -n spatial_ast python=3.11 -y
conda activate spatial_ast

# PyTorch (CUDA 12.1)
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121

# Mamba SSM (需要 CUDA 编译)
pip install mamba-ssm>=1.2.0

# FlashAttention (需要 CUDA 编译, H100 支持 sm_90)
pip install flash-attn>=2.5.0 --no-build-isolation

# 其余依赖
pip install transformers>=4.38.0 datasets>=2.16.0 \
            hydra-core>=1.3 omegaconf>=2.3 \
            wandb>=0.16.0 \
            einops>=0.7.0 timm>=0.9.0 \
            pytest>=8.0 hypothesis>=6.90 \
            rich>=13.0 tqdm>=4.66
```

### 4.2 多卡冒烟测试

```bash
# 确认 4 卡可见
python -c "import torch; print(torch.cuda.device_count())"  # 应输出 4

# 确认 NCCL 通信
torchrun --nproc_per_node=4 -c "
import torch, torch.distributed as dist
dist.init_process_group('nccl')
t = torch.ones(1).cuda()
dist.all_reduce(t)
print(f'Rank {dist.get_rank()}: {t.item()}')
"

# 确认 mamba-ssm 可用
python -c "from mamba_ssm import Mamba; print('Mamba OK')"

# 确认 FlashAttention 可用
python -c "from flash_attn import flash_attn_func; print('FlashAttn OK')"
```

### 4.3 项目初始化

```bash
mkdir -p spatial_ast/{configs/{model,train,diffusion,eval},core,data,diffusion,model,compiler,decoder,training,inference,scripts,tests/{unit,integration,smoke}}

# 创建所有 __init__.py
find spatial_ast -type d -exec touch {}/__init__.py \;

# 初始化 git
git init
echo "__pycache__/\n*.pyc\n.eggs/\nwandb/\ncheckpoints/\ndata/processed/\n*.ckpt" > .gitignore
```

---

## 5. Phase 1：核心数据层

### 5.1 开发顺序

```
core/types.py ──▶ core/ast_node.py ──▶ core/tokenizer.py
                                              │
                              core/grammar.py ◀┘
                                              │
                           core/serializer.py ◀┘
                                              │
                            core/geometry.py  ◀┘
```

### 5.2 各文件职责与验收标准

| 文件 | 职责 | 行数估计 | 验收标准 |
|------|------|---------|---------|
| `types.py` | NodeType 枚举, NodeSpec, NodeRegistry, ChildSlot, ParamDef | ~150 | 注册全部 14+3 种节点类型, 无重复 tag |
| `ast_node.py` | ASTNode 不可变数据类, 树遍历工具 (DFS/BFS/depth/span) | ~120 | frozen=True, 深拷贝安全, 子树切片正确 |
| `tokenizer.py` | 词表常量 (304 tokens), encode_param/decode_param | ~100 | Q8 编码/解码往返无损, 枚举映射完整 |
| `grammar.py` | PARENT_CHILD_MATRIX, CHILDREN_CARDINALITY, validate_ast() | ~180 | 10 个手写合法 AST 通过, 10 个非法 AST 拒绝 |
| `serializer.py` | serialize(AST→tokens), deserialize(tokens→AST) | ~200 | 往返一致 `deserialize(serialize(ast)) == ast` |
| `geometry.py` | extract_geometry_descriptors(ast) → List[GeometryDescriptor] | ~100 | Line 曲率=0, Arc 曲率=1/R, 尺度归一化到 [0,1] |

### 5.3 测试文件

```python
# tests/unit/test_serializer.py
class TestRoundtrip:
    """对 20 个手工构造的 AST (从简单矩形到多 Solid 布尔运算) 验证序列化往返。"""

    @given(st.integers(1, 255), st.integers(1, 255))
    def test_coord_roundtrip(self, x, y):
        coord = ASTNode(NodeType.CRD, depth=5, children=(), params={"x": x, "y": y}, ...)
        tokens = serialize(coord)
        recovered, _ = deserialize(tokens)
        assert recovered.params["x"] == x
        assert recovered.params["y"] == y

    def test_rectangle_solid(self):
        ast = build_rectangle_solid(w=128, h=128, depth=64)
        tokens = serializer.serialize(ast)
        recovered = serializer.deserialize(tokens)
        assert ast_equal(ast, recovered)
```

---

## 6. Phase 2：数据管线

### 6.1 DeepCAD 数据预处理流程

```
DeepCAD 原始数据 (178k JSON)
        │
        ▼
┌───────────────────────────┐
│ scripts/preprocess.py     │  ← 多进程, 64 workers
│                           │
│ for each json:            │
│   1. deepcad_parser.parse │     JSON → 内部命令列表
│   2. decompiler.decompile │     命令列表 → AST 树
│   3. validator.validate   │     语法 + 基数 + 范围检查
│   4. serializer.serialize │     AST → token 序列
│   5. meta_annotator.annotate │  生成 depth/type/role/parent/sibling/geom
│   6. 保存为 Arrow 格式    │
│                           │
│ 过滤:                     │
│   - validate 失败 → 丢弃  │
│   - 编译往返不一致 → 丢弃 │
│   (不再丢弃长序列;         │
│    序列按自然长度存储,      │
│    DataLoader 动态 padding) │
│                           │
│ 产出:                     │
│   data/processed/          │
│   ├── train.arrow  (153k)  │
│   ├── val.arrow    (8.5k)  │
│   └── test.arrow   (8.5k)  │
└───────────────────────────┘
```

### 6.2 预处理性能预估

| 步骤 | 单样本耗时 | 总耗时 (178k, 64 workers) |
|------|-----------|-------------------------|
| JSON 解析 | ~0.5 ms | ~1.4 s |
| 反编译 | ~1 ms | ~2.8 s |
| 校验 | ~0.2 ms | ~0.6 s |
| 序列化 + 元数据标注 | ~2 ms | ~5.6 s |
| Arrow 序列化 | ~0.5 ms | ~1.4 s |
| **总计** | ~4.2 ms | **~12 s** (含调度开销约 2 分钟) |

### 6.3 数据增强策略

```python
# data/augmentation.py
class ASTAugmentor:
    transforms = [
        CoordinateShift(max_shift=10),       # 整体坐标平移 ±10 Q8 单位
        CoordinateScale(range=(0.8, 1.2)),   # 整体缩放 80%-120%
        CoordinateMirror(axes=["x", "y"]),   # X/Y 轴镜像
        EdgeOrderShuffle(),                  # Loop 内 Edge 循环移位 (保持闭合)
        SolidOrderShuffle(),                 # Program 内 Solid 顺序打乱
    ]
```

增强在 DataLoader 中按需在线执行，不预先存储。

---

## 7. Phase 3：模型实现

### 7.1 开发顺序 (自底向上)

```
Week 3                      Week 4                      Week 5
├── embedding.py            ├── hybrid_block.py         ├── denoiser.py
├── attention.py            ├── condition.py            └── 单样本过拟合测试
├── gc_mamba.py             └── timestep.py
├── dwconv.py
├── glf.py
├── adaln.py
└── ffn.py
```

### 7.2 各模块显存预算 (单卡, batch=32, seq≈512 (动态 padding), d_model=768, bf16)

> **注意**：序列长度不再有硬上限。预处理阶段保留所有序列（含 >512 的长序列），
> DataLoader 按 batch 内最长序列动态 padding。下表以 seq≈512 为典型估算；
> 对于偶尔出现的更长 batch (P99≈800+)，可通过 gradient accumulation 或
> bucket batching 控制显存峰值。

| 模块 | 参数显存 | 激活显存 | 合计 |
|------|---------|---------|------|
| SpatialASTEmbedding (7路) | 16 MB | 150 MB | 166 MB |
| 1 × HybridBlock | | | |
| &nbsp;&nbsp;├ DWConv | 0.01 MB | 3 MB | 3 MB |
| &nbsp;&nbsp;├ MHA (FlashAttn) | 14 MB | 48 MB | 62 MB |
| &nbsp;&nbsp;├ GC-Mamba | 18 MB | 64 MB | 82 MB |
| &nbsp;&nbsp;├ GLF | 2.4 MB | 6 MB | 8.4 MB |
| &nbsp;&nbsp;├ CrossAttn | 9 MB | 32 MB | 41 MB |
| &nbsp;&nbsp;├ FFN (SwiGLU) | 18 MB | 48 MB | 66 MB |
| &nbsp;&nbsp;└ 4×AdaLN | 6 MB | 12 MB | 18 MB |
| **18 × HybridBlock 合计** | **~1.2 GB** | **~3.8 GB** | **~5.0 GB** |
| ConditionEncoder (frozen) | 500 MB | 200 MB | 700 MB |
| TimestepEncoder | 2 MB | 1 MB | 3 MB |
| Output Projection | 0.5 MB | 3 MB | 3.5 MB |
| **模型总计** | **~1.7 GB** | **~4.2 GB** | **~5.9 GB** |
| **+ 梯度 (bf16)** | | | **+1.7 GB** |
| **+ 优化器状态 (AdamW fp32)** | | | **+6.8 GB** |
| **单卡总占用** | | | **~14.4 GB** |

> 单卡 80 GB 中实际占用约 14.4 GB，余量充足。可将 batch 提升到 32/卡 (全局 128)，甚至可以探索 48/卡。

### 7.3 单样本过拟合测试

```python
# tests/smoke/test_overfit_one.py
def test_overfit_single_sample():
    """在单张 GPU 上对 1 个样本训练 500 步, 验证 loss → 0。"""
    config = ModelConfig(**load_yaml("configs/model/small.yaml"))
    model = SpatialASTDenoiser(config).cuda()
    sample = load_single_sample("tests/fixtures/rectangle.json")
    optimizer = AdamW(model.parameters(), lr=3e-4)

    for step in range(500):
        loss = train_step(model, sample, optimizer, t=500)
        if step % 100 == 0:
            print(f"Step {step}: loss={loss:.6f}")

    assert loss < 0.01, f"单样本过拟合失败: loss={loss}"
```

---

## 8. Phase 4：扩散引擎

### 8.1 开发顺序

```
diffusion/schedule.py ──▶ diffusion/corruption.py ──▶ diffusion/loss.py
                                                            │
decoder/grammar_mask.py ──▶ decoder/bracket_balancer.py     │
                                     │                      │
decoder/geometry_checker.py ──▶ decoder/pipeline.py ◀───────┘
```

### 8.2 层次化破坏可视化验证

开发 `scripts/visualize_corruption.py`，对一个样本在 t=100, 300, 500, 700, 900 分别做破坏，可视化被破坏的子树（以颜色区分深度），确认：

- t=100 时仅 L5 节点 (坐标/参数) 被破坏
- t=500 时 L3-L5 被破坏, L0-L2 保持完整
- t=900 时 L0-L1 也开始被破坏

### 8.3 约束解码正确性测试

```python
# tests/integration/test_constraint_decoder.py
def test_random_logits_produce_valid_ast():
    """随机 logits 经过约束解码后必须产出合法 AST。"""
    for _ in range(100):
        random_logits = torch.randn(1, 512, 304)
        tokens = constraint_decode(random_logits)
        ast = deserialize(tokens)
        assert validate_ast(ast).is_valid
        assert check_brackets(tokens)
```

---

## 9. Phase 5：训练系统

### 9.1 训练启动命令

```bash
# 4×H100 分布式训练 (DDP)
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    scripts/train.py \
    --config-name=default \
    model=base \
    train=default \
    diffusion=default \
    train.batch_size=128 \
    train.precision=bf16
```

### 9.2 训练阶段划分

```
Steps       0      20k     50k     100k    150k    200k
            ├──────┼───────┼───────┼───────┼───────┤
Warmup      ██
                   
LR Schedule ████████████████████████████████████████    cosine decay
                   
Eval        ○      ○  ○    ○  ○    ○  ○    ○  ○    ○   每 10k steps
                   
Checkpoint  △      △  △    △  △    △  △    △  △    △   每 5k steps

关键里程碑:
  20k steps : loss 应降至初始值的 30% 以下
  50k steps : 编译成功率应 > 50%
 100k steps : 编译成功率应 > 75%, CD 开始有意义
 150k steps : 编译成功率应 > 85%
 200k steps : 最终模型, 编译成功率目标 > 90%
```

### 9.3 训练时间预估

| 参数 | 值 |
|------|-----|
| 每步前向+反向时间 (4×H100, bs=128, bf16) | ~0.35 s |
| 200k steps 总训练时间 | ~19.4 小时 |
| 含 eval 开销 (每 10k steps 采样 500 个) | ~22 小时 |
| 安全余量 (×1.3) | **~29 小时 (约 1.2 天)** |

> 比 architecture.md 中 5-7 天的初始估计显著快，因为 H100 的 bf16 性能和 FlashAttention 带来的加速。如果需要多轮超参搜索，5-7 天的总预算仍然合理（可跑 4-5 轮完整训练）。

### 9.4 监控与预警

```python
# training/callbacks.py
class TrainingMonitor:
    alerts = {
        "loss_spike":    lambda h: h[-1] > 2 * np.mean(h[-100:]),
        "loss_plateau":  lambda h: np.std(h[-500:]) < 1e-4 and len(h) > 1000,
        "grad_explosion": lambda g: g > 10.0,
        "gpu_oom":       "自动降低 batch_size 并重试",
        "nan_loss":      "回滚至上一个 checkpoint 并降低 lr",
    }
```

### 9.5 Checkpoint 策略

```
checkpoints/
├── latest.ckpt              # 每 100 steps 覆盖写入 (快速恢复)
├── step_005000.ckpt         # 每 5k steps 持久保存
├── step_010000.ckpt
├── ...
├── best_compile_rate.ckpt   # 编译成功率最高的模型
└── best_cd.ckpt             # Chamfer Distance 最低的模型

每个 ckpt 包含:
  - model_state_dict
  - optimizer_state_dict
  - scheduler_state_dict
  - scaler_state_dict
  - step, epoch, best_metrics
  - config (完整 OmegaConf)
  - rng_states (所有 GPU)
```

单个 checkpoint 大小估计：~1.5 GB（178M参数 × 4 bytes/fp32 + 优化器状态 × 2）

---

## 10. Phase 6：推理与评估

### 10.1 评估流程

```bash
# 从 best checkpoint 生成 5000 个样本
python scripts/generate.py \
    --checkpoint checkpoints/best_compile_rate.ckpt \
    --num_samples 5000 \
    --num_steps 50 \
    --output_dir results/generation/

# 运行全量评估
python scripts/eval.py \
    --generated results/generation/ \
    --test_set data/processed/test.arrow \
    --output results/metrics.json
```

### 10.2 评估指标计算资源

| 指标 | 计算方式 | 时间估计 (5000 样本) |
|------|---------|---------------------|
| 编译成功率 | CPU, 单进程 | ~30 s |
| 括号匹配率 / 类型兼容率 | CPU | ~5 s |
| Chamfer Distance | GPU 点云采样 (2048 点) | ~15 min (单卡) |
| MMD / COV / JSD | GPU 点云特征 | ~20 min |
| 生成多样性 (Unique/Novel) | CPU 哈希去重 | ~1 min |

### 10.3 推理性能

| 配置 | 吞吐量 |
|------|--------|
| 单卡 H100, bs=1, 50 steps | ~2 samples/s |
| 单卡 H100, bs=16, 50 steps | ~25 samples/s |
| 4 卡并行推理 | ~100 samples/s |
| 5000 样本总生成时间 | **~50 s (4卡)** |

---

## 11. Phase 7：消融实验与调优

### 11.1 消融实验矩阵

| 实验编号 | 消融内容 | 修改 | GPU 时间 |
|---------|---------|------|---------|
| A1 | 无 GC-Mamba (标准 Mamba) | `gc_mamba → BidirectionalMamba` | 29 h |
| A2 | 无 GLF (标量 gate) | `glf → scalar gate` | 29 h |
| A3 | 无 DWConv | 移除 `dwconv` | 29 h |
| A4 | 无层次化破坏 (均匀 mask) | `p(d,t) = p(t)` 不依赖深度 | 29 h |
| A5 | 无辅助损失 (仅 CE+L2) | `node_ce_weight=0, param_ce_weight=0` | 29 h |
| A6 | 纯 Transformer (无 Mamba) | 移除所有 Mamba 子层 | 29 h |
| A7 | 纯 Mamba (无 Attention) | 移除所有 MHA 子层 | 29 h |
| A8 | 12 层 (vs 默认 18 层) | `num_blocks=12` | 22 h |
| A9 | d_model=512 (vs 768) | 降低模型宽度 | 18 h |

**消融总 GPU 时间**：~235 h ≈ 2.4 天 (4×H100 并行可跑 2 个实验)

### 11.2 消融实验调度

```
4×H100 并行调度 (每次跑 1 个完整实验):

Day 1:  A1 (29h) ─────────────────────────────▶
Day 2:  A2 (29h) ─────────────────────────────▶
Day 3:  A3 (29h) ─────────────────────────────▶
Day 4:  A4 (29h) ─────────────────────────────▶
Day 5:  A5 (29h) ─────────────────────────────▶
Day 6:  A6 (29h) ─────────────────────────────▶
Day 7:  A7 (29h) + A8 (22h, 可用更少 GPU)  
Day 8:  A9 (18h) + 结果整理
```

如果 GPU 资源允许 2 路并行（每路 2 卡）：

```
Day 1-2:  A1 (2卡) ∥ A2 (2卡)
Day 2-3:  A3 (2卡) ∥ A4 (2卡)
Day 3-4:  A5 (2卡) ∥ A6 (2卡)
Day 4-5:  A7 (2卡) ∥ A8 (2卡)
Day 5:    A9 (2卡)
总计: ~5 天完成全部消融
```

### 11.3 超参搜索空间

| 超参 | 搜索范围 | 优先级 |
|------|---------|--------|
| 学习率 | {5e-5, 1e-4, 3e-4} | 高 |
| batch size | {64, 128, 256} | 中 |
| 扩散步数 T | {500, 1000} | 中 |
| 破坏 beta_scale | {0.05, 0.1, 0.2} | 高 |
| param_ce_weight | {1.0, 2.0, 4.0} | 中 |
| dwconv_kernel | {3, 5, 7} | 低 |
| num_blocks | {12, 18, 24} | 低 (消融覆盖) |

---

## 12. 4×H100 显存与计算预算分析

### 12.1 单卡显存分布 (batch=32, seq=512, bf16)

```
80 GB H100
├── 模型参数 (bf16)                    1.7 GB  ██
├── 梯度 (bf16)                        1.7 GB  ██
├── 优化器状态 (AdamW fp32, m+v)       6.8 GB  ████████
├── 激活 (FlashAttn 优化后)            4.2 GB  █████
├── ConditionEncoder frozen 参数       0.5 GB  █
├── CUDA Context + Buffers             1.5 GB  ██
├── 可用余量                          63.6 GB  (79.5%)
└── 总计                              16.4 GB
```

### 12.2 Batch Size 敏感性

| 每卡 Batch Size | 激活显存 | 总显存 | 全局 Batch Size |
|----------------|---------|--------|----------------|
| 16 | 2.1 GB | 14.3 GB | 64 |
| 32 | 4.2 GB | 16.4 GB | 128 |
| 48 | 6.3 GB | 18.5 GB | 192 |
| 64 | 8.4 GB | 20.6 GB | 256 |
| **96** | **12.6 GB** | **24.8 GB** | **384** |

> 显存余量充裕。推荐默认 batch=32/卡 (全局 128)，如果训练动态稳定可升至 48/卡。

### 12.3 梯度累积 (如果需要更大有效 batch)

```python
# 有效 batch = batch_per_gpu × num_gpus × accumulation_steps
# 256 = 32 × 4 × 2
accumulation_steps: int = 2
```

### 12.4 计算瓶颈分析

| 组件 | FLOPs / step (B=128, L=512) | 占比 |
|------|---------------------------|------|
| MHA (FlashAttn) | ~2.4 TFLOP | 28% |
| GC-Mamba (双向) | ~1.8 TFLOP | 21% |
| FFN (SwiGLU) | ~2.1 TFLOP | 24% |
| CrossAttn | ~0.8 TFLOP | 9% |
| GLF + DWConv + AdaLN | ~0.3 TFLOP | 4% |
| ConditionEncoder (frozen fwd) | ~0.6 TFLOP | 7% |
| 其余 (embedding, loss, ...) | ~0.6 TFLOP | 7% |
| **总计** | **~8.6 TFLOP / step** | |
| **4×H100 BF16 峰值** | **~3960 TFLOPS** | |
| **MFU (利用率)** | **~62%** | |
| **每步耗时** | **~0.35 s** | |

---

## 13. 分布式训练配置

### 13.1 推荐策略：DDP + Gradient Checkpointing

```python
# training/distributed.py
def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def wrap_model(model, local_rank):
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    return model
```

### 13.2 DDP vs FSDP 决策

| 因素 | DDP | FSDP |
|------|-----|------|
| 模型参数 178M, 单卡显存占用 ~16 GB | 每卡存完整副本, 16 GB × 4 = 64 GB 总显存消耗 | 参数分片, 4 GB/卡 参数显存 |
| 通信开销 | AllReduce 梯度 (~1.7 GB/卡) | AllGather 参数 + ReduceScatter 梯度 |
| 实现复杂度 | 低 | 中 (需要处理 checkpoint 格式) |
| **推荐** | **默认选择 DDP** | 仅当 batch>64/卡 或模型扩至 >500M 时切换 |

### 13.3 通信时间预估

```
NVLink 4.0 带宽: 900 GB/s (双向)
梯度大小: 178M × 2 bytes (bf16) = 356 MB
AllReduce 时间: 2 × 356 MB / 900 GB/s ≈ 0.8 ms
通信占比: 0.8 ms / 350 ms ≈ 0.2% (可忽略)
```

---

## 14. 实验管理与复现

### 14.1 Weights & Biases 项目结构

```
W&B Project: spatial-ast
├── Groups:
│   ├── main           # 正式训练 run
│   ├── ablation       # 消融实验
│   ├── hparam-search  # 超参搜索
│   └── debug          # 调试 run
│
├── Logged Metrics (每 100 steps):
│   ├── train/loss, train/ce_loss, train/reg_loss
│   ├── train/node_ce, train/param_ce
│   ├── train/lr, train/grad_norm
│   └── system/gpu_mem, system/gpu_util
│
├── Logged Metrics (每 10k steps):
│   ├── val/loss
│   ├── val/compile_rate
│   ├── val/bracket_match_rate
│   ├── val/type_compat_rate
│   └── val/chamfer_distance (每 50k steps)
│
└── Artifacts:
    ├── best_model.ckpt
    ├── config.yaml
    └── generated_samples/ (每 50k steps 抽样 20 个可视化)
```

### 14.2 复现要求

每个实验的 config 和 seed 通过 Hydra 自动保存到 `outputs/YYYY-MM-DD/HH-MM-SS/.hydra/`。

```python
# scripts/train.py
@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    # ... 保证确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 15. 测试策略

### 15.1 测试金字塔

```
                    ╱╲
                   ╱  ╲
                  ╱ E2E╲           1-2 个端到端测试
                 ╱      ╲          (全流程: 文本→AST→DeepCAD→3D)
                ╱────────╲
               ╱Integration╲      10-15 个集成测试
              ╱              ╲     (往返编译, 破坏-去噪, 约束解码)
             ╱────────────────╲
            ╱    Unit Tests    ╲   50+ 个单元测试
           ╱                    ╲   (序列化, 类型检查, 量化, 语法掩码)
          ╱──────────────────────╲
```

### 15.2 CI 触发规则

```yaml
# 每次 push:      运行 unit tests (~30s)
# 每次 PR:        运行 unit + integration (~3min)
# 每日 nightly:   运行 smoke tests (单样本过拟合, ~10min, 需 GPU)
# 每周 weekly:    运行 E2E tests (小规模训练 1k steps, ~30min, 需 GPU)
```

### 15.3 关键测试场景

| 测试 | 类型 | 验证内容 |
|------|------|---------|
| `test_serialize_deserialize_roundtrip` | Unit | 20 个 AST 序列化往返无损 |
| `test_q8_quantize_dequantize` | Unit | 0-255 全范围量化往返误差 <1e-6 |
| `test_grammar_validates_legal` | Unit | 10 个合法 AST 全部通过 |
| `test_grammar_rejects_illegal` | Unit | 10 个非法 AST 全部拒绝 |
| `test_decompile_compile_roundtrip` | Integration | 100 个 DeepCAD 样本往返一致 |
| `test_corruption_respects_depth` | Integration | t=100 时不破坏 L0-L2 节点 |
| `test_constraint_decode_always_valid` | Integration | 100 次随机 logits 解码全部产出合法 AST |
| `test_model_forward_shapes` | Integration | 前向传播所有中间张量形状正确 |
| `test_overfit_one_sample` | Smoke | 500 步内 loss < 0.01 |
| `test_4gpu_ddp_gradient_sync` | Smoke | 4 卡梯度同步一致 |

---

## 16. 风险清单与缓解方案

| # | 风险 | 概率 | 影响 | 缓解方案 |
|---|------|------|------|---------|
| R1 | mamba-ssm 自定义核注入接口不支持 `dt_scale`/`B_scale` | 中 | 高 | 备选方案: 将 GC 调制改为前置 MLP 对输入做 FiLM 调制，绕过 SSM 核修改 |
| R2 | 层次化破坏导致训练不稳定 (梯度爆炸) | 中 | 中 | tau_scale 和 beta_scale 做 warmup: 前 5k steps 使用均匀破坏, 渐进切换到层次化 |
| R3 | 编译成功率长期 < 70% | 低 | 高 | 增加约束解码的介入力度: 训练后 50k steps 加入 grammar mask 作为辅助损失 |
| R4 | FlashAttention 编译失败 (sm_90 兼容) | 低 | 低 | 回退到 `torch.nn.functional.scaled_dot_product_attention` (PyTorch 原生) |
| R5 | 数据量不足 (178k 可能不够 178M 参数模型) | 中 | 中 | 加强数据增强 (5× augmentation), 或减小模型至 small config (d=512, 12层, ~60M) |
| R6 | 条件编码器 (T5) 导致显存压力 | 低 | 低 | 已预设 freeze=True; 如仍紧张, 用更小的 T5-small 或先做无条件训练 |
| R7 | GC-Mamba 的双向扫描速度慢于预期 | 中 | 中 | Profiling 后若 Mamba 占比 >40% 总时间, 考虑 chunked scan 或减少深层 Mamba 的 d_state |
| R8 | 评估指标 (CD) 计算需要 CAD 内核执行 | 中 | 中 | 先用 AST 结构指标 (编译率/类型兼容率) 作为 proxy, CD 计算延后到 Phase 7 |

### 16.1 关键决策点

| 时间点 | 检查项 | 决策 |
|--------|--------|------|
| **P3 结束** (第 5 周) | 单样本过拟合是否成功? | 如果失败: 降低模型规模 → small config |
| **50k steps** | 编译成功率是否 > 50%? | 如果否: 增加 grammar mask 辅助损失, 或检查层次化破坏参数 |
| **100k steps** | Loss 是否仍在下降? | 如果平台: 调整 LR, 增加 batch size, 或尝试 warmup 重启 |
| **P7 消融** | GC-Mamba 是否显著优于标准 Mamba? | 如果差距 < 1%: 简化为标准 Mamba 以降低工程复杂度 |

---

> **文档状态**：开发指南 v1.0。随着实际开发推进，各 Phase 的预估时间和资源消耗可能调整。建议每周末更新此文档中的进度和实际数据。
