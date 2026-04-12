# SpatialAST：面向层次化破坏扩散的空间语义 AST 工业级架构规范

> **版本**：v2.0 (GeoFusion-CAD informed revision)  
> **基于**：draft.md 初步构思 + GeoFusion-CAD G-Mamba 设计复盘  
> **目标**：定义一套类型安全、可编译、可扩散、可编辑的 CAD 生成 AST 架构

---

## 目录

1. [系统总览](#1-系统总览)
2. [AST 类型系统](#2-ast-类型系统)
3. [Token 词表与编码](#3-token-词表与编码)
4. [序列化协议](#4-序列化协议)
5. [层次化破坏扩散引擎](#5-层次化破坏扩散引擎)
6. [Mamba-Transformer 混合去噪网络](#6-mamba-transformer-混合去噪网络)
7. [约束解码与校验系统](#7-约束解码与校验系统)
8. [确定性编译器](#8-确定性编译器)
9. [数据管线](#9-数据管线)
10. [训练系统](#10-训练系统)
11. [推理管线](#11-推理管线)
12. [工程实现规范](#12-工程实现规范)

---

## 1. 系统总览

### 1.1 端到端流水线

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SpatialAST Pipeline                             │
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────┐              │
│  │  Input    │    │  Condition   │    │   Hierarchical    │              │
│  │ (Text /  │───▶│  Encoder     │───▶│   Diffusion       │              │
│  │  Image)  │    │  Module      │    │   Engine          │              │
│  └──────────┘    └──────────────┘    └────────┬──────────┘              │
│                                               │                         │
│                                               ▼                         │
│                                      ┌────────────────┐                 │
│                                      │ Mamba-Trans.   │                 │
│                                      │ Hybrid Denoiser│                 │
│                                      └────────┬───────┘                 │
│                                               │                         │
│                                               ▼                         │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────┐                 │
│  │ 3D Model │◀───│  DeepCAD     │◀───│  Constraint    │                 │
│  │ (B-Rep)  │    │  Compiler    │    │  Decoder       │                 │
│  └──────────┘    └──────────────┘    └────────────────┘                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心设计原则

| 原则 | 说明 |
|------|------|
| **类型安全** | 每个 AST 节点有严格的类型签名、子节点约束和参数 schema |
| **深度对齐** | AST 树深度 ↔ 空间粒度 ↔ 扩散噪声调度严格一一对应 |
| **可编译性** | AST 通过确定性编译器无损转换为 DeepCAD 指令序列 |
| **可编辑性** | 用户可在 AST 任意子树级别进行局部修改后重新编译 |
| **可逆性** | DeepCAD 指令序列 ↔ AST 之间双向无损转换 |
| **可扩展性** | 节点类型可通过注册机制扩展（如新增 `Fillet`、`Chamfer`） |

### 1.3 模块依赖关系

```
                    ┌─────────────────────────┐
                    │     Config Registry      │
                    │  (节点注册 / 超参管理)    │
                    └────────┬────────────────┘
                             │
              ┌──────────────┼───────────────────┐
              │              │                   │
              ▼              ▼                   ▼
    ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐
    │ Type System │  │  Tokenizer   │  │ Condition Encoder │
    │ (§2)        │  │  (§3)        │  │ (§6.4)            │
    └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘
           │                │                    │
           ▼                ▼                    │
    ┌──────────────────────────────┐              │
    │   Serializer / Deserializer  │              │
    │   (§4)                       │              │
    └──────────────┬───────────────┘              │
                   │                              │
                   ▼                              │
    ┌──────────────────────────────┐              │
    │  Hierarchical Corruption    │◀─────────────┘
    │  Engine (§5)                 │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Mamba-Transformer Denoiser  │
    │  (§6)                        │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Constraint Decoder (§7)     │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Deterministic Compiler (§8) │
    └──────────────────────────────┘
```

---

## 2. AST 类型系统

### 2.1 节点类型层级定义

AST 采用 6 级深度（L0–L5），每级对应一个空间语义粒度。节点类型通过 **枚举注册表** 管理，保证可扩展。

#### 2.1.1 完整节点类型表

| 深度 | 节点类型 | 类型标签 | 子节点签名 | 参数 Schema | 语义 |
|------|---------|----------|-----------|-------------|------|
| **L0** | `Program` | `PROG` | `children: List[Solid]` (≥1) | `version: str` | 模型根节点 |
| **L1** | `Solid` | `SOL` | `sketch: Sketch, ops: List[Operation]` (ops ≥1) | `name: Optional[str]` | 独立实体 |
| **L1** | `BooleanOp` | `BOOL` | `left: Solid, right: Solid` | `op_type: Enum{union, intersect, subtract}` | 布尔运算 |
| **L2** | `Sketch` | `SKT` | `faces: List[Face]` (≥1) | `plane: PlaneRef, transform: Transform2D` | 2D 草图 |
| **L2** | `Extrude` | `EXT` | ∅ (叶操作节点) | `distance_fwd: Q8, distance_bwd: Q8, direction: Vec3, op_type: Enum{new, cut, join}` | 拉伸操作 |
| **L2** | `Revolve` | `REV` | ∅ (叶操作节点) | `angle: Q8, axis: Vec3, op_type: Enum{new, cut, join}` | 旋转操作（预留扩展） |
| **L3** | `Face` | `FACE` | `outer: Loop, inners: List[Loop]` (inners ≥0) | `face_id: int` | 面 |
| **L4** | `Loop` | `LOOP` | `edges: List[Edge]` (≥1) | `is_outer: bool` | 闭合环 |
| **L4** | `Edge` | `EDGE` | `curve: Curve` (恰好 1 个) | `edge_id: int` | 边 |
| **L5** | `Line` | `LN` | `start: Coord, end: Coord` | ∅ | 线段 |
| **L5** | `Arc` | `ARC` | `start: Coord, mid: Coord, end: Coord` | ∅ | 三点圆弧 |
| **L5** | `Circle` | `CIR` | `center: Coord, radius: Scalar` | ∅ | 完整圆 |
| **L5** | `Coord` | `CRD` | ∅ (叶节点) | `x: Q8, y: Q8` | 2D 坐标值 |
| **L5** | `Scalar` | `SCL` | ∅ (叶节点) | `value: Q8` | 标量参数 |

> **Q8**：8-bit 量化值，范围 [0, 255]，映射到归一化物理坐标 [-1, 1]。

#### 2.1.2 类型约束规则（Grammar Rules）

使用上下文无关文法（CFG）描述合法 AST 结构：

```
Program     → SOL_LIST
SOL_LIST    → Solid SOL_LIST | Solid
Solid       → Sketch OP_LIST
            | BooleanOp
OP_LIST     → Operation OP_LIST | Operation
Operation   → Extrude | Revolve
Sketch      → FACE_LIST
FACE_LIST   → Face FACE_LIST | Face
Face        → Loop INNER_LOOPS
INNER_LOOPS → Loop INNER_LOOPS | ε
Loop        → EDGE_LIST
EDGE_LIST   → Edge EDGE_LIST | Edge
Edge        → Curve
Curve       → Line | Arc | Circle
Line        → Coord Coord
Arc         → Coord Coord Coord
Circle      → Coord Scalar
Coord       → Q8_VAL Q8_VAL
Scalar      → Q8_VAL
```

#### 2.1.3 父子兼容性矩阵

预计算的静态查找表，用于约束解码时的 O(1) 类型检查：

```
PARENT_CHILD_MATRIX[parent_type][child_type] → bool

PROG  : {SOL: ✓, BOOL: ✓}
SOL   : {SKT: ✓, EXT: ✓, REV: ✓}
BOOL  : {SOL: ✓}
SKT   : {FACE: ✓}
EXT   : {}  (无子节点)
REV   : {}  (无子节点)
FACE  : {LOOP: ✓}
LOOP  : {EDGE: ✓}
EDGE  : {LN: ✓, ARC: ✓, CIR: ✓}
LN    : {CRD: ✓}
ARC   : {CRD: ✓}
CIR   : {CRD: ✓, SCL: ✓}
CRD   : {}  (叶节点, 参数内嵌)
SCL   : {}  (叶节点, 参数内嵌)
```

#### 2.1.4 子节点数量约束

```python
CHILDREN_CARDINALITY = {
    "PROG":  {"SOL": (1, MAX_SOLIDS)},        # 至少 1 个 Solid
    "SOL":   {"SKT": (1, 1), "OP": (1, MAX_OPS)},  # 恰好 1 个 Sketch + ≥1 个 Operation
    "BOOL":  {"SOL": (2, 2)},                  # 恰好 2 个 Solid
    "SKT":   {"FACE": (1, MAX_FACES)},         # 至少 1 个 Face
    "FACE":  {"LOOP": (1, MAX_LOOPS)},         # 至少 1 个 Loop（外环）
    "LOOP":  {"EDGE": (1, MAX_EDGES)},         # 至少 1 个 Edge
    "EDGE":  {"CURVE": (1, 1)},                # 恰好 1 条曲线
    "LN":    {"CRD": (2, 2)},                  # 恰好 2 个坐标
    "ARC":   {"CRD": (3, 3)},                  # 恰好 3 个坐标
    "CIR":   {"CRD": (1, 1), "SCL": (1, 1)},  # 1 个坐标 + 1 个标量
}

MAX_SOLIDS = 16
MAX_OPS    = 8
MAX_FACES  = 32
MAX_LOOPS  = 8
MAX_EDGES  = 64
```

### 2.2 节点数据结构

```python
@dataclass(frozen=True)
class ASTNode:
    node_type: NodeType          # 枚举类型标签
    depth: int                   # AST 深度 (0-5)
    children: Tuple[ASTNode, ...]  # 不可变子节点元组
    params: Dict[str, Any]       # 节点参数 (类型化, 可为空)
    node_id: int                 # 唯一标识符 (用于编辑定位)
    span: Tuple[int, int]        # 在序列化 token 流中的 [start, end) 区间

class NodeType(IntEnum):
    # L0
    PROG = 0
    # L1
    SOL  = 1
    BOOL = 2
    # L2
    SKT  = 3
    EXT  = 4
    REV  = 5
    # L3
    FACE = 6
    # L4
    LOOP = 7
    EDGE = 8
    # L5 - Curves
    LN   = 9
    ARC  = 10
    CIR  = 11
    # L5 - Values
    CRD  = 12
    SCL  = 13
    # Special
    MASK  = 14
    NOISE = 15
    NIL   = 16

DEPTH_OF: Dict[NodeType, int] = {
    NodeType.PROG: 0,
    NodeType.SOL: 1, NodeType.BOOL: 1,
    NodeType.SKT: 2, NodeType.EXT: 2, NodeType.REV: 2,
    NodeType.FACE: 3,
    NodeType.LOOP: 4, NodeType.EDGE: 4,
    NodeType.LN: 5, NodeType.ARC: 5, NodeType.CIR: 5,
    NodeType.CRD: 5, NodeType.SCL: 5,
}
```

### 2.3 节点注册机制（可扩展性）

```python
class NodeRegistry:
    """全局节点类型注册表，支持运行时扩展新的节点类型。"""

    _registry: Dict[str, NodeSpec] = {}

    @classmethod
    def register(cls, spec: "NodeSpec"):
        assert spec.tag not in cls._registry, f"Duplicate tag: {spec.tag}"
        cls._registry[spec.tag] = spec

    @classmethod
    def get(cls, tag: str) -> "NodeSpec":
        return cls._registry[tag]

@dataclass
class NodeSpec:
    tag: str                          # 类型标签 (如 "EXT")
    depth: int                        # 深度等级
    child_schema: List[ChildSlot]     # 子节点槽位定义
    param_schema: Dict[str, ParamDef] # 参数定义
    token_id: int                     # 词表中的 token ID

@dataclass
class ChildSlot:
    name: str                    # 槽位名称 (如 "sketch")
    allowed_types: Set[str]      # 允许的子节点类型
    min_count: int               # 最小数量
    max_count: int               # 最大数量

@dataclass
class ParamDef:
    name: str
    dtype: str                   # "q8" | "enum" | "vec3" | "str"
    enum_values: Optional[List[str]] = None
    default: Optional[Any] = None
```

---

## 3. Token 词表与编码

### 3.1 词表设计

词表分为 4 个区段，每区段占固定位宽，总词表大小约 **300 个 token**：

```
┌────────────────────────────────────────────────────────────────┐
│                      Token Vocabulary                          │
├────────────┬──────────┬────────────────────────────────────────┤
│  区段       │  ID 范围  │  说明                                  │
├────────────┼──────────┼────────────────────────────────────────┤
│ Special    │  0 - 7   │ PAD, BOS, EOS, MASK, NOISE, SEP,      │
│            │          │ UNK, NIL                               │
├────────────┼──────────┼────────────────────────────────────────┤
│ Structure  │  8 - 31  │ 节点类型 token (PROG, SOL, BOOL, ...)  │
│            │          │ + 开括号 "(" + 闭括号 ")"               │
│            │          │ + 参数分隔符 ","                        │
├────────────┼──────────┼────────────────────────────────────────┤
│ Enum       │  32 - 47 │ 枚举参数值 (union, intersect, subtract,│
│            │          │  new, cut, join, x_axis, y_axis, ...)  │
├────────────┼──────────┼────────────────────────────────────────┤
│ Quantized  │  48 - 303│ Q8 数值 token (0-255)                  │
│ Values     │          │ 映射：token_id = value + 48            │
└────────────┴──────────┴────────────────────────────────────────┘
```

### 3.2 特殊 Token 定义

| Token | ID | 功能 |
|-------|----|------|
| `[PAD]` | 0 | 序列填充 |
| `[BOS]` | 1 | 序列起始 |
| `[EOS]` | 2 | 序列结束 |
| `[MASK]` | 3 | 扩散掩码（替代被破坏的节点） |
| `[NOISE]` | 4 | 噪声子树标记（替代被破坏的子树） |
| `[SEP]` | 5 | 子节点列表分隔符 |
| `[UNK]` | 6 | 未知 token |
| `[NIL]` | 7 | 空节点标记 |

### 3.3 Token 元数据

每个 token 携带以下元数据（不参与词表编码，作为辅助信息传入模型）：

```python
@dataclass
class TokenMeta:
    position: int          # 序列中的位置索引
    depth: int             # 当前 token 所在 AST 节点的深度 (0-5)
    node_type: NodeType    # 当前 token 所属节点的类型
    role: TokenRole        # 该 token 在节点中的角色
    parent_type: NodeType  # 父节点的类型 (根节点为 NIL)
    sibling_idx: int       # 在兄弟节点中的序号 (0-based)

class TokenRole(IntEnum):
    NODE_TAG   = 0   # 节点类型标签 (如 PROG, SOL, ...)
    OPEN_PAREN = 1   # 开括号 "("
    CLOSE_PAREN = 2  # 闭括号 ")"
    PARAM_VALUE = 3  # 参数值 (Q8 或枚举)
    SEPARATOR  = 4   # 分隔符
    SPECIAL    = 5   # 特殊 token (BOS/EOS/MASK/...)
```

### 3.4 几何描述符 (借鉴 GeoFusion-CAD)

除离散元数据外，每个 token 还关联一个连续几何描述符向量，用于条件化 SSM 状态转移核：

```python
@dataclass
class GeometryDescriptor:
    scale: float           # 局部几何尺度 (边长/面直径/草图跨度，归一化到 [0,1])
    curvature: float       # 曲率描述符 (Line=0, Arc=1/R, Circle=1/R)
    depth_ratio: float     # 归一化深度 depth / max_depth ∈ [0, 1]
    subtree_size: int      # 当前节点子树的 token 数 (用于尺度感知)
```

几何描述符在数据预处理阶段从 AST 结构中提取，作为额外的输入通道传入模型。与 GeoFusion-CAD 中的 \(\Delta_k = g(s_k, d_k, r_k)\) 对应，但我们将其与 AST 的层次化语义绑定。

---

## 4. 序列化协议

### 4.1 DFS 括号序列格式

AST 树通过深度优先前序遍历序列化为线性 token 流。格式规则如下：

```
serialize(node) :=
    NODE_TAG(node.type)
    PARAMS(node.params)      // 参数值紧跟节点标签
    "("
    serialize(child_1)
    serialize(child_2)
    ...
    serialize(child_n)
    ")"
```

叶节点（无子节点）省略括号：

```
serialize(leaf) :=
    NODE_TAG(leaf.type)
    PARAMS(leaf.params)
```

### 4.2 序列化示例

对于一个简单的矩形拉伸模型：

```
AST 树:
  Program
  └── Solid
      ├── Sketch
      │   └── Face
      │       └── Loop
      │           ├── Edge → Line(0,0 → 128,0)
      │           ├── Edge → Line(128,0 → 128,128)
      │           ├── Edge → Line(128,128 → 0,128)
      │           └── Edge → Line(0,128 → 0,0)
      └── Extrude(distance=64, op=new)

序列化 Token 流:
[BOS] PROG ( SOL ( SKT ( FACE ( LOOP (
  EDGE ( LN CRD 0 0 CRD 128 0 )
  EDGE ( LN CRD 128 0 CRD 128 128 )
  EDGE ( LN CRD 128 128 CRD 0 128 )
  EDGE ( LN CRD 0 128 CRD 0 0 )
) ) ) EXT 64 0 new ) ) [EOS]

对应深度标注:
 0    0  0  1  1  2  2  3  3  4  4
  4   4  5  5 5 5  5  5 5 5  4
  4   4  5  5 5 5  5  5  5 5  4
  4   4  5  5  5 5  5  5 5 5  4
  4   4  5  5 5 5  5  5 5 5  4
 4  3  2  2  2 2  2  1  0    0
```

### 4.3 最大序列长度

根据 DeepCAD 数据集统计：

| 统计量 | 值 |
|--------|-----|
| 平均序列长度 | ~120 tokens |
| P95 序列长度 | ~280 tokens |
| P99 序列长度 | ~450 tokens |
| 最大序列长度 | ~800 tokens |
| **设计策略** | **可变长度 + 动态 padding** (保留全部样本) |

序列长度不设硬上限。预处理阶段保留所有样本的自然长度序列（含 >512 的复杂 CAD 模型），
训练时由 DataLoader 按 batch 内最长序列动态 padding，并结合 bucket batching
将长度相近的样本分组以减少 padding 浪费。Mamba 的 O(n) 复杂度使长序列训练可行，
而 FlashAttention 的 O(n²) 部分在 n≈800 时仍在 H100 的可承受范围内。

### 4.4 序列化/反序列化伪代码

```python
class ASTSerializer:
    def serialize(self, root: ASTNode) -> List[int]:
        tokens = [TOKEN_BOS]
        self._dfs_serialize(root, tokens)
        tokens.append(TOKEN_EOS)
        return self._pad_or_truncate(tokens, MAX_SEQ_LEN)

    def _dfs_serialize(self, node: ASTNode, tokens: List[int]):
        tokens.append(node.node_type.token_id)
        for p in node.params.values():
            tokens.append(self._encode_param(p))

        if node.children:
            tokens.append(TOKEN_LPAREN)
            for child in node.children:
                self._dfs_serialize(child, tokens)
            tokens.append(TOKEN_RPAREN)

    def deserialize(self, tokens: List[int]) -> ASTNode:
        tokens = self._strip_special(tokens)
        root, _ = self._dfs_deserialize(tokens, pos=0)
        return root

    def _dfs_deserialize(self, tokens: List[int], pos: int) -> Tuple[ASTNode, int]:
        node_type = NodeType(tokens[pos])
        pos += 1
        spec = NodeRegistry.get(node_type.name)

        params = {}
        for pname, pdef in spec.param_schema.items():
            params[pname] = self._decode_param(tokens[pos], pdef)
            pos += 1

        children = []
        if pos < len(tokens) and tokens[pos] == TOKEN_LPAREN:
            pos += 1  # skip "("
            while pos < len(tokens) and tokens[pos] != TOKEN_RPAREN:
                child, pos = self._dfs_deserialize(tokens, pos)
                children.append(child)
            pos += 1  # skip ")"

        return ASTNode(
            node_type=node_type,
            depth=DEPTH_OF[node_type],
            children=tuple(children),
            params=params,
            node_id=self._next_id(),
            span=(start_pos, pos),
        ), pos
```

---

## 5. 层次化破坏扩散引擎

### 5.1 架构概览

```
┌──────────────────────────────────────────────────────────────────┐
│              Hierarchical Corruption Engine                      │
│                                                                  │
│  输入: clean AST token sequence X_0, timestep t                  │
│  输出: corrupted sequence X_t, corruption mask M_t               │
│                                                                  │
│  ┌────────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│  │ Depth-Aware    │    │  Subtree         │    │  Corruption  │  │
│  │ Noise Schedule │───▶│  Selector        │───▶│  Applicator  │  │
│  │ p(d, t)        │    │  (哪些子树破坏)   │    │  (如何破坏)   │  │
│  └────────────────┘    └──────────────────┘    └──────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 深度感知噪声调度

#### 5.2.1 调度函数

```python
def corruption_probability(depth: int, t: int, T: int, config: DiffusionConfig) -> float:
    """
    计算深度 d 的节点在时间步 t 被破坏的概率。

    核心性质:
    - t→0: 所有深度的 p→0 (无破坏)
    - t→T: 浅层(d=0) p→1, 深层(d=5) p 仍可控
    - 固定 t: p 随 d 增大而单调递减 (粗粒度先破坏)
    """
    tau_d = (1.0 - depth / config.max_depth) * config.tau_scale * T
    beta = config.beta_scale * T
    return sigmoid((t - tau_d) / beta)

@dataclass
class DiffusionConfig:
    T: int = 1000              # 总扩散步数
    max_depth: int = 5         # AST 最大深度
    tau_scale: float = 0.9     # 破坏阈值缩放因子
    beta_scale: float = 0.1    # sigmoid 陡峭度
    corruption_modes: List[str] = field(
        default_factory=lambda: ["mask", "noise", "shuffle"]
    )
```

#### 5.2.2 破坏阈值时间表

| 深度 d | τ(d) / T | 含义 |
|--------|----------|------|
| 0 (Program) | 0.90 | 仅在极高噪声时才破坏根节点 |
| 1 (Solid/Bool) | 0.72 | 高噪声阶段破坏实体级 |
| 2 (Sketch/Extrude) | 0.54 | 中高噪声阶段破坏操作级 |
| 3 (Face) | 0.36 | 中等噪声阶段破坏面级 |
| 4 (Loop/Edge) | 0.18 | 低噪声阶段破坏环/边级 |
| 5 (Curve/Param) | 0.00 | 最先开始破坏的细节级 |

#### 5.2.3 可视化

```
p(d,t) ▲
  1.0  │                            ╱───── d=5 (参数)
       │                         ╱╱─────── d=4 (边)
       │                      ╱╱╱───────── d=3 (面)
  0.5  │                   ╱╱╱╱─────────── d=2 (草图)
       │                ╱╱╱╱╱───────────── d=1 (实体)
       │             ╱╱╱╱╱╱─────────────── d=0 (程序)
  0.0  │____________╱╱╱╱╱╱╱
       └──────────────────────────────▶ t
       0                              T
```

### 5.3 破坏操作类型

```python
class CorruptionMode(Enum):
    MASK     = "mask"      # 用 [MASK] token 替换节点标签
    NOISE    = "noise"     # 用 [NOISE] 替换整棵子树
    SHUFFLE  = "shuffle"   # 随机打乱同级子节点顺序
    RESAMPLE = "resample"  # 用随机 Q8 值替换数值参数

class SubtreeCorruptor:
    def corrupt_node(self, tokens: List[int], node_span: Tuple[int, int],
                     mode: CorruptionMode) -> List[int]:
        start, end = node_span
        if mode == CorruptionMode.MASK:
            tokens[start] = TOKEN_MASK
        elif mode == CorruptionMode.NOISE:
            tokens[start:end] = [TOKEN_NOISE]
        elif mode == CorruptionMode.SHUFFLE:
            children_spans = self._get_children_spans(tokens, start, end)
            random.shuffle(children_spans)
            self._reorder_spans(tokens, children_spans)
        elif mode == CorruptionMode.RESAMPLE:
            for i in range(start, end):
                if self._is_q8_token(tokens[i]):
                    tokens[i] = random.randint(Q8_OFFSET, Q8_OFFSET + 255)
        return tokens
```

### 5.4 训练时前向破坏流程

```python
def hierarchical_corrupt(
    tokens: List[int],
    depth_map: List[int],         # 每个 token 对应的 AST 深度
    subtree_map: List[int],       # 每个 token 所属子树的根节点 ID
    t: int,
    config: DiffusionConfig,
) -> Tuple[List[int], List[bool]]:
    """
    返回: (corrupted_tokens, corruption_mask)
    corruption_mask[i] = True 表示 token i 被破坏
    """
    corrupted = tokens.copy()
    mask = [False] * len(tokens)
    corrupted_subtrees = set()

    for node_id, (span, depth) in enumerate(subtree_info):
        if node_id in corrupted_subtrees:
            continue

        p = corruption_probability(depth, t, config.T, config)
        if random.random() < p:
            mode = random.choice(config.corruption_modes)
            corrupted = corrupt_node(corrupted, span, mode)
            mark_subtree_corrupted(mask, span)
            corrupted_subtrees.add(node_id)
            corrupted_subtrees.update(get_descendant_ids(node_id))

    return corrupted, mask
```

### 5.5 训练损失

v2.0 在 v1.0 的 mask prediction CE + L2 回归基础上，借鉴 GeoFusion-CAD 的混合损失设计，新增 **节点类型辅助 CE** 和 **参数分组 CE**，使结构预测和数值预测各有专门的监督信号。

对应 GeoFusion 的总损失：\(\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \sum_i [\text{CCE}(\hat{c}_i, c_i) + \eta \sum_j \text{ACE}(\hat{a}_{i,j}, a_{i,j})]\)

```python
def compute_loss(
    model_output: Tensor,       # [B, L, V] logits
    target_tokens: Tensor,      # [B, L] ground truth token IDs
    corruption_mask: Tensor,    # [B, L] bool mask
    depth_map: Tensor,          # [B, L] depth of each token
    role_map: Tensor,           # [B, L] token role (NODE_TAG / PARAM_VALUE / ...)
    config: LossConfig,
) -> Tensor:
    # === 1. 主损失: 掩码预测 CE (深度加权) ===
    ce_loss = F.cross_entropy(
        model_output[corruption_mask].view(-1, V),
        target_tokens[corruption_mask].view(-1),
        reduction='none'
    )
    depth_weights = 1.0 + config.depth_weight_alpha * depth_map[corruption_mask].float()
    weighted_ce = (ce_loss * depth_weights).mean()

    # === 2. 数值参数 L2 回归 ===
    is_q8 = (target_tokens >= Q8_OFFSET) & (target_tokens < Q8_OFFSET + 256)
    q8_mask = corruption_mask & is_q8
    if q8_mask.any():
        pred_vals = model_output[q8_mask].argmax(-1).float() - Q8_OFFSET
        true_vals = target_tokens[q8_mask].float() - Q8_OFFSET
        reg_loss = F.mse_loss(pred_vals, true_vals)
    else:
        reg_loss = torch.tensor(0.0)

    # === 3. [NEW] 节点类型辅助 CE (← GeoFusion CCE) ===
    # 单独监督结构 token 的预测，加速模型学会正确的 AST 骨架
    is_node_tag = (role_map == TokenRole.NODE_TAG)
    node_mask = corruption_mask & is_node_tag
    if node_mask.any():
        node_ce = F.cross_entropy(
            model_output[node_mask].view(-1, V),
            target_tokens[node_mask].view(-1),
        )
    else:
        node_ce = torch.tensor(0.0)

    # === 4. [NEW] 参数值辅助 CE (← GeoFusion ACE) ===
    # 单独监督参数 token 的分类预测
    is_param = (role_map == TokenRole.PARAM_VALUE)
    param_mask = corruption_mask & is_param
    if param_mask.any():
        param_ce = F.cross_entropy(
            model_output[param_mask].view(-1, V),
            target_tokens[param_mask].view(-1),
        )
    else:
        param_ce = torch.tensor(0.0)

    return (weighted_ce
            + config.reg_weight * reg_loss
            + config.node_ce_weight * node_ce
            + config.param_ce_weight * param_ce)

@dataclass
class LossConfig:
    reg_weight: float = 0.1
    depth_weight_alpha: float = 0.2
    node_ce_weight: float = 0.5     # [NEW] 节点类型辅助损失权重
    param_ce_weight: float = 2.0    # [NEW] 参数辅助损失权重 (← GeoFusion η=2)
```

> **设计考量**：GeoFusion 中 η=2 (参数损失权重高于命令损失)，因为 CAD 参数精度对最终几何质量影响更大。我们沿用此策略，`param_ce_weight=2.0`。同时新增 `node_ce_weight=0.5` 监督结构 token，帮助模型在扩散早期快速确定 AST 骨架。

---

## 6. Mamba-Transformer 混合去噪网络

### 6.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Hybrid Denoiser Network                           │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Input Embedding Layer                     │    │
│  │  e_k = TokenEmbed(x_k) + PosEmbed(k) + DepthEmbed(d_k)    │    │
│  │        + TypeEmbed(type_k) + RoleEmbed(role_k)              │    │
│  └─────────────────────────┬───────────────────────────────────┘    │
│                            │                                        │
│                            ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Timestep Conditioning (AdaLN)                  │    │
│  │              t_emb = MLP(sinusoidal(t))                     │    │
│  └─────────────────────────┬───────────────────────────────────┘    │
│                            │                                        │
│       ┌────────────────────┼─────────────────────┐                  │
│       │                    │                     │                  │
│       ▼                    ▼                     ▼                  │
│  ┌──────────┐       ┌──────────┐          ┌──────────┐             │
│  │ Block 1  │       │ Block k  │   ...    │ Block L  │             │
│  │ (Attn    │  ...  │ (Balanced│          │ (Mamba   │             │
│  │  Heavy)  │       │  Hybrid) │          │  Heavy)  │             │
│  └──────────┘       └──────────┘          └──────────┘             │
│       │                    │                     │                  │
│       └────────────────────┼─────────────────────┘                  │
│                            │                                        │
│                            ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Output Projection                        │    │
│  │              Linear(d_model → vocab_size)                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 输入嵌入层

借鉴 GeoFusion-CAD 的层次化位置编码 \(\Pi_k = \text{PE}(p_k, \sigma_k, \tau_k)\)，我们在原有 4 路嵌入基础上新增 **父节点类型嵌入** 和 **兄弟索引嵌入**，使模型能显式感知树结构拓扑：

```python
class SpatialASTEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embed   = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed     = RotaryPositionalEncoding(config.d_model)
        self.depth_embed   = nn.Embedding(config.max_depth + 1, config.d_model)
        self.type_embed    = nn.Embedding(config.num_node_types, config.d_model)
        self.role_embed    = nn.Embedding(config.num_roles, config.d_model)
        self.parent_embed  = nn.Embedding(config.num_node_types, config.d_model)
        self.sibling_embed = nn.Embedding(config.max_siblings, config.d_model)
        self.geom_proj     = nn.Linear(config.geom_desc_dim, config.d_model)
        self.layer_norm    = nn.LayerNorm(config.d_model)
        self.dropout       = nn.Dropout(config.embed_dropout)

    def forward(self, token_ids, depth_ids, type_ids, role_ids,
                parent_ids, sibling_ids, geom_desc):
        x = (self.token_embed(token_ids)
             + self.depth_embed(depth_ids)
             + self.type_embed(type_ids)
             + self.role_embed(role_ids)
             + self.parent_embed(parent_ids)
             + self.sibling_embed(sibling_ids)
             + self.geom_proj(geom_desc))
        x = self.pos_embed(x)
        return self.dropout(self.layer_norm(x))
```

> **设计决策**：GeoFusion 将 \(\Pi_k\) 仅注入到 Mamba 的深度卷积后，而我们选择在输入嵌入层统一注入，使 Transformer 和 Mamba 子层都能受益。`geom_proj` 将连续几何描述符（尺度/曲率/深度比/子树大小，共 4 维）线性投影到 `d_model` 维后与离散嵌入相加。

### 6.3 混合块设计

v2.0 的混合块借鉴 GeoFusion-CAD 的三个关键设计：(1) **深度卷积局部平滑**，(2) **几何条件化 SSM 核**，(3) **全局-局部门控融合**（替代 v1.0 的标量 gate）。

```
┌─────────────────────────────────────────────────────────────────┐
│                    HybridBlock (v2.0)                            │
│                                                                 │
│  Input x ──▶ DWConv ──▶ ┌─────────┐   ┌─────────────┐          │
│                          │  MHA    │   │  GC-Mamba   │          │
│              (局部平滑)   │ (局部)  │   │  (全局)     │          │
│                          └────┬────┘   └──────┬──────┘          │
│                               │               │                │
│                               ▼               ▼                │
│                          ┌────────────────────────┐             │
│                          │  Global-Local Gated    │             │
│                          │  Fusion (GLF)          │             │
│                          │  out = W(h_attn ⊙ σ(  │             │
│                          │        h_mamba))       │             │
│                          └───────────┬────────────┘             │
│                                      │                         │
│                                      ▼                         │
│                          ┌────────────────────┐                │
│                          │  CrossAttention     │                │
│                          │  (条件注入)         │                │
│                          └───────────┬────────┘                │
│                                      │                         │
│                                      ▼                         │
│                          ┌────────────────────┐                │
│                          │  SwiGLU FFN        │                │
│                          └───────────┬────────┘                │
│                                      │                         │
│                                  Output x                      │
└─────────────────────────────────────────────────────────────────┘
```

```python
class HybridBlock(nn.Module):
    """
    v2.0 混合块: DWConv → MHA ∥ GC-Mamba → GLF → CrossAttn → FFN
    核心改进:
    1. 深度卷积预平滑 (← GeoFusion DWConv)
    2. 几何条件化 Mamba 核 (← GeoFusion GSM-SSD)
    3. 全局-局部门控融合替代标量 gate (← GeoFusion Hadamard gating)
    """
    def __init__(self, config: ModelConfig, block_idx: int):
        super().__init__()
        total_blocks = config.num_blocks
        phase = block_idx / total_blocks

        if phase < 1/3:
            attn_dim = int(config.d_model * 0.7)
            mamba_dim = int(config.d_model * 0.3)
        elif phase < 2/3:
            attn_dim = int(config.d_model * 0.5)
            mamba_dim = int(config.d_model * 0.5)
        else:
            attn_dim = int(config.d_model * 0.3)
            mamba_dim = int(config.d_model * 0.7)

        # [NEW] 深度卷积: 局部平滑 + 相邻曲线段连续性
        self.dwconv = DepthwiseConv1d(config.d_model, kernel_size=config.dwconv_kernel)
        self.dwconv_norm = nn.LayerNorm(config.d_model)

        # MHA 子层 (局部几何细节)
        self.norm1 = AdaLayerNorm(config.d_model, config.time_embed_dim)
        self.attn = MultiHeadAttention(
            d_model=config.d_model,
            num_heads=max(1, attn_dim // config.head_dim),
            head_dim=config.head_dim,
            use_rope=True,
        )

        # [NEW] 几何条件化 Mamba (全局结构传播)
        self.norm2 = AdaLayerNorm(config.d_model, config.time_embed_dim)
        self.gc_mamba = GeometryConditionedMamba(
            d_model=config.d_model,
            d_state=mamba_dim,
            d_conv=config.mamba_conv_dim,
            expand=config.mamba_expand,
            geom_dim=config.geom_desc_dim,
        )

        # [NEW] 全局-局部门控融合 (替代 v1.0 标量 gate)
        self.glf = GlobalLocalGatedFusion(config.d_model)

        # Cross-Attention (条件注入)
        self.norm3 = AdaLayerNorm(config.d_model, config.time_embed_dim)
        self.cross_attn = MultiHeadCrossAttention(
            d_model=config.d_model,
            num_heads=config.cross_attn_heads,
            head_dim=config.head_dim,
        )

        # FFN
        self.norm4 = AdaLayerNorm(config.d_model, config.time_embed_dim)
        self.ffn = SwiGLU_FFN(d_model=config.d_model, d_ff=config.d_ff)

    def forward(self, x, t_emb, cond_kv, geom_desc, mask=None):
        # Step 0: 深度卷积局部平滑 (← GeoFusion DWConv + Π_k)
        x = x + self.dwconv_norm(self.dwconv(x))

        # Step 1: MHA → 捕捉局部几何细节
        h_attn = self.norm1(x, t_emb)
        h_attn = self.attn(h_attn, h_attn, h_attn, mask=mask)

        # Step 2: GC-Mamba → 传播全局结构依赖 (几何条件化核)
        h_mamba = self.norm2(x, t_emb)
        h_mamba = self.gc_mamba(h_mamba, geom_desc)

        # Step 3: 门控融合 (← GeoFusion GSM Hadamard gating)
        # h_attn 扮演 GeoFusion 中的 "局部几何 z"
        # h_mamba 扮演 GeoFusion 中的 "全局拓扑 h"
        h_fused = self.glf(h_global=h_mamba, h_local=h_attn)
        x = x + h_fused

        # Step 4: Cross-Attention (条件注入)
        h = self.norm3(x, t_emb)
        h = self.cross_attn(query=h, key=cond_kv, value=cond_kv)
        x = x + h

        # Step 5: FFN
        h = self.norm4(x, t_emb)
        h = self.ffn(h)
        x = x + h

        return x
```

### 6.4 自适应层归一化（AdaLN）

```python
class AdaLayerNorm(nn.Module):
    """通过时间步嵌入自适应调制 LayerNorm 的 scale 和 shift。"""
    def __init__(self, d_model: int, t_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(t_dim, 2 * d_model)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        scale, shift = self.proj(t_emb).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
```

### 6.5 几何条件化双向 Mamba (GC-Mamba)

**核心改进 (借鉴 GeoFusion-CAD GSM-SSD)**：原始 Mamba 的 SSM 核参数 \((\bar{A}, \bar{B}, C, D)\) 虽然通过 selective scan 已是输入相关的，但缺乏对 CAD 几何语义的显式感知。GeoFusion 通过轻量 MLP 将几何描述符映射为 token 级别的 SSM 核参数。我们在双向 Mamba 中集成这一思路：

```python
class GeometryConditionedMamba(nn.Module):
    """
    几何条件化双向 Mamba。
    借鉴 GeoFusion GSM-SSD: 用几何描述符动态调制 SSM 核参数，
    使不同几何语义的 token (直线段 vs 圆弧 vs 拉伸操作) 使用不同的状态转移动力学。
    """
    def __init__(self, d_model, d_state, d_conv, expand, geom_dim):
        super().__init__()
        d_inner = d_model * expand

        # 几何条件化 MLP: Δ_k → {A_mod, B_mod, C_mod, G_mod}
        # 对应 GeoFusion 的 f_geom([Δ_k, Π_k]) → {Ā_k, B̄_k, C_k, G_k}
        self.geom_to_kernels = nn.Sequential(
            nn.Linear(geom_dim, d_state * 2),
            nn.SiLU(),
            nn.Linear(d_state * 2, 4 * d_state),
        )

        # 双向 SSM
        self.forward_ssm = MambaBlock(d_model, d_state, d_conv, expand)
        self.backward_ssm = MambaBlock(d_model, d_state, d_conv, expand)

        # 融合
        self.merge = nn.Linear(2 * d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

        # 残差门控 (对应 GeoFusion 的 G_k)
        self.residual_gate = nn.Linear(geom_dim, d_model)

    def forward(self, x, geom_desc):
        # 生成几何条件化的核调制因子 [B, L, 4*d_state]
        kernels = self.geom_to_kernels(geom_desc)
        A_mod, B_mod, C_mod, G_mod = kernels.chunk(4, dim=-1)

        # 调制 SSM 内部参数 (通过 scale 方式注入)
        fwd = self.forward_ssm(x, dt_scale=torch.sigmoid(A_mod),
                                B_scale=torch.sigmoid(B_mod))
        bwd = self.backward_ssm(x.flip(dims=[1]),
                                 dt_scale=torch.sigmoid(A_mod.flip(dims=[1])),
                                 B_scale=torch.sigmoid(B_mod.flip(dims=[1])))
        bwd = bwd.flip(dims=[1])

        merged = self.merge(torch.cat([fwd, bwd], dim=-1))

        # 几何条件化残差门控 (← GeoFusion G_k 机制)
        gate = torch.sigmoid(self.residual_gate(geom_desc))
        return self.norm(merged * torch.sigmoid(C_mod[..., :merged.shape[-1]]) + gate * x)
```

**与 GeoFusion GSM-SSD 的对应关系**：

| GeoFusion GSM-SSD | 我们的 GC-Mamba | 说明 |
|-------------------|----------------|------|
| \(f_{\text{geom}}([\Delta_k, \Pi_k]) \to \{\bar{A}_k, \bar{B}_k, C_k, G_k\}\) | `geom_to_kernels(geom_desc)` → A/B/C/G_mod | 几何描述符 → SSM 核调制 |
| \(\hat{Z}_k^c = \text{DWConv}(Z_k^c) + \Pi_k\) | 前置 `DepthwiseConv1d` 在 HybridBlock 中 | 局部平滑 + 层次编码 |
| \(C_k \hat{h} + G_k Z_k^c\) | `merged * σ(C_mod) + gate * x` | 条件化输出 + 残差门控 |
| 单向扫描 | 双向扫描 + 融合 | 我们用双向捕捉前后文依赖 |

### 6.5.1 深度卷积层

```python
class DepthwiseConv1d(nn.Module):
    """逐通道 1D 卷积，捕捉相邻 token 的局部连续性 (← GeoFusion DWConv)。"""
    def __init__(self, d_model: int, kernel_size: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv(x.transpose(1, 2)).transpose(1, 2))
```

### 6.6 全局-局部门控融合 (GLF)

**核心改进**：v1.0 使用标量 `gate_attn`/`gate_mamba` 对两路输出加权求和，表达力不足。借鉴 GeoFusion-CAD 的 GSM 门控 Hadamard 融合 \(\hat{h} = \text{Linear}(h \odot \sigma(z))\)，我们将其扩展为向量级别的门控融合：

```python
class GlobalLocalGatedFusion(nn.Module):
    """
    全局-局部解耦门控融合。
    借鉴 GeoFusion GSM 的 Hadamard 门控:
      h_global (Mamba) ↔ GeoFusion 的全局拓扑 h
      h_local  (Attn)  ↔ GeoFusion 的局部几何 z
      output = Linear(h_global ⊙ σ(h_local))

    与 GeoFusion 的区别:
    - GeoFusion 在单个 SSM 内部做 h/z 分解
    - 我们在 Mamba 和 Attention 两路输出之间做融合
    - 语义对应: Attention 天然捕捉局部 (稀疏精确), Mamba 天然传播全局 (序列扫描)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_global: Tensor, h_local: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate_proj(h_local))
        fused = h_global * gate
        return self.norm(self.out_proj(fused))
```

**直觉**：Attention 输出 `h_local` 的每个维度控制"是否让对应的 Mamba 全局信息通过"。对于需要精确局部几何控制的 token（如坐标值），Attention 提供的细粒度信号占主导；对于需要全局一致性的 token（如节点类型标签），Mamba 的长程信息通过更多。

### 6.7 条件编码器

```python
class ConditionEncoder(nn.Module):
    """多模态条件编码: 文本 (T5/CLIP) + 图像 (ViT/ConvNeXt)"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.text_encoder = T5EncoderWrapper(
            model_name=config.text_encoder_name,
            freeze=config.freeze_text_encoder,
            proj_dim=config.d_model,
        )
        self.image_encoder = ViTEncoderWrapper(
            model_name=config.image_encoder_name,
            freeze=config.freeze_image_encoder,
            proj_dim=config.d_model,
        )
        self.modality_embed = nn.Embedding(2, config.d_model)  # 0=text, 1=image
        self.fuse = nn.Linear(config.d_model, config.d_model)

    def forward(self, text_tokens=None, image=None):
        features = []
        if text_tokens is not None:
            t_feat = self.text_encoder(text_tokens)
            t_feat = t_feat + self.modality_embed(torch.zeros_like(t_feat[..., 0].long()))
            features.append(t_feat)
        if image is not None:
            i_feat = self.image_encoder(image)
            i_feat = i_feat + self.modality_embed(torch.ones_like(i_feat[..., 0].long()))
            features.append(i_feat)
        cond = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        return self.fuse(cond)
```

### 6.8 时间步编码

```python
class TimestepEncoder(nn.Module):
    def __init__(self, d_model: int, t_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.mlp[0].in_features // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim)
        emb = t[:, None].float() * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)
```

### 6.9 完整模型组装

```python
class SpatialASTDenoiser(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = SpatialASTEmbedding(config)
        self.time_encoder = TimestepEncoder(config.d_model, config.time_embed_dim)
        self.condition_encoder = ConditionEncoder(config)

        self.blocks = nn.ModuleList([
            HybridBlock(config, i) for i in range(config.num_blocks)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, token_ids, depth_ids, type_ids, role_ids,
                parent_ids, sibling_ids, geom_desc, t,
                text_tokens=None, image=None, mask=None):
        x = self.embedding(token_ids, depth_ids, type_ids, role_ids,
                           parent_ids, sibling_ids, geom_desc)
        t_emb = self.time_encoder(t)
        cond_kv = self.condition_encoder(text_tokens, image)

        for block in self.blocks:
            x = block(x, t_emb, cond_kv, geom_desc, mask)

        x = self.final_norm(x)
        logits = self.output_proj(x)
        return logits
```

### 6.10 模型配置

```python
@dataclass
class ModelConfig:
    # Vocabulary
    vocab_size: int = 304
    max_seq_len: int = None  # 可变长度; DataLoader 动态 padding

    # Embedding (v2.0: 新增 parent/sibling/geom)
    d_model: int = 768
    max_depth: int = 5
    num_node_types: int = 17
    num_roles: int = 6
    max_siblings: int = 64       # [NEW] 最大兄弟索引
    geom_desc_dim: int = 4       # [NEW] 几何描述符维度 (scale, curvature, depth_ratio, subtree_size)
    embed_dropout: float = 0.1

    # Transformer
    num_blocks: int = 18
    head_dim: int = 64
    d_ff: int = 3072             # 4 * d_model
    cross_attn_heads: int = 8

    # Mamba (v2.0: 几何条件化)
    mamba_conv_dim: int = 4
    mamba_expand: int = 2

    # [NEW] 深度卷积 (← GeoFusion DWConv)
    dwconv_kernel: int = 5

    # Timestep
    time_embed_dim: int = 512

    # Condition Encoder
    text_encoder_name: str = "google/flan-t5-base"
    image_encoder_name: str = "facebook/convnext-base-224"
    freeze_text_encoder: bool = True
    freeze_image_encoder: bool = True
```

### 6.11 模型规模估算

| 组件 | 参数量 | v2.0 变化 |
|------|--------|----------|
| Embedding Layer (含 parent/sibling/geom_proj) | ~8M | +3M (新增 3 路嵌入) |
| 18 × HybridBlock (DWConv + MHA + GC-Mamba + GLF + CrossAttn + FFN) | ~170M | +20M (GC核/GLF/DWConv) |
| Condition Encoder (frozen, 不计入可训练参数) | ~(250M frozen) | 不变 |
| Output Projection | ~0.2M | 不变 |
| **总可训练参数** | **~178M** | +23M (+15%) |

> 参数量增长约 15% 来自几何条件化核生成 MLP (每层 ~0.5M) 和门控融合 (每层 ~1.2M)。考虑到 GeoFusion-CAD 在 12 层 G-Mamba 上也约 ~120M 参数，我们的 178M 在同一量级且提供了更丰富的表达力。

---

## 6.A 与 GeoFusion-CAD G-Mamba 的对比复盘

### 采纳的改进

| # | GeoFusion 设计要素 | 我们的适配方案 | 改进幅度 |
|---|-------------------|--------------|---------|
| 1 | **几何条件化 SSM 核** \(f_{\text{geom}} \to \{\bar{A}_k, \bar{B}_k, C_k, G_k\}\) | `GeometryConditionedMamba.geom_to_kernels` 从 4 维几何描述符生成 SSM 核调制因子 | 关键改进 |
| 2 | **深度卷积局部平滑** DWConv + \(\Pi_k\) | 每个 HybridBlock 前置 `DepthwiseConv1d(kernel=5)` | 中等改进 |
| 3 | **GSM 门控 Hadamard 融合** \(\hat{h} = W(h \odot \sigma(z))\) | `GlobalLocalGatedFusion`: Mamba 输出作 h_global, Attention 输出作 h_local | 关键改进 |
| 4 | **层次化位置编码** \(\Pi_k = \text{PE}(p_k, \sigma_k, \tau_k)\) | 新增 `parent_embed` + `sibling_embed` | 中等改进 |
| 5 | **几何描述符** \(\Delta_k = g(s_k, d_k, r_k)\) | `GeometryDescriptor(scale, curvature, depth_ratio, subtree_size)` + `geom_proj` | 中等改进 |

### 有意保留的差异

| # | GeoFusion 做法 | 我们的做法 | 保留理由 |
|---|--------------|-----------|---------|
| 1 | **连续 DDPM** (高斯加噪, 预测 ε) | **离散层次化破坏** (mask-based, 深度感知) | 我们的核心创新：噪声调度与 AST 深度严格对齐，连续 DDPM 无法表达"先破坏粗粒度、后破坏细粒度"的层次逻辑 |
| 2 | **纯 Mamba** (12 层 G-Mamba) | **Mamba-Transformer 混合** (18 层, 层级交替) | AST 括号序列中存在强局部依赖（如坐标对的 x/y 必须一起正确），Transformer 注意力在此场景有不可替代的优势 |
| 3 | **单向 SSM** | **双向 Mamba** | AST 序列不是自回归生成，mask 预测需要前后文双向信息 |
| 4 | **无约束解码** | **语法掩码 + 括号平衡 + 几何检查** | 我们输出的是结构化 AST 而非平坦指令序列，必须保证类型安全和树结构合法性 |
| 5 | **在 SSM 内部做 h/z 分解** | **在 Attention/Mamba 两路输出间做融合** | 我们有两个天然的全局/局部信号源（Mamba vs Attention），比在单一 SSM 内部人工分解更自然 |

### 潜在的未来借鉴

| 方向 | GeoFusion 启发 | 适用条件 |
|------|--------------|---------|
| **辅助损失** | 命令类型 CCE + 参数 ACE 辅助损失 (权重 η=2) | 如果 mask prediction CE 收敛慢，可加入节点类型和参数值的分类辅助损失 |
| **SSD 变体** | 使用 Mamba-2 的 SSD (State Space Duality) 替代经典 SSM | 当 mamba-ssm 库稳定支持 SSD 的自定义核注入时 |
| **动态步长 Δ** | GeoFusion 中 \(\bar{A}_k\) 实际控制了离散化步长 Δ_k | 可探索让几何尺度 s_k 直接调制 Mamba 的 dt 参数 |

---

## 7. 约束解码与校验系统

### 7.1 架构

```
┌──────────────────────────────────────────────────────────────┐
│                  Constraint Decoder Pipeline                  │
│                                                              │
│  Raw Logits ──▶ ┌───────────────┐                            │
│                 │ Grammar Mask  │ (类型兼容过滤)              │
│                 └───────┬───────┘                            │
│                         │                                    │
│                         ▼                                    │
│                 ┌───────────────┐                            │
│                 │ Bracket       │ (括号平衡强制)              │
│                 │ Balancer      │                            │
│                 └───────┬───────┘                            │
│                         │                                    │
│                         ▼                                    │
│                 ┌───────────────┐                            │
│                 │ Cardinality   │ (子节点数量检查)            │
│                 │ Checker       │                            │
│                 └───────┬───────┘                            │
│                         │                                    │
│                         ▼                                    │
│                 ┌───────────────┐                            │
│                 │ Value Range   │ (Q8 数值范围裁剪)           │
│                 │ Clipper       │                            │
│                 └───────┬───────┘                            │
│                         │                                    │
│                         ▼                                    │
│                 Constrained Token                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 语法掩码（Grammar Mask）

在每个解码步骤中，根据当前解析状态动态生成合法 token 的掩码：

```python
class GrammarMask:
    def __init__(self):
        self.stack: List[ParseState] = []

    def get_valid_tokens(self, current_state: ParseState) -> Set[int]:
        valid = set()
        if current_state.expecting == "node_type":
            parent_type = current_state.parent_type
            valid = {spec.token_id for spec in NodeRegistry.get_children_of(parent_type)}
        elif current_state.expecting == "param":
            param_def = current_state.current_param_def
            if param_def.dtype == "q8":
                valid = set(range(Q8_OFFSET, Q8_OFFSET + 256))
            elif param_def.dtype == "enum":
                valid = {ENUM_TOKEN_MAP[v] for v in param_def.enum_values}
        elif current_state.expecting == "open_paren":
            valid = {TOKEN_LPAREN}
        elif current_state.expecting == "close_paren_or_child":
            parent_type = current_state.parent_type
            child_types = {spec.token_id for spec in NodeRegistry.get_children_of(parent_type)}
            valid = child_types | {TOKEN_RPAREN}
            if current_state.child_count < current_state.min_children:
                valid.discard(TOKEN_RPAREN)
            if current_state.child_count >= current_state.max_children:
                valid = {TOKEN_RPAREN}
        return valid

    def apply_mask(self, logits: Tensor, valid_tokens: Set[int]) -> Tensor:
        mask = torch.full_like(logits, float('-inf'))
        for t in valid_tokens:
            mask[t] = 0.0
        return logits + mask
```

### 7.3 括号平衡器

```python
class BracketBalancer:
    def check_and_repair(self, tokens: List[int]) -> List[int]:
        stack = []
        repaired = []
        for tok in tokens:
            if tok == TOKEN_LPAREN:
                stack.append(len(repaired))
                repaired.append(tok)
            elif tok == TOKEN_RPAREN:
                if stack:
                    stack.pop()
                    repaired.append(tok)
                # else: skip unmatched ")"
            else:
                repaired.append(tok)
        while stack:
            stack.pop()
            repaired.append(TOKEN_RPAREN)
        return repaired
```

### 7.4 几何一致性检查

```python
class GeometryChecker:
    def check_loop_closure(self, loop: ASTNode) -> bool:
        """检查 Loop 中的 Edge 序列是否首尾相连形成闭合环。"""
        edges = loop.children
        if not edges:
            return False
        first_start = self._get_start_coord(edges[0])
        prev_end = first_start
        for edge in edges:
            start = self._get_start_coord(edge)
            if not self._coords_close(prev_end, start, tol=1):
                return False
            prev_end = self._get_end_coord(edge)
        return self._coords_close(prev_end, first_start, tol=1)

    def repair_loop_closure(self, loop: ASTNode) -> ASTNode:
        """通过微调末尾坐标使 Loop 闭合。"""
        edges = list(loop.children)
        if not edges:
            return loop
        first_start = self._get_start_coord(edges[0])
        last_edge = edges[-1]
        edges[-1] = self._set_end_coord(last_edge, first_start)
        return ASTNode(
            node_type=loop.node_type,
            depth=loop.depth,
            children=tuple(edges),
            params=loop.params,
            node_id=loop.node_id,
            span=loop.span,
        )
```

---

## 8. 确定性编译器

### 8.1 编译流水线

```
AST (tree)
    │
    ▼
┌───────────────────┐
│  Semantic Check   │  ← 类型兼容性 / 子节点数量 / 数值范围
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Tree Normalizer  │  ← 消除冗余节点 / 规范化子节点顺序
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  IR Emitter       │  ← AST → 中间表示 (IR) 指令列表
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  DeepCAD Backend  │  ← IR → DeepCAD 指令序列（含量化反映射）
└────────┬──────────┘
         │
         ▼
DeepCAD Command Sequence
```

### 8.2 中间表示（IR）

```python
@dataclass
class IRInstruction:
    opcode: str          # "sketch_start", "sketch_end", "line", "arc", "circle",
                         # "loop_start", "loop_end", "extrude", "boolean"
    params: List[float]  # 浮点参数列表

class IREmitter:
    def emit(self, root: ASTNode) -> List[IRInstruction]:
        instructions = []
        self._visit(root, instructions)
        return instructions

    def _visit(self, node: ASTNode, out: List[IRInstruction]):
        handler = self._dispatch.get(node.node_type)
        if handler:
            handler(self, node, out)

    _dispatch: Dict[NodeType, Callable] = {}

    @_register(NodeType.PROG)
    def _emit_program(self, node, out):
        for child in node.children:
            self._visit(child, out)

    @_register(NodeType.SOL)
    def _emit_solid(self, node, out):
        sketch = node.children[0]
        self._visit(sketch, out)
        for op in node.children[1:]:
            self._visit(op, out)

    @_register(NodeType.SKT)
    def _emit_sketch(self, node, out):
        out.append(IRInstruction("sketch_start", []))
        for face in node.children:
            self._visit(face, out)
        out.append(IRInstruction("sketch_end", []))

    @_register(NodeType.FACE)
    def _emit_face(self, node, out):
        for loop in node.children:
            self._visit(loop, out)

    @_register(NodeType.LOOP)
    def _emit_loop(self, node, out):
        out.append(IRInstruction("loop_start", []))
        for edge in node.children:
            self._visit(edge, out)
        out.append(IRInstruction("loop_end", []))

    @_register(NodeType.LN)
    def _emit_line(self, node, out):
        start, end = node.children
        sx, sy = self._dequantize_coord(start)
        ex, ey = self._dequantize_coord(end)
        out.append(IRInstruction("line", [sx, sy, ex, ey]))

    @_register(NodeType.ARC)
    def _emit_arc(self, node, out):
        start, mid, end = node.children
        sx, sy = self._dequantize_coord(start)
        mx, my = self._dequantize_coord(mid)
        ex, ey = self._dequantize_coord(end)
        out.append(IRInstruction("arc", [sx, sy, mx, my, ex, ey]))

    @_register(NodeType.EXT)
    def _emit_extrude(self, node, out):
        d_fwd = self._dequantize(node.params["distance_fwd"])
        d_bwd = self._dequantize(node.params["distance_bwd"])
        op = EXTRUDE_OP_MAP[node.params["op_type"]]
        out.append(IRInstruction("extrude", [d_fwd, d_bwd, op]))
```

### 8.3 反量化

```python
def dequantize(q8_value: int, q_min: float = -1.0, q_max: float = 1.0) -> float:
    return q_min + (q_max - q_min) * q8_value / 255.0

def quantize(value: float, q_min: float = -1.0, q_max: float = 1.0) -> int:
    normalized = (value - q_min) / (q_max - q_min)
    return int(round(max(0, min(255, normalized * 255))))
```

### 8.4 反编译器（DeepCAD → AST）

```python
class DeepCADDecompiler:
    """将 DeepCAD 指令序列解析为 AST 树，用于训练数据生成。"""
    def decompile(self, commands: List[Dict]) -> ASTNode:
        solids = []
        i = 0
        while i < len(commands):
            solid, i = self._parse_solid(commands, i)
            solids.append(solid)
        return ASTNode(
            node_type=NodeType.PROG,
            depth=0,
            children=tuple(solids),
            params={"version": "1.0"},
            node_id=self._next_id(),
            span=(0, 0),
        )

    def _parse_solid(self, cmds, i):
        sketch, i = self._parse_sketch(cmds, i)
        ops = []
        while i < len(cmds) and cmds[i]["type"] in ("extrude", "revolve"):
            op, i = self._parse_operation(cmds, i)
            ops.append(op)
        return ASTNode(
            node_type=NodeType.SOL, depth=1,
            children=(sketch, *ops),
            params={}, node_id=self._next_id(), span=(0, 0),
        ), i
```

---

## 9. 数据管线

### 9.1 数据流

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Data Pipeline                                │
│                                                                      │
│  DeepCAD JSON ──▶ Decompiler ──▶ AST Tree ──▶ Validator ──▶         │
│                                                                      │
│  ──▶ Serializer ──▶ Token Seq ──▶ Meta Annotator ──▶                │
│                                                                      │
│  ──▶ ┌──────────────────┐                                            │
│      │  Training Sample  │                                           │
│      │  {                │                                           │
│      │    tokens: [int]  │                                           │
│      │    depths: [int]  │                                           │
│      │    types:  [int]  │                                           │
│      │    roles:  [int]  │                                           │
│      │    spans:  [...]  │                                           │
│      │    text:   str    │  (可选, 来自 Text2CAD)                     │
│      │    image:  path   │  (可选, 渲染的工程图)                      │
│      │  }                │                                           │
│      └──────────────────┘                                            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.2 数据集统计

| 指标 | 值 |
|------|-----|
| 源数据集 | DeepCAD (178,238 models) |
| 转换后有效 AST | ~170,000 (过滤不合法/超长) |
| 训练集 | 153,000 |
| 验证集 | 8,500 |
| 测试集 | 8,500 |
| 平均 token 数 | ~120 |
| 词表大小 | 304 |
| 文本条件覆盖率 | ~60% (Text2CAD 子集) |

### 9.3 DataLoader 设计

```python
class SpatialASTDataset(Dataset):
    """可变长度 dataset; padding 由 collate_fn 在 batch 维度动态完成."""
    def __init__(self, data_path: str):
        self.samples = self._load_and_validate(data_path)

    def __getitem__(self, idx) -> Dict[str, Tensor]:
        sample = self.samples[idx]
        # 返回自然长度 tensor, 由 collate_fn 按 batch 内最长序列 padding
        tokens   = sample["tokens"]     # variable length
        depths   = sample["depths"]
        types    = sample["types"]
        roles    = sample["roles"]
        parents  = sample["parents"]
        siblings = sample["siblings"]
        geom     = sample["geom_desc"]  # [L_i, 4]

        return {
            "token_ids":    torch.tensor(tokens, dtype=torch.long),
            "depth_ids":    torch.tensor(depths, dtype=torch.long),
            "type_ids":     torch.tensor(types, dtype=torch.long),
            "role_ids":     torch.tensor(roles, dtype=torch.long),
            "parent_ids":   torch.tensor(parents, dtype=torch.long),
            "sibling_ids":  torch.tensor(siblings, dtype=torch.long),
            "geom_desc":    torch.tensor(geom, dtype=torch.float32),
            "attention_mask": torch.tensor(
                [1 if t != TOKEN_PAD else 0 for t in tokens], dtype=torch.bool
            ),
            "text": sample.get("text", ""),
            "image_path": sample.get("image_path", ""),
        }

    def collate_fn(self, batch):
        collated = {k: torch.stack([b[k] for b in batch])
                    for k in batch[0] if isinstance(batch[0][k], Tensor)}
        collated["texts"] = [b["text"] for b in batch]
        collated["image_paths"] = [b["image_path"] for b in batch]
        return collated
```

---

## 10. 训练系统

### 10.1 训练循环

```python
class Trainer:
    def __init__(self, model, config: TrainConfig):
        self.model = model
        self.config = config
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98),
        )
        self.scheduler = CosineAnnealingWarmup(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        )
        self.corruption_engine = HierarchicalCorruptionEngine(config.diffusion)
        self.scaler = GradScaler()

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        token_ids = batch["token_ids"]              # [B, L]
        depth_ids = batch["depth_ids"]              # [B, L]
        type_ids  = batch["type_ids"]               # [B, L]
        role_ids  = batch["role_ids"]               # [B, L]

        B = token_ids.shape[0]
        t = torch.randint(1, self.config.diffusion.T + 1, (B,), device=token_ids.device)

        corrupted, corruption_mask = self.corruption_engine.batch_corrupt(
            token_ids, depth_ids, t
        )

        with autocast(dtype=torch.bfloat16):
            logits = self.model(
                corrupted, depth_ids, type_ids, role_ids, t,
                text_tokens=batch.get("text_tokens"),
                image=batch.get("images"),
                mask=batch["attention_mask"],
            )
            loss = compute_loss(logits, token_ids, corruption_mask, depth_ids, self.config.loss)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        return {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}
```

### 10.2 训练超参数

```python
@dataclass
class TrainConfig:
    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 200_000
    batch_size: int = 128          # 全局 batch size (4 GPU × 32)
    accumulation_steps: int = 1

    # Diffusion
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    # Distributed
    num_gpus: int = 4
    precision: str = "bf16"
    strategy: str = "ddp"

    # Checkpointing
    save_every: int = 5000
    eval_every: int = 2000
    log_every: int = 100
```

---

## 11. 推理管线

### 11.1 迭代去噪流程

```python
class SpatialASTSampler:
    def __init__(self, model, config: SamplerConfig):
        self.model = model
        self.config = config
        self.grammar_mask = GrammarMask()
        self.bracket_balancer = BracketBalancer()
        self.geometry_checker = GeometryChecker()

    @torch.no_grad()
    def sample(self, condition, num_steps: int = 50) -> ASTNode:
        B = 1
        x = self._init_noise_sequence(B)           # 全 [MASK] 序列
        depth_ids = torch.zeros(B, self.config.max_seq_len, dtype=torch.long)
        type_ids  = torch.zeros_like(depth_ids)
        role_ids  = torch.zeros_like(depth_ids)

        cond_kv = self.model.condition_encoder(**condition)
        timesteps = self._get_schedule(num_steps)   # [T, ..., 1]

        for t in timesteps:
            t_tensor = torch.full((B,), t, dtype=torch.long, device=x.device)
            logits = self.model(x, depth_ids, type_ids, role_ids, t_tensor,
                                cond_kv=cond_kv)

            confidence = F.softmax(logits, dim=-1).max(dim=-1).values  # [B, L]
            mask_positions = (x == TOKEN_MASK)
            num_unmask = self._num_to_unmask(t, mask_positions.sum())

            topk_positions = confidence[mask_positions].topk(num_unmask).indices
            new_tokens = logits[mask_positions][topk_positions].argmax(dim=-1)

            # 语法约束过滤
            for pos, tok in zip(topk_positions, new_tokens):
                valid = self.grammar_mask.get_valid_tokens_at(x, pos)
                if tok.item() not in valid:
                    filtered_logits = self.grammar_mask.apply_mask(
                        logits[0, pos], valid
                    )
                    tok = filtered_logits.argmax()
                x[0, pos] = tok

            depth_ids, type_ids, role_ids = self._recompute_metadata(x)

        tokens = x[0].tolist()
        tokens = self.bracket_balancer.check_and_repair(tokens)
        ast = ASTSerializer().deserialize(tokens)
        ast = self.geometry_checker.repair_loop_closure_recursive(ast)

        return ast

    def _get_schedule(self, num_steps: int) -> List[int]:
        """余弦调度: 从 T 到 1 的非均匀步长。"""
        indices = torch.linspace(0, 1, num_steps + 1)
        timesteps = (torch.cos(indices * math.pi / 2) * self.config.T).long()
        return timesteps[:-1].tolist()

    def _num_to_unmask(self, t: int, total_masked: int) -> int:
        """每步解码的 token 数量，随 t 递减而递增。"""
        ratio = 1.0 - t / self.config.T
        return max(1, int(total_masked * ratio * self.config.unmask_ratio))
```

### 11.2 推理加速

| 技术 | 说明 |
|------|------|
| **KV Cache** | Transformer 注意力子层缓存 key/value |
| **Mamba Chunked Scan** | 分块扫描减少内存占用 |
| **推测解码** | 小模型预测 + 大模型校验 (可选) |
| **DDIM-style 跳步** | 50 步推理 ≈ 1000 步扩散质量 |
| **FlashAttention-2** | 注意力计算加速 |
| **torch.compile** | 图编译优化 |

---

## 12. 工程实现规范

### 12.1 项目结构

```
spatial_ast/
├── configs/
│   ├── model.yaml              # 模型超参数
│   ├── train.yaml              # 训练超参数
│   └── diffusion.yaml          # 扩散超参数
├── core/
│   ├── __init__.py
│   ├── types.py                # NodeType, NodeSpec, NodeRegistry
│   ├── ast_node.py             # ASTNode 数据结构
│   ├── tokenizer.py            # Token 词表 + 编码/解码
│   ├── serializer.py           # AST ↔ Token 序列
│   └── grammar.py              # CFG 规则 + 父子兼容性矩阵
├── diffusion/
│   ├── __init__.py
│   ├── corruption.py           # 层次化破坏引擎
│   ├── schedule.py             # 深度感知噪声调度
│   └── loss.py                 # 训练损失函数
├── model/
│   ├── __init__.py
│   ├── embedding.py            # 输入嵌入层
│   ├── hybrid_block.py         # Mamba-Transformer 混合块
│   ├── mamba.py                # 双向 Mamba SSM
│   ├── attention.py            # MHA + CrossAttention
│   ├── adaln.py                # 自适应层归一化
│   ├── condition.py            # 条件编码器 (T5/ViT)
│   ├── timestep.py             # 时间步编码
│   └── denoiser.py             # 完整去噪网络
├── compiler/
│   ├── __init__.py
│   ├── ir.py                   # 中间表示定义
│   ├── emitter.py              # AST → IR
│   ├── backend.py              # IR → DeepCAD 指令
│   ├── decompiler.py           # DeepCAD 指令 → AST
│   └── quantize.py             # 量化/反量化工具
├── decoder/
│   ├── __init__.py
│   ├── grammar_mask.py         # 语法掩码生成
│   ├── bracket_balancer.py     # 括号平衡修复
│   ├── geometry_checker.py     # 几何一致性检查
│   └── constraint_decoder.py   # 约束解码管线
├── data/
│   ├── __init__.py
│   ├── dataset.py              # SpatialASTDataset
│   ├── pipeline.py             # 数据预处理管线
│   └── augmentation.py         # 数据增强 (坐标平移/缩放/镜像)
├── training/
│   ├── __init__.py
│   ├── trainer.py              # 训练循环
│   ├── evaluator.py            # 评估指标计算
│   └── callbacks.py            # 日志 / 检查点 / 早停
├── inference/
│   ├── __init__.py
│   ├── sampler.py              # 迭代去噪采样器
│   └── edit.py                 # AST 局部编辑接口
├── scripts/
│   ├── preprocess.py           # DeepCAD → AST 批量转换
│   ├── train.py                # 训练入口
│   ├── eval.py                 # 评估入口
│   ├── generate.py             # 生成入口
│   └── visualize.py            # AST 可视化
├── tests/
│   ├── test_serializer.py
│   ├── test_corruption.py
│   ├── test_compiler.py
│   ├── test_grammar_mask.py
│   └── test_geometry.py
├── requirements.txt
└── README.md
```

### 12.2 技术栈

| 组件 | 技术选型 |
|------|---------|
| 框架 | PyTorch 2.x + torch.compile |
| Mamba | `mamba-ssm` (official) |
| 注意力 | FlashAttention-2 |
| 分布式 | FSDP / DeepSpeed ZeRO-2 |
| 文本编码器 | HuggingFace Transformers (T5/CLIP) |
| 图像编码器 | torchvision / timm (ConvNeXt/ViT) |
| 配置管理 | Hydra + OmegaConf |
| 实验追踪 | Weights & Biases |
| 数据格式 | Apache Arrow / HuggingFace Datasets |
| 测试 | pytest + hypothesis (property-based) |
| CI/CD | GitHub Actions |

### 12.3 评估指标体系

| 指标类别 | 具体指标 | 说明 |
|---------|---------|------|
| **结构合法性** | Bracket Match Rate | 括号匹配率 |
| | Type Compatibility Rate | 父子类型兼容率 |
| | Cardinality Compliance Rate | 子节点数量合规率 |
| **编译成功率** | Compile Success Rate | AST → DeepCAD 可执行的比例 |
| | Roundtrip Accuracy | AST → DeepCAD → AST 重建一致率 |
| **几何质量** | Chamfer Distance (CD) | 点云距离 |
| | Minimum Matching Distance (MMD) | 最小匹配距离 |
| | Coverage (COV) | 覆盖率 |
| | Jensen-Shannon Divergence (JSD) | 分布散度 |
| **生成多样性** | Unique Rate | 去重后的唯一模型比例 |
| | Novel Rate | 未出现在训练集中的比例 |
| **可编辑性** | Edit Fidelity | 局部修改后模型合理性（人工评估） |
| | Edit Compile Rate | 编辑后 AST 可编译比例 |

### 12.4 硬件需求

| 配置 | 规格 |
|------|------|
| **训练** | 4× NVIDIA H100 80GB, NVLink |
| **内存** | ≥512 GB 系统内存 |
| **存储** | ≥2 TB NVMe SSD (数据 + checkpoints) |
| **训练时长** | ~5–7 天 (200k steps, batch=128) |
| **推理** | 单张 H100 / A100, ~0.5s per sample (50 steps) |

---

> **文档状态**：完整架构规范 v1.0，可作为实现参考。各模块接口已定义，具体实现细节在对应源文件中补充。
