"""类别路由 LoRA(class-routed LoRA)—— BrainGFM 下游适配核心模块。

设计目标(对应论文 Methods: Adaptation of BrainGFM to Downstream Tasks):
  "生成哪个模态的数据,就 LoRA 哪个类别,其他的不动。"

实现方式:
  每个条件类别(0..num_classes-1 = 7 个模态 + num_classes = uncond/CFG 空类)
  持有一套独立的低秩矩阵 (A_r, B_r)。前向时按 batch 的 label 路由:

      y = W x + (alpha / r) * B[label] @ A[label] @ x

  - A 用 kaiming 初始化、B 零初始化 => 初始 ΔW=0,LoRA 挂上后模型行为不变;
  - 训练时只有被路由到的类别行有梯度,其他类别的 A/B 保持不变
    (B 恒为 0 => ΔW 恒为 0 => 其他模态的生成分布严格不动);
  - cond_drop(CFG 训练)会把部分样本的 label 替换为 num_classes(uncond),
    这些样本自动路由到 uncond LoRA,使无条件分支与目标域同步适配,
    保证微调后 classifier-free guidance 仍然成立。

LoRA 挂载位置(与论文一致):
  - attention 的 query/key/value(mat_qkv)与输出投影(proj)
  - 前馈网络(fc1, fc2)
  - adaptive linear layers(每个 block 的 ada_lin 与 head_nm.ada_lin)
超参: rank r=8, scaling alpha=16, dropout=0.05。

路由信号的传递:
  D12 生成模型.forward / conditional_infer_cfg / autoregressive_infer_cfg
  在确定 label 后调用 set_lora_route(label_B);RoutedLoRALinear 在前向中
  通过 get_lora_route() 取路由。无 LoRA 或路由为 None 时退化为原始 Linear。
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 路由上下文(进程级单例;单进程单卡微调/推理足够,不跨进程共享)
# ---------------------------------------------------------------------------
class _RouteContext:
    __slots__ = ('route', 'sub')

    def __init__(self):
        self.route: Optional[torch.Tensor] = None
        self.sub: Optional[torch.Tensor] = None


_ROUTE = _RouteContext()


def set_lora_route(route: Optional[torch.Tensor]):
    """设置当前 batch 的 LoRA 路由索引。

    route: (B,) long tensor,取值 [0, num_routes),通常就是
    cond_drop 之后的 label_B(模态类别 0..6 或 uncond=num_classes)。
    传 None 表示"无路由"——所有 RoutedLoRALinear 退化为原始 Linear。
    """
    _ROUTE.route = route


def get_lora_route() -> Optional[torch.Tensor]:
    return _ROUTE.route


def set_sub_cond(sub: Optional[torch.Tensor]):
    """设置当前 batch 的任务子条件索引(二级条件,配合 attach_task_cond)。

    sub: (B,) long tensor,取值 [0, n_task_classes),n_task_classes 行为
    uncond(空任务条件)。None 表示本 batch 无任务子条件
    (纯模态模式,cond_BD 不加 task_emb,行为与预训练一致)。
    """
    _ROUTE.sub = sub


def get_sub_cond() -> Optional[torch.Tensor]:
    return _ROUTE.sub


def clear_lora_route():
    _ROUTE.route = None
    _ROUTE.sub = None


# ---------------------------------------------------------------------------
# 类别路由 LoRA Linear
# ---------------------------------------------------------------------------
class RoutedLoRALinear(nn.Module):
    """包装一个冻结的 nn.Linear,叠加按类别路由的低秩增量。

    base 的 weight/bias 通过 property 透传,依赖 F.linear(module.weight)
    写法的旧代码仍然可读(但不会叠加 LoRA;需叠加的调用点见
    basic_blocks.SelfAttention.forward 的 forward_with_extra_bias 分支)。
    """

    def __init__(self, base: nn.Linear, num_routes: int,
                 rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.num_routes = int(num_routes)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        in_f, out_f = base.in_features, base.out_features

        # A: 所有类别统一 kaiming 初始化(非零,保证该类别一旦被训练即有梯度);
        # B: 零初始化 => 初始 ΔW = 0,挂载不影响任何类别的生成。
        # A/B 直接建在 base 权重所在设备(挂载时模型已 .to(device) 的场景)。
        dev = base.weight.device
        A = torch.empty(self.num_routes, self.rank, in_f, device=dev)
        nn.init.kaiming_uniform_(A.reshape(-1, in_f), a=math.sqrt(5))
        self.lora_A = nn.Parameter(A.reshape(self.num_routes, self.rank, in_f).contiguous())
        self.lora_B = nn.Parameter(
            torch.zeros(self.num_routes, out_f, self.rank, device=dev))
        self.lora_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # -- Linear 兼容属性 ----------------------------------------------------
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    # -- LoRA 增量 -----------------------------------------------------------
    def _delta(self, x: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
        """x: (B, L, in) 或 (B, in);route: (B,) long。返回与 x 同形状增量。"""
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)
        xr = self.lora_drop(x)
        A = self.lora_A.index_select(0, route)     # (B, r, in)
        Bm = self.lora_B.index_select(0, route)    # (B, out, r)
        d = torch.bmm(A, xr.transpose(1, 2))       # (B, r, L)
        d = torch.bmm(Bm, d).transpose(1, 2)       # (B, L, out)
        if squeeze:
            d = d.squeeze(1)
        return d * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        route = get_lora_route()
        y = self.base(x)
        if route is None:
            return y
        return y + self._delta(x, route)

    def forward_with_extra_bias(self, x: torch.Tensor, extra_bias) -> torch.Tensor:
        """给 mat_qkv 用的入口:额外 bias 由 q_bias/zero_k_bias/v_bias 拼接而来
        (base Linear 本身 bias=False),LoRA 增量叠加其上。"""
        route = get_lora_route()
        y = F.linear(x, self.base.weight, extra_bias)
        if route is None:
            return y
        return y + self._delta(x, route)


# ---------------------------------------------------------------------------
# 任务子条件(二级条件:模态类别之下再划分的 task class conditioning)
# ---------------------------------------------------------------------------
def attach_task_cond(d12: nn.Module, num_task_classes: int,
                     lora_route_sub: bool = False):
    """在模态条件之下挂一层任务子类条件(论文 task-specific class conditioning)。

    - d12.task_emb = nn.Embedding(n_task+1, C):
        行 0..n_task-1 = 各子类(小随机初始化,联合微调时各自优化);
        行 n_task = uncond(零初始化;cond_drop 联动 drop 后的空任务条件)。
    - cond 注入方式:cond_BD = class_emb(模态) + task_emb(子类)
      (由 D12.forward / 推理入口完成,见 lora_route_compose 处)。
    - lora_route_sub=True 时 LoRA 路由升级为"模态×子类"复合索引
      (每个 (模态,子类) 组合独立 A/B,子类之间严格互不干扰);
      False 时 LoRA 仍按模态路由(同一模态的各子类共享 LoRA,
      只有 task_emb 行各自独立——论文的共享 LoRA 方式)。
    """
    n = int(num_task_classes)
    assert n >= 1
    C = d12.C
    dev = next(d12.parameters()).device      # 跟随模型所在设备
    emb = nn.Embedding(n + 1, C)
    nn.init.trunc_normal_(emb.weight.data[:n], mean=0, std=1e-2)   # 子类行:小随机
    emb.weight.data[n].zero_()                                     # uncond 行:0(无任务条件)
    emb = emb.to(dev)
    d12.task_emb = emb
    d12.task_num_classes = n
    d12.task_lora_route_sub = bool(lora_route_sub)
    set_route_compose_width(d12)      # 自动记录复合路由宽度,脚本无需手动调
    print(f"[LoRA] 已挂任务子条件 task_emb ({n}+1, C={C}); "
          f"LoRA 路由 = {'模态×子类 复合' if lora_route_sub else '模态级共享'}")
    return emb


def has_task_cond(d12: nn.Module) -> bool:
    return getattr(d12, 'task_emb', None) is not None


def lora_num_routes(d12: nn.Module) -> int:
    """当前模型挂载 LoRA 时需要的路由表大小。

    纯模态: num_classes+1;复合路由: (num_classes+1) * (n_task+1)。
    """
    n_mod = d12.num_classes + 1
    if has_task_cond(d12) and getattr(d12, 'task_lora_route_sub', False):
        n_sub = d12.task_emb.num_embeddings          # n_task+1(含 uncond)
        return n_mod * n_sub
    return n_mod


def compose_route(label_B: torch.Tensor,
                  sub_B: Optional[torch.Tensor]) -> torch.Tensor:
    """按当前模型配置合成 LoRA 路由索引(在 D12 入口调用)。

    无子条件 / 模态级共享 LoRA -> route = label_B(原逻辑不变);
    复合路由 -> route = label_B * (n_task+1) + sub_B,索引"模态×子类"行。
    宽度由 set_route_compose_width() 预先记录。
    """
    w = _current_route_width()
    if sub_B is None or w is None:
        return label_B
    return label_B * w + sub_B


# compose_route 需要知道 (n_task+1);用模块级变量记录(attach 后由脚本调用
# set_route_compose_width 设置),避免给 compose_route 增加 d12 参数。
_ROUTE_WIDTH = {'w': None}


def _current_route_width():
    return _ROUTE_WIDTH['w']


def set_route_compose_width(d12: nn.Module):
    """attach_task_cond 之后由训练/推理脚本调用,记录复合路由宽度。"""
    if has_task_cond(d12) and getattr(d12, 'task_lora_route_sub', False):
        _ROUTE_WIDTH['w'] = d12.task_emb.num_embeddings
    else:
        _ROUTE_WIDTH['w'] = None
def apply_lora_to_d12(d12: nn.Module, num_routes: int,
                      rank: int = 8, alpha: float = 16.0, dropout: float = 0.05,
                      verbose: bool = True) -> int:
    """把 RoutedLoRALinear 挂到 D12 生成模型的 qkv/proj/ffn/ada_lin 上。

    挂载点(论文:query, key, value, attention output projections,
    feed-forward network, adaptive linear layers):
      blocks.*.attn.mat_qkv / blocks.*.attn.proj
      blocks.*.ffn.fc1 / blocks.*.ffn.fc2   (fused MLP 路径被禁用以让 LoRA 生效)
      blocks.*.ada_lin[-1] / head_nm.ada_lin[-1]
    返回挂载数量。重复调用会报错(防双重包装)。
    """
    n_wrapped = 0
    for name, m in list(d12.named_modules()):
        if isinstance(m, RoutedLoRALinear):
            raise RuntimeError(f"重复挂载 LoRA: {name} 已是 RoutedLoRALinear")
        if name.endswith('.attn'):
            m.mat_qkv = RoutedLoRALinear(m.mat_qkv, num_routes, rank, alpha, dropout)
            m.proj = RoutedLoRALinear(m.proj, num_routes, rank, alpha, dropout)
            n_wrapped += 2
        elif name.endswith('.ffn'):
            m.fc1 = RoutedLoRALinear(m.fc1, num_routes, rank, alpha, dropout)
            m.fc2 = RoutedLoRALinear(m.fc2, num_routes, rank, alpha, dropout)
            # fused_mlp_func 直接用 fc1.weight/fc2.weight 走 F.linear,绕过模块
            # forward => 会丢 LoRA 增量。挂 LoRA 后必须走显式 fc1/fc2 路径。
            m.fused_mlp_func = None
            n_wrapped += 2
        elif name.endswith('.ada_lin') and isinstance(m, nn.Sequential):
            m[-1] = RoutedLoRALinear(m[-1], num_routes, rank, alpha, dropout)
            n_wrapped += 1
    if verbose:
        print(f"[LoRA] 已挂载 {n_wrapped} 个 RoutedLoRALinear "
              f"(routes={num_routes}, rank={rank}, alpha={alpha}, dropout={dropout})")
    return n_wrapped


def freeze_base_for_lora(d12: nn.Module) -> int:
    """冻结全部参数(class_emb 的行级放行由 register_class_emb_row_hook 处理,
    task_emb 的行级放行由 attach_task_cond 之后统一恢复)。"""
    n_frozen = 0
    for p in d12.parameters():
        if p.requires_grad:
            p.requires_grad_(False)
            n_frozen += 1
    # RoutedLoRALinear 的 A/B 恢复可训(挂载产生的新参数默认可训,freeze 在
    # apply 之后调用时会被关掉,这里统一恢复)
    for m in d12.modules():
        if isinstance(m, RoutedLoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)
    # 任务子条件 embedding 恢复可训(各子类行由各自样本的梯度自然隔离)
    if has_task_cond(d12):
        d12.task_emb.weight.requires_grad_(True)
    return n_frozen


def register_class_emb_row_hook(d12: nn.Module, target_idx: int,
                                tune_uncond: bool = True):
    """class_emb 只放行 target 模态行 + uncond 行(num_classes),其余行梯度清零。

    "其他的不动":非目标模态行的梯度被 hook 置 0;配合优化器中该参数
    weight_decay=0(train_lora.py 已单列 param group),其他行严格不更新。
    """
    emb = d12.class_emb.weight
    emb.requires_grad_(True)
    allow = torch.zeros(emb.shape[0], device=emb.device)
    allow[target_idx] = 1.0
    if tune_uncond:
        allow[d12.num_classes] = 1.0
    allow = allow.view(-1, 1)

    def _hook(grad):
        # grad 的设备跟随 weight(注册时模型可能尚未 .to(device),这里动态对齐)
        return grad * allow.to(grad.device)

    emb.register_hook(_hook)
    return allow.view(-1)


def lora_state_dict(d12: nn.Module) -> dict:
    """只收集 LoRA 增量 + 条件 embedding(体积小,独立于 backbone ckpt)。"""
    sd = {}
    for name, m in d12.named_modules():
        if isinstance(m, RoutedLoRALinear):
            sd[name + '.lora_A'] = m.lora_A.detach().cpu()
            sd[name + '.lora_B'] = m.lora_B.detach().cpu()
    sd['class_emb.weight'] = d12.class_emb.weight.detach().cpu()
    if has_task_cond(d12):
        sd['task_emb.weight'] = d12.task_emb.weight.detach().cpu()
    return sd


def load_lora_into_d12(d12: nn.Module, lora_sd: dict,
                       class_emb_rows=None, device='cpu',
                       task_emb_rows=None) -> int:
    """把 lora_state_dict 的增量加载回挂好 LoRA 的模型。

    class_emb_rows: None = 整表覆盖;否则只覆盖指定行(如 [3, 7] =
    flair + uncond)。多任务 LoRA 组合加载时用行级合并,互不覆盖。
    task_emb_rows: 任务子条件 embedding 的行级合并(通常整表加载)。
    返回加载的 LoRA 模块数。
    """
    n_loaded = 0
    for name, m in d12.named_modules():
        if isinstance(m, RoutedLoRALinear):
            kA, kB = name + '.lora_A', name + '.lora_B'
            if kA not in lora_sd or kB not in lora_sd:
                raise KeyError(f"LoRA ckpt 缺 key: {kA}")
            m.lora_A.data.copy_(lora_sd[kA].to(m.lora_A.device))
            m.lora_B.data.copy_(lora_sd[kB].to(m.lora_B.device))
            n_loaded += 1
    if 'class_emb.weight' in lora_sd:
        w = lora_sd['class_emb.weight'].to(d12.class_emb.weight.device)
        if class_emb_rows is None:
            d12.class_emb.weight.data.copy_(w)
        else:
            for r in class_emb_rows:
                d12.class_emb.weight.data[r] = w[r]
    if has_task_cond(d12) and 'task_emb.weight' in lora_sd:
        w = lora_sd['task_emb.weight'].to(d12.task_emb.weight.device)
        if task_emb_rows is None:
            d12.task_emb.weight.data.copy_(w)
        else:
            for r in task_emb_rows:
                d12.task_emb.weight.data[r] = w[r]
    return n_loaded


def lora_trainable_parameters(d12: nn.Module):
    """可训练参数收集:LoRA A/B + class_emb(行级 hook 限制)+ task_emb。

    task_emb 各子类行的梯度天然只来自对应子类的样本
    (sub_B 路由 + embedding 索引梯度隔离),无需额外 hook。
    """
    lora_params, class_emb_params = [], []
    for m in d12.modules():
        if isinstance(m, RoutedLoRALinear):
            lora_params += [m.lora_A, m.lora_B]
    class_emb_params.append(d12.class_emb.weight)
    if has_task_cond(d12):
        class_emb_params.append(d12.task_emb.weight)
    return lora_params, class_emb_params


def count_lora_params(d12: nn.Module) -> int:
    return sum(m.lora_A.numel() + m.lora_B.numel()
               for m in d12.modules() if isinstance(m, RoutedLoRALinear))