import sys

import torch
import inspect
import logging
import functools
import torch.nn as nn
# import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import defaultdict
from awq.utils.calib_data import get_calib_dataset
from awq.quantize.scale import apply_scale, apply_clip
from awq.utils.utils import clear_memory, get_best_device
from awq.modules.linear.gemm import WQLinear_GEMM
from awq.modules.linear.gemv import WQLinear_GEMV
from awq.modules.linear.marlin import WQLinear_Marlin
from awq.utils.module import (
    append_str_prefix,
    get_op_name,
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
)

import transformers
import math
import os
class GPTQ:
    def __init__(self, inp):
        self.inp = inp.clone()
        self.dev = self.inp.device
        self.column = self.inp.shape[-1]
        self.H = torch.zeros((self.column, self.column), device=self.dev)
        self.nsamples = 0
        self.dead = None    #清除W中使H对角线为0的列
        self.use_batch_update = False
        self.batch_update() #计算H

    def add_batch(self, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        # 只支持线性层
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        """
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        """
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())
        del inp

    def batch_update(self):
        if len(self.inp.shape) == 3 and self.use_batch_update:
            for i in range(self.inp.shape[0]):
                self.add_batch(self.inp[i])
        else:
            self.add_batch(self.inp)

    def cal_Hinv(self, H=None):
        if H == None:
            H = self.H.clone()
        dead = torch.diag(H) == 0
        self.dead = dead
        H[dead, dead] = 1

        damp = 0.01 * torch.mean(torch.diag(H))
        diag = torch.arange(self.column, device=self.dev)

        # H[diag, diag] += damp
        while True:
            try:
                H = torch.linalg.cholesky(H)
                break
            except Exception as e:
                H[diag, diag] += damp
        Hinv = torch.cholesky_inverse(H)
        assert torch.isfinite(Hinv).all()
        del H
        return(Hinv)

    """
    def get_Hinv(self, H=None):
        if H == None:
            H = self.H.clone()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1

        damp = 0.01 * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        while True:
            try:
                H = torch.linalg.cholesky(H)
                break
            except Exception as e:
                H[diag, diag] += damp
        H = torch.cholesky_inverse(H)
        Hinv_diag = torch.diag(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv_cholesky = H
        return(Hinv_cholesky, Hinv_diag)
    """
    def reset(self):
        self.H = torch.zeros((self.column, self.column), device=self.dev)
        self.nsamples = 0

    def free(self):
        self.H = None
        self.inp = None
        torch.cuda.empty_cache()

class AwqQuantizer:
    def __init__(
        self,
        awq_model,
        model,
        tokenizer,
        w_bit,
        group_size,
        zero_point,
        version,
        calib_data,
        split,
        text_column,
        duo_scaling,
        modules_to_not_convert=None,
        export_compatible=False,

        apply_scale=True,
        apply_clip=False,
        apply_update=False,
        optimize_all_linears=False,
    ) -> None:
        self.awq_model = awq_model
        self.model = model
        self.tokenizer = tokenizer
        self.w_bit = w_bit
        self.group_size = group_size
        self.zero_point = zero_point
        self.version = version
        self.calib_data = calib_data
        self.split = split
        self.text_column = text_column
        self.duo_scaling = duo_scaling
        self.export_compatible = export_compatible

        # 修改
        self.apply_scale= True
        self.apply_clip = True
        self.apply_update = True
        self.optimize_all_linears = True
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )
        self.modules, self.module_kwargs, self.inps = self.init_quant()

    def pseudo_quantize_tensor(self, w: torch.Tensor):
        org_bit = self.w_bit
        # ==============================================================================
        self.w_bit = 4
        org_w_shape = w.shape
        assert self.group_size > 0
        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0
            w = w.reshape(-1, self.group_size)
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0
        assert torch.isinf(w).sum() == 0

        # zero point quantization
        if self.zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2**self.w_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            w = (
                torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
            ) * scales
            zeros = zeros.view(org_w_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (self.w_bit - 1) - 1
            min_int = -(2 ** (self.w_bit - 1))
            scales = max_val / max_int
            zeros = None
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        scales = scales.view(org_w_shape[0], -1)
        w = w.reshape(org_w_shape)

        self.w_bit = org_bit
        return w, scales, zeros

    def pseudo_dequantize_tensor(
        self, w: nn.Linear, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None
    ):
        # get repeated count
        repeat_count = w.weight.data.shape[-1] // scales.shape[-1]
        scales = scales.repeat(1, repeat_count).reshape(w.weight.data.shape)

        # dequantize
        if self.zero_point:
            zeros = zeros.repeat(1, repeat_count).reshape(w.weight.data.shape)
            w = (w.weight.data - zeros) * scales
        else:
            w = w.weight.data * scales

        return w

    def quantize(self):
        for i in tqdm(range(len(self.modules)), desc="OOO"):
            # Move module and inputs to correct device
            common_device = next(self.modules[i].parameters()).device
            if common_device is None or str(common_device) == "cpu":
                if torch.cuda.is_available():
                    best_device = "cuda:" + str(i % torch.cuda.device_count())
                else:
                    best_device = get_best_device()

                self.modules[i] = self.modules[i].to(best_device)
                common_device = next(self.modules[i].parameters()).device

            if self.module_kwargs.get("position_ids") is not None:
                self.module_kwargs["position_ids"] = self.module_kwargs[
                    "position_ids"
                ].to(common_device)

            if self.module_kwargs.get("attention_mask") is not None:
                self.module_kwargs["attention_mask"] = self.module_kwargs[
                    "attention_mask"
                ].to(common_device)

            self.inps = self.inps.to(common_device)

            # [STEP 1]: Get layer, extract linear modules, extract input features
            named_linears = get_named_linears(self.modules[i])

            # Filter out the linear layers we don't want to exclude
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )

            #{'attn.c_attn': Linear(in_features=2048, out_features=6144, bias=True),
            # 'attn.c_proj': Linear(in_features=2048, out_features=2048, bias=False),
            # 'mlp.w1': Linear(in_features=2048, out_features=5504, bias=False),
            # 'mlp.w2': Linear(in_features=2048, out_features=5504, bias=False),
            # 'mlp.c_proj': Linear(in_features=5504, out_features=2048, bias=False)}

            # 'self_attn.q_proj': Linear(in_features=896, out_features=896, bias=True),
            # 'self_attn.k_proj': Linear(in_features=896, out_features=128, bias=True),
            # 'self_attn.v_proj': Linear(in_features=896, out_features=128, bias=True),
            # 'self_attn.o_proj': Linear(in_features=896, out_features=896, bias=False),
            # 'mlp.gate_proj': Linear(in_features=896, out_features=4864, bias=False),
            # 'mlp.up_proj': Linear(in_features=896, out_features=4864, bias=False),
            # 'mlp.down_proj': Linear(in_features=4864, out_features=896, bias=False)


            input_feat = self._get_input_feat(self.modules[i], named_linears)
            clear_memory()

            # [STEP 2]: Compute and apply scale list
            module_config: List[Dict] = self.awq_model.get_layers_for_scaling(
                self.modules[i], input_feat, self.module_kwargs
            )

            layers_to_scale = []
            for m in module_config:
                for n in m['layers']:
                    layers_to_scale.append(get_op_name(self.modules[i], n))

            linears_not_scale = exclude_layers_to_not_quantize(
                named_linears, layers_to_scale
            )
            del layers_to_scale

            """
            def get_layers_for_scaling(module, input_feat, module_kwargs):
                layers = []
                # attention
                layers.append(
                    dict(
                        prev_op=module.ln_1,
                        layers=[module.attn.c_attn],
                        inp=input_feat["attn.c_attn"],
                        module2inspect=module.attn,
                        kwargs=module_kwargs,
                    )
                )
                # mlp
                layers.append(
                    dict(
                        prev_op=module.ln_2,
                        layers=[module.mlp.w2, module.mlp.w1],
                        inp=input_feat["mlp.w2"],
                        module2inspect=module.mlp,
                    )
                )
                # linear 2
                layers.append(
                    dict(
                        prev_op=module.mlp.w1,
                        layers=[module.mlp.c_proj],
                        inp=input_feat["mlp.c_proj"],
                    )
                )
                return layers
            """

            for layer in module_config:
                # scale，clip，补偿权重
                if self.apply_scale:
                    scales_list = [ self._search_best_scale(self.modules[i], **layer) ]
                    # scale激活值
                    apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)

                for prev_op_name, layer_names, scales in scales_list:
                    for layer_name in layer_names:
                        if layer_name in input_feat:
                            # input_feat[layer_name].cpu()
                            del input_feat[layer_name]



            # scales_list = append_str_prefix(
            #     scales_list, get_op_name(self.model, self.modules[i]) + "."
            # )

            # [STEP 3]: Compute and apply clipping for linears not to scale
            if self.optimize_all_linears:
                self._search_best_clip(self.modules[i], linears_not_scale, input_feat)

            del input_feat

            """
            # [STEP 3]: Compute and apply clipping list
            clip_list = self._search_best_clip(
                self.modules[i], named_linears, input_feat
            )
            #apply_clip(self.modules[i], clip_list, input_feat) #修改部分
            clip_list = append_str_prefix(
                clip_list, get_op_name(self.model, self.modules[i]) + "."
            )
            """

            # [STEP 4]: Quantize weights
            if not self.export_compatible:
                self._apply_quant(self.modules[i], named_linears)

            clear_memory()

    def pack(self):
        for i in tqdm(range(len(self.modules)), desc="Packing"):
            named_linears = get_named_linears(self.modules[i])
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )
            self._apply_quant(self.modules[i], named_linears)
            clear_memory()

    def _apply_quant(self, module, named_linears: Dict[str, nn.Linear]):
        for name, linear_layer in named_linears.items():
            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            linear_layer = linear_layer.to(get_best_device()).half()

            linear_layer.weight.data, scales, zeros = self.pseudo_quantize_tensor(
                linear_layer.weight.data
            )

            if self.version == "gemm":
                scales = scales.t().contiguous()
                zeros = zeros.t().contiguous()
                q_linear_module = WQLinear_GEMM

            elif self.version == "gemv":
                q_linear_module = WQLinear_GEMV

            elif self.version == "marlin":
                q_linear_module = WQLinear_Marlin

            else:
                raise ValueError(f"Unknown version {self.version}")

            q_linear = q_linear_module.from_linear(
                linear=linear_layer,
                w_bit=self.w_bit,
                group_size=self.group_size,
                init_only=False,
                scales=scales,
                zeros=zeros,
            )

            linear_layer.cpu()
            q_linear.to(next(module.parameters()).device)
            set_op_by_name(module, name, q_linear)
            clear_memory()


    @torch.no_grad()
    def _search_best_scale(
        self,
        module,
        prev_op,
        layers: List[nn.Linear],
        inp: torch.Tensor,
        module2inspect=None,
        kwargs={},
    ):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        module2inspect = module2inspect.to(get_best_device())

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        # Put x on the right device
        inp = inp.to(next(module2inspect.parameters()).device)

        device = inp.device
        ci = inp.shape[-1]
        assert len(inp.shape) == 3, inp.shape
        n_sample = inp.shape[0] if len(inp.shape) == 3 else 1
        n_sample_token = inp.view(-1, ci).shape[0]
        # group_size = self.group_size if self.group_size > 0 and ci % self.group_size == 0 else ci
        assert self.group_size > 0 and ci % self.group_size == 0
        group_size = self.group_size

        # Compute mean of x
        x_mean = inp.abs().view(-1, inp.shape[-1]).mean(0)

        # weights = list(m.weight.data.clone() for m in layers)
        weights = list(m.weight.data for m in layers)

        if (len(weights) > 1):
            # alpha = [
            #     (inp.view(-1, ci) @ (w.T)).float().pow(2).sum().item() for w in weights
            # ]

            alpha = [0] * len(weights)  # 初始化alpha列表，长度与weights相同
            inp_view = inp.view(-1, ci)
            batch_size = 512
            for i in range(0, inp_view.size(0), batch_size):
                inp_batch = inp_view[i : min((i + batch_size), inp_view.size(0))]
                batch_results = [(inp_batch @ (w.T)).float().pow(2).sum().item() for w in weights]
                for r, result in enumerate(batch_results):
                    alpha[r] += result

        cross_group_update = [
            w[:, :group_size].clone() for w in weights
        ]

        best_scales_all = torch.ones(1, ci, dtype=x_mean.dtype, device=device)
        
        for i in range(ci // group_size):
            i1 = i * group_size
            i2 = i1 + group_size

            x = inp[..., i1: i2].view(-1, group_size)

            # [STEP 1]: Compute per-channel mean of normalised weights
            # All layer group weights are concatted together
            weight = torch.cat([_m[:, i1 :i2] for _m in weights], dim=0)

            # Calculates the relative magnitude of the weights within each of the quantization groups,
            # and rescales each group individually so that each group has weights on a 0-1 scale.
            w_scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)

            # Gets the average rescaled magnitude for each output channel
            w_mean = w_scale.mean(0)
            clear_memory(weight)

            group_x = inp[..., i1: i2]
            gptq = GPTQ(group_x)
            H = gptq.H
            gptq.free()
            del gptq

            if self.apply_scale:
                # 搜索本组best_scale
                n_grid = 20
                history = []
                best_ratio = -1
                best_scales = None
                best_error = float("inf")

                for ratio in range(n_grid):
                    # create new scales
                    ratio = ratio / n_grid

                    if self.duo_scaling:
                        scales = (x_mean[i1:i2].pow(ratio) / (w_mean.pow(1 - ratio)+ 1e-4)).clamp(min=1e-4)
                    else:
                        scales = x_mean[i1:i2].pow(ratio).clamp(min=1e-4).view(-1)
                    # scales = scales / (scales.max() * scales.min()).sqrt()

                    scales_view = scales.view(1, -1).to(device)
                    # avoid scaling values that overflow
                    scales[torch.isinf(scales)] = 1
                    scales[torch.isnan(scales)] = 1

                    loss_all = 0.0
                    for j in range(len(weights)):
                        # H_scale = (scales_view.float()) @ (scales_view.t().float())
                        # assert torch.sum(H_scale == 0) == 0
                        # scaled_H = H / H_scale

                        ori_w = weights[j][:, i1: i2] * scales_view
                        q_w = self.pseudo_quantize_tensor(ori_w)[0] / scales_view

                        err = (cross_group_update[j] - q_w).float()
                        del ori_w
                        del q_w

                        loss = (err @ H) * err
                        # 检查
                        # loss = loss.sum(dim=-1, keepdim=True)
                        # assert torch.sum(loss < 0) == 0
                        loss = loss.sum().item()

                        if (len(weights) > 1):
                            loss /= alpha[j]
                            assert torch.isfinite(torch.tensor(loss)).all()
                        loss_all += loss

                    history.append(loss_all)
                    if loss_all < best_error:
                        best_error = loss_all
                        best_ratio = ratio
                        best_scales = scales.clone()

                assert best_error != float("inf")
                best_scales_all[:, i1: i2] = best_scales

                H_scale = (best_scales_all[:, i1: i2].float()) @ (best_scales_all[:, i1: i2].t().float())
                assert torch.sum(H_scale == 0) == 0
                H /= H_scale


            for k in range(len(weights)):
                layer_weight = weights[k]
                ori_w = layer_weight[:, i1: i2]

                if self.apply_scale:
                    # W * S 原地修改
                    layer_weight[:, i1: i2] *= best_scales_all[:, i1: i2]
                    assert torch.isfinite(layer_weight[:, i1: i2]).all()
                    cross_group_update[k] *= best_scales_all[:, i1: i2]
                    assert torch.isfinite(cross_group_update[k]).all()


                if self.apply_clip:
                    # Clipper
                    # 搜索本组best_clip
                    n_grid = 20
                    max_shrink = 0.5
                    org_max_val = ori_w.abs().amax(dim=-1, keepdim=True)
                    best_max_val = org_max_val.clone()
                    min_errs = torch.ones_like(org_max_val.float()) * 1e9

                    for i_s in range(int(max_shrink * n_grid)):
                        max_val = org_max_val * (1 - i_s / n_grid)
                        min_val = -max_val
                        # 生成新tensor
                        cur_w = torch.clamp(ori_w, min_val, max_val)
                        q_w = self.pseudo_quantize_tensor(cur_w)[0]
                        del cur_w

                        err = (cross_group_update[k] - q_w).float()
                        err = (err @ H) * err
                        err = err.sum(dim=-1, keepdim=True)
                        assert torch.sum(err < 0) == 0
                        # if i_s == 0:
                        #     print('原始：',err)
                        assert torch.isfinite(err).all()
                        del q_w

                        cur_best_idx = err < min_errs
                        min_errs[cur_best_idx] = err[cur_best_idx]
                        best_max_val[cur_best_idx] = max_val[cur_best_idx]

                    assert torch.sum(min_errs == 1e9) == 0

                    # ori_w -> clip_w -> quant_w
                    # clip_w 原地修改
                    torch.clamp_(ori_w, -best_max_val, best_max_val)


                if self.apply_update and i2 != ci:
                    quant_w = self.pseudo_quantize_tensor(ori_w)[0]

                    cross_group_x = inp[..., i1: i2 + group_size].clone()
                    cross_group_x[..., : group_size] /= best_scales_all[:, i1: i2]
                    assert torch.isfinite(cross_group_x).all()
                    gptq = GPTQ(cross_group_x)
                    del cross_group_x
                    Hinv = gptq.cal_Hinv()
                    # inv = torch.linalg.cholesky(Hinv[:group_size, :group_size])
                    # inv = torch.cholesky_inverse(inv)
                    inv = torch.linalg.inv(Hinv[:group_size, :group_size])
                    assert torch.isfinite(inv).all()

                    dead = gptq.dead
                    gptq.free()
                    del gptq

                    layer_weight[:, i1: i2 + group_size][:, dead] = 0

                    # 更新下一组权重
                    err = (cross_group_update[k] - quant_w).float()
                    err = (err @ inv) @ (Hinv[: group_size, group_size:])
                    del quant_w


                    assert torch.isnan(err).sum() == 0
                    assert torch.isinf(err).sum() == 0

                    cross_group_update[k] = layer_weight[:, i2: i2 + group_size] - err
                    assert torch.isnan(cross_group_update[k]).sum() == 0
                    assert torch.isinf(cross_group_update[k]).sum() == 0

                elif i2 != ci:
                    cross_group_update[k] = layer_weight[:, i2: i2 + group_size].clone()

        best_scales_all = best_scales_all.view(-1)

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            best_scales_all.detach().cpu(),
        )

    @torch.no_grad()
    def _search_best_clip(self, module, named_linears, input_feat):
        # clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in named_linears:
            if not self.apply_clip:
                continue

            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue

            device = get_best_device()
            named_linears[name].to(device)
            input_feat[name] = input_feat[name].to(device)

            weight = named_linears[name].weight.data
            inp = input_feat[name]

            ci = inp.shape[-1]
            assert self.group_size > 0 and ci % self.group_size == 0
            group_size = self.group_size

            cross_group_update = weight[:, : group_size].clone()

            for i in range(ci // group_size):
                i1 = i * group_size
                i2 = i1 + group_size

                ori_w = weight[:, i1 :i2]

                if i2 != ci:
                    cross_group_x = inp[..., i1: i2 + group_size]
                    gptq = GPTQ(cross_group_x)
                    del cross_group_x
                    H = gptq.H[:group_size, :group_size]

                    Hinv = gptq.cal_Hinv()
                    # inv = torch.linalg.cholesky(Hinv[:group_size, :group_size])
                    # inv = torch.cholesky_inverse(inv)
                    inv = torch.linalg.inv(Hinv[:group_size, :group_size])
                    weight[:, i1: i2 + group_size][:, gptq.dead] = 0
                else:
                    cross_group_x = inp[..., i1: i2]
                    gptq = GPTQ(cross_group_x)
                    del cross_group_x
                    H = gptq.H
                gptq.free()
                del gptq


                if self.apply_clip:
                    # Clipper
                    n_grid = 20
                    max_shrink = 0.5
                    org_max_val = ori_w.abs().amax(dim=-1, keepdim=True)
                    best_max_val = org_max_val.clone()
                    min_errs = torch.ones_like(org_max_val.float()) * 1e9

                    for i_s in range(int(max_shrink * n_grid)):
                        max_val = org_max_val * (1 - i_s / n_grid)
                        min_val = -max_val
                        # 生成新tensor
                        cur_w = torch.clamp(ori_w, min_val, max_val)
                        q_w = self.pseudo_quantize_tensor(cur_w)[0]
                        del cur_w

                        err = (cross_group_update - q_w).float()
                        err = (err @ H) * err
                        err = err.sum(dim=-1, keepdim=True)

                        assert torch.sum(err < 0) == 0
                        assert torch.isfinite(err).all()
                        del q_w

                        cur_best_idx = err < min_errs
                        min_errs[cur_best_idx] = err[cur_best_idx]
                        best_max_val[cur_best_idx] = max_val[cur_best_idx]

                    assert torch.sum(min_errs == 1e9) == 0, torch.sum(min_errs == 1e9)

                    # ori_w -> clip_w -> quant_w
                    # clip_w 原地修改
                    torch.clamp_(ori_w, -best_max_val, best_max_val)


                if self.apply_update and i2 != ci:
                    quant_w = self.pseudo_quantize_tensor(ori_w)[0]

                    err = (cross_group_update - quant_w).float()
                    err = err @ inv @ (Hinv[: group_size, group_size:])

                    del quant_w

                    # 更新权重
                    cross_group_update = weight[:, i2: i2 + group_size] - err
                    assert torch.isnan(cross_group_update).sum() == 0
                    assert torch.isinf(cross_group_update).sum() == 0

                elif i2 != ci:
                    cross_group_update = weight[:, i2: i2 + group_size].clone()


            clear_memory(input_feat[name])
            named_linears[name].cpu()


            # clip_list.append((name, max_val))

        # return clip_list

    def init_quant(self, n_samples=128, seqlen=512):    #128    512
        modules = self.awq_model.get_model_layers(self.model)
        samples = get_calib_dataset(
            data="pileval",
            tokenizer=self.tokenizer,
            n_samples=n_samples,
            block_size=seqlen,
            split=self.split,
            text_column=self.text_column,
        )
        samples = torch.cat(samples, dim=0)
        # print("samples.shape:",samples.shape)

        inps = []
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)

        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # assume first input to forward is hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs
        modules[0] = Catcher(modules[0])
        try:
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:  # work with early exit
            pass

        # Update the layer kwargs with `prepare_inputs_for_generation` method
        # that takes care of everything to avoid unexpected errors.
        layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs)
        # Pop the input_ids as they are not needed at all.
        layer_kwargs.pop("input_ids")

        del samples
        modules[0] = modules[0].module  # restore
        inps = inps[0]
        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")

        clear_memory()

        if layer_kwargs.get("attention_mask") is not None:
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(
                best_device
            )

        return modules, layer_kwargs, inps

    def _get_input_feat(self, layer, named_linears):
        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []

        # FIXME: Workaround for Mixtral to use block_sparse_moe input features
        if self.awq_model.model_type == "mixtral":
            named_linears = {
                **named_linears,
                "block_sparse_moe": layer.block_sparse_moe,
            }

        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        self.inps = self.inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input

        # Sanitize the kwargs in case we use transformers version that contains
        # kwargs that are not handled by the module.
        # Useful for trust_remote_code models.
        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)

        self.inps = layer(self.inps, **module_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        return input_feat

    def _sanitize_kwargs(self, inputs_kwargs, module):
        """
        Remove the arguments that are not supported in the module's
        forward pass to avoid breaking behaviour between different versions
        of transformers.

        Args:
            inputs_kwargs (`dict`):
                The input dictionary to pass to the model layer
            module (`torch.nn.Module`):
                Target module to quantize.
        """
        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
        for k, v in inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs
