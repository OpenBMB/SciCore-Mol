# sft_tester2.py
# -*- coding: utf-8 -*-
import os
import sys
import io
import logging
import inspect
from typing import Optional, Dict, Any, List, Union, Tuple
import re

# 确保stdout和stderr使用UTF-8编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 如果stdout/stderr不是UTF-8，则重新包装
if hasattr(sys.stdout, 'buffer') and (not hasattr(sys.stdout, 'encoding') or sys.stdout.encoding != 'utf-8'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except (AttributeError, ValueError):
        pass

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==========================================
# 1. 自定义颜色格式化器
# ==========================================
class ColoredFormatter(logging.Formatter):
    """时间蓝色，级别/名称变色，消息原色"""
    blue = "\x1b[34;20m"    # 蓝色用于时间
    green = "\x1b[32;20m"   # 绿色用于 INFO
    yellow = "\x1b[33;20m"  # 黄色用于 WARNING
    red = "\x1b[31;20m"     # 红色用于 ERROR
    bold_red = "\x1b[31;1m" # 粗体红用于 CRITICAL
    reset = "\x1b[0m"
    
    LEVEL_COLORS = {
        logging.INFO: green,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red
    }

    def format(self, record):
        level_color = self.LEVEL_COLORS.get(record.levelno, self.reset)
        # 提取最后的类名/模块名
        short_name = record.name.split('.')[-1]
        
        # 构造格式：[时间](蓝色) 级别 [名称](变色): 消息(原色)
        log_fmt = (
            f"{self.blue}[%(asctime)s]{self.reset} "
            f"{level_color}%(levelname)s [{short_name}]:{self.reset} "
            f"%(message)s"
        )
        
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

# ==========================================
# 2. UTF-8 编码安全处理器
# ==========================================
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream or sys.stdout)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            if hasattr(self.stream, 'buffer'):
                self.stream.buffer.write(msg.encode('utf-8', errors='replace'))
                self.stream.buffer.write(b'\n')
                self.flush()
            else:
                self.stream.write(msg + '\n')
                self.flush()
        except Exception:
            self.handleError(record)

def init_logger(logger_name="SFT-Tester"):
    """
    初始化全局日志系统，并返回 SFT-Tester 专用的 logger
    """
    # 配置根日志记录器 (Root Logger) 以捕获所有模块的输出
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    
    # 清理旧的 Handlers，防止重复打印
    if root.hasHandlers():
        root.handlers.clear()
        
    # 创建并添加彩色 UTF-8 处理器
    console_handler = UTF8StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())
    root.addHandler(console_handler)
    
    # 屏蔽第三方库的干扰
    logging.getLogger("rdkit").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    return logging.getLogger(logger_name)

logger = init_logger()


from modules.mol_aware_lm import MolAwareCausalLM

# LDMol 支持
LDMOL_ENABLED = True # LDMol 使用开关 
from modules.ldmol_component import LDMolInferer # LDMol 的默认配置位于 modules/ldmol_component/ldmol_config.yaml 

def _get_model_device(model_llm: torch.nn.Module) -> torch.device:
    """兼容 device_map="auto" 的场景：输入放到第一个参数所在 device"""
    return next(model_llm.parameters()).device

def _encode_with_chat_template(
    tokenizer,
    system_msg: str,
    user_msg: str,
    enable_thinking: bool = False,
):
    """
    官方风格：优先 apply_chat_template(tokenize=True, return_tensors="pt")
    如果 tokenizer 不支持，就返回 None 让外层 fallback。
    """
    if not hasattr(tokenizer, "apply_chat_template") or getattr(tokenizer, "chat_template", None) is None:
        return None

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # 不要硬塞 enable_thinking；只有签名支持才传
    extra_kwargs = {"enable_thinking": enable_thinking}

    try:
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            **extra_kwargs
        )
        # HF 的行为：可能直接返回 input_ids，也可能返回 BatchEncoding
        if isinstance(enc, torch.Tensor):
            return {"input_ids": enc, "attention_mask": torch.ones_like(enc)}
        return enc
    except Exception:
        return None


def _encode_fallback_plain(tokenizer, system_msg: str, user_msg: str):
    """最通用的兜底：纯文本对话，不拼任何特殊 token"""
    text = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"
    return tokenizer(text, return_tensors="pt", add_special_tokens=True)

class MolAwareGenerator2:
    """
    基于 tester.py 的结构，整合 sft_tester.py 的加载逻辑和 mlp_inference.py 的 token 分类功能。
    """

    def __init__(self):
        self.model: Optional[MolAwareCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.loaded_cfg: Dict[str, Any] = {}
        self.use_multi_gpu: bool = False  # 是否使用多GPU
        # 是否使用 Llama 3.x 风格的对话模板（通过 vocab 中是否含有 header token 粗略判断）
        self.is_llama_chat_format: bool = False
        # 是否启用 thinking 模式（默认关闭）
        self.enable_thinking: bool = False
        # LDMol 组件
        self.ldmol = None   

    # ------------------------ 内部工具 ------------------------
    def _ensure_special_tokens(self):
        """
        只校验特殊 token 是否存在；推理期禁止新增，以免撕裂 embedding 权重。
        """
        assert self.tokenizer is not None
        vocab = self.tokenizer.get_vocab()

        # 标记当前 tokenizer 是否支持 Llama 风格的 header token，用于后续选择对话模板
        self.is_llama_chat_format = (
            "<|start_header_id|>" in vocab and "<|end_header_id|>" in vocab
        )

        # 只需要检查 <mol> token，其他特殊对话 token 交给各模型自身的 tokenizer/chat_template 处理
        needed = ["<mol>"]
        missing = [t for t in needed if t not in vocab]
        # if missing:
        #     raise RuntimeError(
        #         f"[vocab-mismatch] 推理期禁止新增 token。缺失: {missing}。"
        #         f"请确保导出的 tokenizer 已包含这些 token。"
        #     )

        # 兜底 eos/bos/pad
        if getattr(self.tokenizer, "eos_token_id", None) is None:
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None and eot_id >= 0:
                self.tokenizer.eos_token = "<|eot_id|>"

        if getattr(self.tokenizer, "eos_token_id", None) is None:
            try_ids = [
                getattr(self.tokenizer, "eos_token_id", None),
                getattr(self.tokenizer, "sep_token_id", None),
                getattr(self.tokenizer, "cls_token_id", None),
                getattr(self.tokenizer, "bos_token_id", None),
            ]
            try_ids = [t for t in try_ids if t is not None]
            self.tokenizer.eos_token_id = try_ids[0] if try_ids else 0

        if self.tokenizer.pad_token is None:
            if isinstance(self.tokenizer.eos_token, str):
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Llama 等 decoder-only 模型在推理/批量场景下需要左侧 padding
        self.tokenizer.padding_side = "left"

    def _sync_vocab_and_embeddings(self, strict: bool = True):
        """
        校验 tokenizer 与模型的词表大小是否一致。
        """
        assert self.model is not None and self.tokenizer is not None
        v_tok = len(self.tokenizer)
        v_model = self.model.llm.get_input_embeddings().weight.size(0)

        if v_tok == v_model:
            self.model.llm.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.llm.config.eos_token_id = self.tokenizer.eos_token_id
            self.model.llm.config.bos_token_id = self.tokenizer.bos_token_id
            return

        if strict:
            # 对于部分模型（如 Qwen 系列），HF 官方权重中 tokenizer 与 config.vocab_size
            # 本身就可能不一致，这里自动对齐而不是直接报错。
            try:
                print(f"[vocab-mismatch] tokenizer({v_tok}) != model-emb({v_model})，尝试自动 resize_token_embeddings...")
                self.model.llm.resize_token_embeddings(v_tok)
                self.model.llm.config.vocab_size = v_tok
                self.model.llm.config.pad_token_id = self.tokenizer.pad_token_id
                self.model.llm.config.eos_token_id = self.tokenizer.eos_token_id
                self.model.llm.config.bos_token_id = self.tokenizer.bos_token_id
                print(f"[vocab-mismatch] ✅ 已自动将 model-emb 调整为 {v_tok}")
                return
            except Exception as e:
                raise RuntimeError(
                    f"[vocab-mismatch] tokenizer({v_tok}) != model-emb({v_model})，且自动对齐失败: {e}"
                )

    # ------------------------ 内部辅助方法 ------------------------
    def _get_system_message(self, task_type: Optional[str], realtime_mol: bool) -> str:
        """根据任务类型选择 system message"""
        if (task_type == "molecule_generation" or task_type == "i2s") and realtime_mol:
            return "You are a helpful chemist. Generate a detailed description of a molecule based on the user's request. Include information about the molecule's structure, properties, and potential applications. Only output the description."
        elif task_type == "drug_optim" and realtime_mol:
            return "You are a medicinal chemistry assistant. Given an original molecule (SMILES) and its ADMET profile, design a single optimized molecule that best improves the stated ADMET liabilities while preserving potency.\n\nWrite naturally: first give a brief rationale (6-8 sentences) explaining the key modifications and why they help (e.g., solubility, permeability, metabolic stability, hERG, clearance). Then provide exactly one SMILES for your best design in a code block:\n\n```smiles\n<one valid SMILES here>\n```",
            # return "You are a medicinal chemistry assistant. Given an original molecule (SMILES) and its ADMET profile, design an optimization strategy.\n\nFirst, provide a 'Rationale' (6-8 sentences) explaining the medicinal chemistry reasoning, such as bioisosteric replacements or metabolic site blocking to fix ADMET issues while preserving potency.\n\nThen, provide exactly one 'SMILES Description' (1-2 sentences) that explicitly describes the structural change (e.g., 'Replace the terminal methyl group with a trifluoromethyl group'). This description must be wrapped in the specific tag below:\n\n<smiles_description>\n[Your concise modification instruction here]\n</smiles_description>"
        else:
            return "You are a careful chemist. Follow the requested output format exactly. Please avoid duplicate outputs"
    
    def _encode_prompts(
        self,
        prompts: List[str],
        system_msg: str,
        add_dialog_wrapper: bool,
    ) -> Dict[str, torch.Tensor]:
        """编码 prompts 为 input_ids 和 attention_mask"""
        if add_dialog_wrapper:
            # 逐条 apply_chat_template，再 padding
            encoded_list = []
            for p in prompts:
                enc = _encode_with_chat_template(
                    self.tokenizer, system_msg, p, 
                    enable_thinking=self.enable_thinking
                )
                if enc is None:
                    enc = _encode_fallback_plain(self.tokenizer, system_msg, p)
                encoded_list.append(enc)
            
            # 统一 padding（decoder-only 建议 left padding，已在 load 里设置）
            input_ids = [e["input_ids"].squeeze(0) for e in encoded_list]
            attn = [e["attention_mask"].squeeze(0) for e in encoded_list]
            batch_enc = self.tokenizer.pad(
                {"input_ids": input_ids, "attention_mask": attn},
                padding=True,
                return_tensors="pt"
            )
        else:
            # 不包对话就直接 tokenize
            batch_enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        
        return batch_enc
    
    def _prepare_extra_embeddings(
        self,
        extra_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        model_device: torch.device,
        num_prompts: int,
    ) -> Optional[torch.Tensor]:
        """
        准备额外的 embedding，用于作为 inputs_embeds 传入（类似 GVP 虚拟步）。
        现在支持同时传入 input_ids 和 inputs_embeds，其中 inputs_embeds 作为额外 embedding。
        
        Returns:
            inputs_embeds: [B, N, D] 或 None（如果没有额外的 embedding）
        """
        if extra_embeddings is None:
            return None
        
        # 处理 extra_embeddings：可能是单个 tensor 或列表
        if isinstance(extra_embeddings, list):
            extra_emb_list = extra_embeddings
        else:
            extra_emb_list = [extra_embeddings]
        
        # 确保数量匹配（如果只有一个，复制给所有样本）
        if len(extra_emb_list) == 1 and num_prompts > 1:
            extra_emb_list = extra_emb_list * num_prompts
        
        if len(extra_emb_list) != num_prompts:
            raise ValueError(f"extra_embeddings 数量 ({len(extra_emb_list)}) 与 prompts 数量 ({num_prompts}) 不匹配")
        
        # 处理每个 embedding，确保形状正确
        processed_embeds = []
        for emb in extra_emb_list:
            emb = emb.to(model_device)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)  # [D] -> [1, D]
            elif emb.dim() == 2:
                pass  # 已经是 [N, D]
            else:
                raise ValueError(f"extra_embeddings 的形状不正确: {emb.shape}，期望 [D] 或 [N, D]")
            processed_embeds.append(emb)
        
        # 找到最大长度并 padding（如果需要）
        max_len = max(emb.shape[0] for emb in processed_embeds)
        if max_len > 1:
            # 如果长度不一致，需要 padding
            padded_embeds = []
            for emb in processed_embeds:
                pad_len = max_len - emb.shape[0]
                if pad_len > 0:
                    # 使用零向量进行 padding
                    pad_emb = torch.zeros(pad_len, emb.shape[1], device=model_device, dtype=emb.dtype)
                    emb = torch.cat([emb, pad_emb], dim=0)
                padded_embeds.append(emb)
            return torch.stack(padded_embeds, dim=0)  # [B, N, D]
        else:
            # 所有 embedding 都是单个向量，直接 stack
            return torch.stack(processed_embeds, dim=0)  # [B, 1, D]
    
    def _get_eos_pad_ids(self, eos_token_id: Optional[int]) -> Tuple[Any, int]:
        """获取 eos 和 pad token ID"""
        eos_id = eos_token_id
        if eos_id is None:
            eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = getattr(self.model.llm.config, "eos_token_id", None)
        if isinstance(eos_id, (list, tuple)):
            pass
        elif eos_id is None:
            eos_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eos_id is None or eos_id < 0:
                eos_id = 0
        
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None or pad_id < 0:
            pad_id = eos_id if not isinstance(eos_id, (list, tuple)) else (eos_id[0] if eos_id else 0)
        
        return eos_id, pad_id
    
    def _call_model_generate(
        self,
        input_ids: Optional[torch.Tensor],
        input_embeds: Optional[torch.Tensor],  # 额外的 embedding（类似 GVP 虚拟步）
        attention_mask: torch.Tensor,
        realtime_mol: bool,
        verbose_logging: bool,
        max_text_length_for_detection: int,
        gen_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """调用模型生成
        现在支持同时传入 input_ids 和 input_embeds（额外的 embedding）
        - input_ids: 用于 token 检测
        - input_embeds: 作为额外 embedding 插入（类似 GVP 虚拟步）
        """
        if realtime_mol:
            gen_kwargs["enable_thinking"] = self.enable_thinking
            # 同时传入 input_ids 和 inputs_embeds（如果有额外的 embedding）
            return self.model.generate(
                input_ids=input_ids,
                inputs_embeds=input_embeds,  # 额外的 embedding（类似 GVP 虚拟步）
                attention_mask=attention_mask,
                realtime_mol=True,
                verbose_logging=verbose_logging,
                max_text_length_for_detection=max_text_length_for_detection,
                **gen_kwargs
            )
        else:
            # 非 realtime_mol 模式：如果同时有 input_ids 和 input_embeds，优先使用 input_ids
            if input_ids is not None:
                return self.model.llm.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )
            elif input_embeds is not None:
                return self.model.llm.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )
            else:
                raise ValueError("必须提供 input_ids 或 input_embeds")
    
    def _decode_generated_ids(
        self,
        out_ids: torch.Tensor,
        prompt_lens: List[int],
        skip_special_tokens: bool,
    ) -> Tuple[List[str], List[str]]:
        """解码生成的 token IDs 为文本"""
        results = []
        raw_outputs = []
        
        for i in range(out_ids.size(0)):
            start = prompt_lens[i]
            gen_ids = out_ids[i, start:]
            
            text = self.tokenizer.decode(
                gen_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=True
            ).strip()
            
            results.append(text)
            raw_outputs.append(text)
        
        return results, raw_outputs
    
    def _postprocess_special_tasks(
        self,
        results: List[str],
        task_type: Optional[str],
        realtime_mol: bool,
        verbose_logging: bool,
        src_smiles: Optional[str],
    ) -> List[str]:
        """处理特殊任务（molecule_generation 等）"""
        if not results:
            return results
        
        assistant_text = results[0]
        
        # 如果输出恰好就是占位符（没有其他内容），说明模型只输出了占位符token
        if assistant_text.strip() in ("<think>", "<think>"):
            assistant_text = ""
            results[0] = assistant_text
        
        # 如果是分子生成任务，使用diffusion生成最终分子
        if (task_type == "molecule_generation" or task_type == "molecule_editing") and realtime_mol:
            if verbose_logging:
                print(f"\n[Molecule Generation] 📝 LLM 生成的描述:")
                print(f"{assistant_text}")
                print()
            
            assert self.ldmol is not None, "LDMolInferer is not initialized"
            if verbose_logging:
                print(f"[Molecule Generation] 🟣 开始使用 Diffusion 从描述生成 SMILES...")
            try:
                pattern = r"<smiles_description>(.*?)</smiles_description>"
                match = re.search(pattern, assistant_text, re.DOTALL)
                if match:
                    smiles_description = match.group(1).strip()
                else:
                    raise ValueError(f"No smiles description found in the assistant text: {assistant_text}")

                if task_type == "molecule_generation":
                    generated_smiles = self.ldmol.generate_molecule(
                        description=assistant_text,
                        qwen=self.model.llm,
                        qwen_tokenizer=self.tokenizer,
                    )
                elif task_type == "molecule_editing":
                    generated_smiles = self.ldmol.edit_molecule(
                        prompt=assistant_text,
                        src_smiles=src_smiles,
                    )
                    raise NotImplementedError("Molecule editing is not implemented yet")

                if generated_smiles:
                    if verbose_logging:
                        print(f"[Molecule Generation] ✅ Diffusion 生成 SMILES: {generated_smiles}")
                    results[0] = generated_smiles
                else:
                    if verbose_logging:
                        print(f"[Molecule Generation] ❌ Diffusion 未能生成 SMILES，返回描述")
            except Exception as e:
                if verbose_logging:
                    print(f"[Molecule Generation] ❌ 错误: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            if verbose_logging:
                print(f"[Molecule Generation] ⚠️  LDMol components 不可用，返回描述")
        
        return results

    # ------------------------ 对外 API ------------------------
    def load(self, cfg: Dict[str, Any]) -> None:
        """
        cfg 示例：
        {
          "ckpt_dir": "...",  # 根目录，应包含 llm/ 和 extras/ 子目录
          "device": "cuda:0",  # 单卡模式时使用，多卡模式可忽略
          "device_map": "auto" | None | dict,  # 多卡模式：None=单卡, "auto"=自动分配, dict=手动指定
          "dtype": "bf16" | "fp32",
          "token_classifier_path": "...",  # token classifier 权重路径
        }
        """
        self.loaded_cfg = cfg
        ckpt_dir = cfg["ckpt_dir"]
        self.device = cfg.get("device", self.device)
        
        # 是否启用 thinking 模式（默认关闭）
        self.enable_thinking = cfg.get("enable_thinking", False)
        
        # 检查是否使用多GPU
        device_map = cfg.get("device_map", None)
        self.use_multi_gpu = device_map is not None and device_map != "cpu"
        
        # 如果没有指定device_map，检查是否有多张GPU可用
        if device_map is None and torch.cuda.device_count() > 1:
            # 默认使用单卡，保持向后兼容
            logger.info(f"检测到 {torch.cuda.device_count()} 张GPU，但未启用多卡模式。"
                        f"要启用多卡推理，请设置 device_map='auto'")

        # 使用 model_init 的逻辑来处理 checkpoint 加载
        from modules.model_init import (
            init_tokenizer, init_llm, init_model, 
            load_model_weights_from_checkpoint_dir
        )
        from pathlib import Path
        
        # 检查 checkpoint 目录结构
        ckpt_path = Path(ckpt_dir)
        llm_dir = ckpt_path / "llm"
        extras_dir = ckpt_path / "extras"
        has_llm_dir = llm_dir.exists() and llm_dir.is_dir()
        has_extras_dir = extras_dir.exists() and extras_dir.is_dir()
        
        # 检查是否需要拆分 checkpoint（如果有 pytorch_model.bin 或 model.safetensors 但没有 llm 目录）
        needs_split = False
        if not has_llm_dir:
            bin_path = ckpt_path / "pytorch_model.bin"
            safetensors_path = ckpt_path / "model.safetensors"
            if bin_path.exists() or safetensors_path.exists():
                needs_split = True
                logger.info(f"📦 检测到混合 checkpoint，需要拆分: {ckpt_dir}")
                try:
                    split_script_path = Path(__file__).parent / "scripts" / "ckpt" / "split_llm_extras.py"
                    if split_script_path.exists():
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("split_llm_extras", split_script_path)
                        split_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(split_module)
                        success = split_module.split_checkpoint(str(ckpt_path), str(ckpt_path))
                        if success:
                            logger.info(f"✅ Checkpoint 拆分完成")
                            has_llm_dir = (ckpt_path / "llm").exists()
                            has_extras_dir = (ckpt_path / "extras").exists()
                        else:
                            logger.warning(f"⚠️ Checkpoint 拆分失败，尝试继续加载...")
                except Exception as e:
                    logger.warning(f"⚠️ 自动拆分失败: {e}")
        
        # 加载 tokenizer
        # 优先从 checkpoint 根目录加载（tokenizer 通常保存在根目录）
        tokenizer_dir = str(ckpt_path)
        tokenizer_loaded = False
        
        # 检查根目录是否有 tokenizer 文件
        tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
        has_tokenizer_in_root = any((ckpt_path / f).exists() for f in tokenizer_files)
        
        if has_tokenizer_in_root:
            logger.info(f"从 checkpoint 根目录加载 tokenizer: {tokenizer_dir}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True, trust_remote_code=True)
                tokenizer_loaded = True
            except Exception as e:
                logger.warning(f"从根目录加载 tokenizer 失败: {e}")
        
        # 如果根目录加载失败，尝试从 llm 目录加载
        if not tokenizer_loaded and has_llm_dir:
            tokenizer_dir = str(llm_dir)
            logger.info(f"尝试从 checkpoint 的 llm 目录加载 tokenizer: {tokenizer_dir}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True, trust_remote_code=True)
                tokenizer_loaded = True
            except Exception as e:
                logger.warning(f"从 llm 目录加载 tokenizer 失败: {e}")
        
        # 如果都失败，尝试使用 base_llm_path
        if not tokenizer_loaded:
            base_llm_path = cfg.get("base_llm_path")
            if base_llm_path:
                logger.info(f"回退到 base LLM 路径加载 tokenizer: {base_llm_path}")
                self.tokenizer = init_tokenizer(base_llm_path, mol_token="<mol>")
            else:
                raise RuntimeError(f"无法加载 tokenizer，请检查 checkpoint 目录或提供 base_llm_path")
        
        self._ensure_special_tokens()

        # 精度
        dtype_flag = str(cfg.get("dtype", "bf16")).lower()
        torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and "bf16" in dtype_flag) else torch.float32

        # 加载模型
        if has_llm_dir:
            # 有 llm 目录，优先使用 from_pretrained 加载
            try:
                # 准备 Layer2 配置
                layer2_config = cfg.get("layer2")
                use_layer2 = cfg.get("train", {}).get("use_layer2", False)
                
                if self.use_multi_gpu:
                    logger.info(f"使用多GPU模式，device_map={device_map}")
                    self.model = MolAwareCausalLM.from_pretrained(
                        save_directory=str(ckpt_path),
                        tokenizer=self.tokenizer,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        layer2_config=layer2_config,
                        use_layer2=use_layer2,
                    )
                else:
                    self.model = MolAwareCausalLM.from_pretrained(
                        save_directory=str(ckpt_path),
                        tokenizer=self.tokenizer,
                        torch_dtype=torch_dtype,
                        device_map=None,
                        layer2_config=layer2_config,
                        use_layer2=use_layer2,
                    ).to(self.device)
            except (ValueError, RuntimeError) as e:
                # 如果 from_pretrained 失败（比如 torch 版本问题），使用 model_init 逻辑
                error_msg = str(e)
                if "torch.load" in error_msg or "v2.6" in error_msg or "CVE" in error_msg:
                    logger.warning(f"from_pretrained 失败（torch版本限制）: {e}")
                    logger.info("使用 model_init 逻辑作为备选方案")
                else:
                    logger.warning(f"from_pretrained 失败: {e}，使用 model_init 逻辑")
                
                # 使用 model_init 逻辑加载
                # 如果 llm_dir 不存在或没有模型文件，尝试从根目录加载
                actual_llm_dir = str(llm_dir)
                if not has_llm_dir or not os.path.exists(actual_llm_dir):
                    # llm 目录不存在，尝试从根目录加载
                    actual_llm_dir = str(ckpt_path)
                    logger.info(f"llm 目录不存在，从根目录加载: {actual_llm_dir}")
                
                # 直接使用 init_llm，它会处理 torch 版本限制和自动转换
                logger.info(f"使用 init_llm 从 {actual_llm_dir} 加载 base LLM（处理 torch 版本限制）")
                base_llm = init_llm(actual_llm_dir, self.tokenizer, "bf16" in dtype_flag, self.device)
                
                # 构建简化的 config
                simple_cfg = {
                    "tokens": {"mol_token": "<mol>"},
                    "train": {
                        "use_diffusion": False,
                        "use_layer2": cfg.get("train", {}).get("use_layer2", False),  # 保留 Layer2 配置
                    },
                    "network": {},
                    "diffusion": {},
                }
                # 保留 Layer2 配置
                if "layer2" in cfg:
                    simple_cfg["layer2"] = cfg["layer2"]
                self.model = init_model(simple_cfg, self.tokenizer, base_llm, self.device)
                # 加载 extras（如果存在）
                if has_extras_dir:
                    load_model_weights_from_checkpoint_dir(self.model, str(ckpt_path), self.device)
        else:
            # 没有 llm 目录，可能是纯 LLM checkpoint 或需要拆分的混合 checkpoint
            base_llm_path = cfg.get("base_llm_path")
            
            # 检查 checkpoint 根目录是否有模型文件（可能是纯 LLM checkpoint）
            has_model_files = any([
                (ckpt_path / "pytorch_model.bin").exists(),
                (ckpt_path / "model.safetensors").exists(),
                (ckpt_path / "model.safetensors.index.json").exists(),
            ])
            
            if not base_llm_path:
                if has_model_files:
                    # 纯 LLM checkpoint：从根目录加载
                    logger.info(f"检测到纯 LLM checkpoint（无 llm/ 目录），从根目录加载: {ckpt_dir}")
                    base_llm_path = str(ckpt_path)
                elif ckpt_path.exists():
                    # 尝试拆分混合 checkpoint
                    logger.info(f"checkpoint 没有 llm 目录，尝试从根目录加载并拆分")
                    # 使用 load_model_weights_from_checkpoint_dir 的逻辑，它会自动拆分
                    # 但我们需要先有一个 base_llm
                    # 如果没有 base_llm_path，无法继续
                    raise RuntimeError(
                        f"checkpoint 没有 llm 目录，且未提供 base_llm_path。"
                        f"请提供 base_llm_path 或确保 checkpoint 包含 llm/ 子目录。"
                    )
                else:
                    raise RuntimeError(f"checkpoint 目录不存在: {ckpt_dir}")
            
            logger.info(f"使用 model_init 逻辑从 base_llm_path 加载: {base_llm_path}")
            base_llm = init_llm(base_llm_path, self.tokenizer, "bf16" in dtype_flag, self.device)
            
            # 构建简化的 config
            # 对于纯 LLM checkpoint，禁用 GNN（因为没有 GVP 和 mol_adapter）
            simple_cfg = {
                "tokens": {"mol_token": "<mol>"},
                "train": {
                    "use_diffusion": False,
                    "use_offline_spans": False,  # 禁用 GNN
                    "use_layer2": cfg.get("train", {}).get("use_layer2", False),  # 保留 Layer2 配置
                },
                "network": {},
                "diffusion": {},
                "paths": {"checkpoint_dir": str(ckpt_path)} if ckpt_path.exists() else {},
            }
            # 保留 Layer2 配置
            if "layer2" in cfg:
                simple_cfg["layer2"] = cfg["layer2"]
            self.model = init_model(simple_cfg, self.tokenizer, base_llm, self.device)
            
            # 如果 checkpoint 存在且有 extras 目录，尝试加载权重（会自动处理拆分）
            # 但对于纯 LLM checkpoint，不会有 extras 目录，所以这里不会加载 GVP/mol_adapter
            if ckpt_path.exists() and has_extras_dir:
                load_model_weights_from_checkpoint_dir(self.model, str(ckpt_path), self.device)
            elif ckpt_path.exists() and not has_extras_dir:
                logger.info(f"纯 LLM checkpoint（无 extras 目录），跳过 GVP/mol_adapter 加载")

        # 加载 token_classifier_head
        # 优先从 extras 目录查找，如果没有则使用配置指定的路径
        token_classifier_path = None
        extras_token_classifier = ckpt_path / "extras" / "token_classifier.pt"
        if extras_token_classifier.exists() and extras_token_classifier.is_file():
            token_classifier_path = str(extras_token_classifier)
            logger.info(f"✅ Found token_classifier in extras directory: {token_classifier_path}")
        else:
            # 如果没有找到，使用配置中指定的路径
            token_classifier_path = cfg.get("token_classifier_path")
            if token_classifier_path:
                logger.info(f"Using token_classifier_path from config: {token_classifier_path}")
        
        if token_classifier_path and os.path.isfile(token_classifier_path):
            try:
                hidden_size = self.model.llm.config.hidden_size
                model_dtype = next(self.model.llm.parameters()).dtype
                
                # 确定token classifier应该放在哪个设备
                if self.use_multi_gpu:
                    # 放在LLM第一个层的设备上
                    token_classifier_device = next(self.model.llm.parameters()).device
                else:
                    token_classifier_device = self.device
                
                # 先检查 checkpoint 中的 hidden_size 是否匹配
                try:
                    ckpt = torch.load(token_classifier_path, map_location="cpu")
                    # 处理多种可能的 checkpoint 结构
                    if isinstance(ckpt, dict):
                        if "state_dict" in ckpt:
                            raw_sd = ckpt["state_dict"]
                        elif "head_state_dict" in ckpt:
                            # 新的格式：state_dict 存储在 head_state_dict 中
                            raw_sd = ckpt["head_state_dict"]
                            # 同时检查 hidden_size 是否匹配
                            if "hidden_size" in ckpt:
                                ckpt_hidden_size = ckpt["hidden_size"]
                                if ckpt_hidden_size != hidden_size:
                                    logger.warning(
                                        f"⚠️ Token classifier hidden_size mismatch: "
                                        f"checkpoint={ckpt_hidden_size}, model={hidden_size}. "
                                        f"Skipping token_classifier_head loading."
                                    )
                                    setattr(self.model, "token_classifier_head", None)
                                    raise ValueError("Hidden size mismatch")  # 触发外层 except 处理
                        elif "model_state_dict" in ckpt:
                            raw_sd = ckpt["model_state_dict"]
                        else:
                            # 检查是否顶层就是 state_dict（所有值都是 tensor）
                            if all(isinstance(v, torch.Tensor) for v in ckpt.values() if v is not None):
                                raw_sd = ckpt
                            else:
                                # 如果顶层有非 tensor 值，尝试查找可能的 state_dict
                                raw_sd = None
                                for key in ["head_state_dict", "state_dict", "model_state_dict", "classifier"]:
                                    if key in ckpt and isinstance(ckpt[key], dict):
                                        raw_sd = ckpt[key]
                                        break
                                if raw_sd is None:
                                    raise ValueError(f"Could not find state_dict in checkpoint. Available keys: {list(ckpt.keys())}")
                    else:
                        raw_sd = ckpt
                    
                    # 查找第一个线性层的权重来确定原始 hidden_size（如果还没有从顶层检查过）
                    ckpt_hidden_size = None
                    # 如果已经从顶层 ckpt 中读取了 hidden_size，就不需要再从 state_dict 中推断
                    if isinstance(ckpt, dict) and "hidden_size" in ckpt:
                        ckpt_hidden_size = ckpt["hidden_size"]
                    else:
                        # 从 state_dict 中推断 hidden_size
                        for key, value in raw_sd.items():
                            if isinstance(value, torch.Tensor) and "weight" in key and len(value.shape) == 2:
                                # 通常是第一层的输入维度
                                ckpt_hidden_size = value.shape[1]
                                break
                    
                    if ckpt_hidden_size is not None and ckpt_hidden_size != hidden_size:
                        logger.warning(
                            f"⚠️ Token classifier hidden_size mismatch: "
                            f"checkpoint={ckpt_hidden_size}, model={hidden_size}. "
                            f"Skipping token_classifier_head loading. "
                            f"Model will use text-matching fallback for entity detection."
                        )
                        setattr(self.model, "token_classifier_head", None)
                    else:
                        # 维度匹配，继续加载
                        # 从 checkpoint 中推断中间层维度（如果可能）
                        intermediate_dim = 128  # 默认值
                        if isinstance(ckpt, dict) and "head_state_dict" in ckpt:
                            # 尝试从 state_dict 中推断中间层维度
                            for key, value in raw_sd.items():
                                if isinstance(value, torch.Tensor) and "0.weight" in key and len(value.shape) == 2:
                                    # 第一层 Linear 的输出维度就是中间层维度
                                    intermediate_dim = value.shape[0]
                                    logger.info(f"📐 Inferred intermediate_dim from checkpoint: {intermediate_dim}")
                                    break
                        
                        # 如果配置中指定了中间层维度，使用配置的值
                        if "token_classifier_intermediate_dim" in cfg:
                            intermediate_dim = cfg["token_classifier_intermediate_dim"]
                            logger.info(f"📐 Using intermediate_dim from config: {intermediate_dim}")
                        
                        token_head = nn.Sequential(
                            nn.Linear(hidden_size, intermediate_dim),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(intermediate_dim, 2)
                        ).to(device=token_classifier_device, dtype=model_dtype)
                        
                        # 使用与 init_offline_token_classifier 相同的逻辑来清理 state_dict
                        from collections import OrderedDict
                        
                        # 清理 state_dict：移除可能的 module. 和 net. 前缀
                        clean_sd = OrderedDict()
                        for k, v in raw_sd.items():
                            name = k
                            # 移除 module. 前缀（DDP 训练产生的）
                            if name.startswith("module."):
                                name = name[7:]
                            # 移除 net. 前缀（某些 checkpoint 格式）
                            if name.startswith("net."):
                                name = name[4:]
                            clean_sd[name] = v
                        
                        # 构建最终的 state_dict，尝试多种可能的 key 格式
                        # 只处理 tensor 值，跳过字符串等其他类型
                        final_sd = OrderedDict()
                        for k, v in clean_sd.items():
                            # 只处理 tensor 值
                            if not isinstance(v, torch.Tensor):
                                continue
                            
                            # 尝试多种可能的 key 格式
                            if k.startswith("classifier."):
                                final_sd[k.replace("classifier.", "")] = v
                            elif k.startswith("token_classifier."):
                                final_sd[k.replace("token_classifier.", "")] = v
                            elif not "." in k or k.count(".") <= 1:
                                # 如果 key 看起来像是分类器的参数（没有太多层级），直接使用
                                final_sd[k] = v
                            elif "0.weight" in k or "0.bias" in k or "3.weight" in k or "3.bias" in k:
                                # 匹配 Sequential 模型的 key（0 是第一层 Linear，3 是第二层 Linear）
                                final_sd[k] = v
                        
                        # 如果还是没有找到，尝试更宽松的匹配
                        if not final_sd:
                            # 查找所有包含 weight 或 bias 的 key，并尝试匹配 Sequential 的索引格式
                            for k, v in clean_sd.items():
                                # 只处理 tensor 值
                                if not isinstance(v, torch.Tensor):
                                    continue
                                
                                if "weight" in k or "bias" in k:
                                    parts = k.split(".")
                                    # 尝试匹配 Sequential 的索引格式（0, 1, 2, 3）
                                    # 例如：0.weight, 0.bias, 3.weight, 3.bias
                                    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1] in ["weight", "bias"]:
                                        idx = int(parts[-2])
                                        if idx in [0, 3]:  # 只匹配第一层和最后一层 Linear
                                            final_sd[k] = v
                                    elif len(parts) == 1:
                                        # 直接是 weight 或 bias（无层级）
                                        final_sd[k] = v
                        
                        if not final_sd:
                            logger.warning(f"⚠️ No matching keys found in checkpoint. Available keys: {list(clean_sd.keys())[:20]}")
                            raise ValueError(f"No matching keys found in checkpoint. Available keys: {list(clean_sd.keys())[:20]}")
                        
                        logging.debug(f"📋 Matched {len(final_sd)} keys: {list(final_sd.keys())}")
                        
                        # 转换数据类型并加载（只处理 tensor，跳过非 tensor 值）
                        final_sd_clean = OrderedDict()
                        for k, v in final_sd.items():
                            if isinstance(v, torch.Tensor):
                                final_sd_clean[k] = v.to(dtype=model_dtype)
                            else:
                                logger.warning(f"⚠️ Skipping non-tensor value for key '{k}': type={type(v)}")
                        
                        if not final_sd_clean:
                            raise ValueError(f"No valid tensor values found in final_sd. Keys: {list(final_sd.keys())}")
                        
                        token_head.load_state_dict(final_sd_clean, strict=True)
                        token_head.eval()
                        
                        setattr(self.model, "token_classifier_head", token_head)
                        for p in self.model.token_classifier_head.parameters():
                            p.requires_grad = False
                        
                        logger.info(f"✅ Loaded token_classifier_head from {token_classifier_path}")
                except Exception as inner_e:
                    raise inner_e
            except Exception as e:
                logger.warning(
                    f"⚠️ Failed to load token_classifier_head from {token_classifier_path}: {e}. "
                    f"Model will use text-matching fallback for entity detection. "
                    f"This is usually fine and will not affect generation quality."
                )
                # 确保 token_classifier_head 是 None，这样会使用 fallback 方法
                setattr(self.model, "token_classifier_head", None)

        self.model.debug = bool(cfg.get("debug", False))
        for p in self.model.parameters():
            p.requires_grad = False
        logger.info(f'self.model #parameters: {sum(p.numel() for p in self.model.parameters())}, #trainable: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')

        # 严格校验 tokenizer/embedding 一致性
        self._sync_vocab_and_embeddings(strict=True)
        
        ##############################
        # NOTE: LDMOL Init (lazy / opt-in)
        # TODO: 未来将LDMol ckpt应放在 ckpt_path目录下
        self.ldmol = None  # 默认不加载，避免 ChemBench 这种纯文本评测 OOM

        enable_ldmol = bool(cfg.get("enable_ldmol", False))  # 默认 False
        ldmol_device = cfg.get("ldmol_device", "cpu")        # 默认放 CPU，更安全

        if enable_ldmol and LDMOL_ENABLED:
            logger.info(f"🧪 Enable LDMolInferer: device={ldmol_device}")
            self.ldmol = LDMolInferer(device=ldmol_device)
        else:
            logger.info("🧪 LDMolInferer unavailable")
        ##############################

        if self.use_multi_gpu:
            logger.info(f"✅ Model & tokenizer loaded from {ckpt_dir} on multiple GPUs (device_map={device_map}).")
        else:
            logger.info(f"✅ Model & tokenizer loaded from {ckpt_dir} on {self.device}.")
        
        logger.info(f"ℹ️  Thinking 模式: {'启用' if self.enable_thinking else '禁用'} (enable_thinking={self.enable_thinking})")
        
        # 检查 Layer2 是否可用
        self.use_layer2 = hasattr(self.model, 'layer2_inferer') and self.model.layer2_inferer is not None
        if self.use_layer2:
            logger.info("✅ Layer2 已启用，支持反应产率预测 pipeline")
        else:
            logger.info("ℹ️  Layer2 未启用，使用标准生成模式")

    @torch.no_grad()
    def generate(
        self,
        prompt,
        *,
        add_dialog_wrapper: bool = True,
        realtime_mol: bool = True,
        max_new_tokens: int = 256,
        max_tokens: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.15,  # 提高默认值以减少重复（从1.05提高到1.15）
        no_repeat_ngram_size: int = 3,  # 防止重复n-gram（默认3，即防止3个token的重复）
        skip_special_tokens: bool = True,  # 默认跳过特殊token（官方风格）
        eos_token_id: Optional[int] = None,
        return_ids: bool = False,
        verbose_logging: bool = False,  # 控制详细日志输出
        max_text_length_for_detection: int = 4096,  # 超出此长度跳过实体检测（但不停止生成），支持few-shot等长prompt
        use_diffusion_as_smiles_supplement: bool = False,  # 如果不是SMILES就调用diffusion得到SMILES
        task_type: Optional[str] = None, 
        src_smiles: str = None,  # TODO: 分子编辑任务的输入smiles
        use_layer2_pipeline: bool = False,  # 是否使用 Layer2 pipeline
        extract_reactant_fn: Optional[callable] = None,  # 自定义提取反应物 SMILES 的函数（用于 Layer2）
        return_intermediate: bool = False,  # 是否返回中间结果（用于 Layer2）
        force_use_layer2: Optional[bool] = None,  # 强制使用 Layer2（用于 Layer2）
        extra_embeddings: Optional[torch.Tensor | List[torch.Tensor]] = None,  # 额外的 embedding 追加到序列末尾（像 GVP 一样）
    ):
        """
        生成文本，支持实时分子标注和 GVP 调用。
        
        推理时的GNN流程（与训练时一致）：
        1. 检测到 <mol>...</mol> 标签，提取SMILES
        2. SMILES -> GVP encoder -> 图embedding
        3. 图embedding -> mol_adapter -> LLM维度embedding
        4. 这个embedding追加到序列中（不作为token显示，但作为hidden state存在）
        5. 继续生成后续文本
        
        注意：GNN的embedding会一直存在于hidden states中，影响后续生成，
        但不会生成对应的token，这是设计上的特性。
        
        Args:
            use_layer2_pipeline: 如果为 True，使用 Layer2 pipeline（两轮生成 + Layer2 预测）
            其他参数同 generate_with_layer2
        """
        assert self.model is not None and self.tokenizer is not None, "请先调用 load(config) 完成初始化。"
        
        # 如果启用 Layer2 pipeline，调用 generate_with_layer2
        if use_layer2_pipeline:
            return self.generate_with_layer2(
                prompt=prompt,
                add_dialog_wrapper=add_dialog_wrapper,
                realtime_mol=realtime_mol,
                max_new_tokens=max_new_tokens,
                max_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                task_type=task_type,
                extract_reactant_fn=extract_reactant_fn,
                return_intermediate=return_intermediate,
                force_use_layer2=force_use_layer2,
            )

        # 1) 统一成 batch（官方风格：batch/single 用同一套逻辑）
        is_batched = isinstance(prompt, (list, tuple))
        prompts: List[str] = list(prompt) if is_batched else [prompt]
        
        # 2) 选择 system message
        system_msg = self._get_system_message(task_type, realtime_mol)
        logger.info(f"task_type: {task_type}, use system_msg: {system_msg}")
        
        # 3) 编码 prompts
        batch_enc = self._encode_prompts(prompts, system_msg, add_dialog_wrapper)
        
        # 4) 放到正确 device（兼容 device_map="auto"）
        model_device = _get_model_device(self.model.llm)
        input_ids = batch_enc["input_ids"].to(model_device)
        attention_mask = batch_enc.get("attention_mask", torch.ones_like(input_ids)).to(model_device)
        
        # 5) 处理额外的 embedding（如果有）
        # 现在支持同时传入 input_ids 和 inputs_embeds（额外的 embedding）
        # input_ids 用于 token 检测，inputs_embeds 作为额外 embedding 插入（类似 GVP 虚拟步）
        extra_inputs_embeds = None
        
        if extra_embeddings is not None:
            extra_inputs_embeds = self._prepare_extra_embeddings(
                extra_embeddings, model_device, len(prompts)
            )
        
        # 6) 获取 eos/pad IDs
        eos_id, pad_id = self._get_eos_pad_ids(eos_token_id)
        
        # 7) 计算 prompt_len
        prompt_lens = [input_ids.shape[1] for _ in range(len(prompts))]
        
        if verbose_logging:
            print(f"[DEBUG] batch={len(prompts)}, input_ids={tuple(input_ids.shape)}, device={model_device}")
            print(f"[DEBUG] eos_id={eos_id}, pad_id={pad_id}")
            if extra_inputs_embeds is not None:
                print(f"[DEBUG] Using extra inputs_embeds with shape {extra_inputs_embeds.shape}")
        
        # 8) 构建生成参数
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        
        if max_tokens is not None:
            gen_kwargs["max_tokens"] = max_tokens
        
        # 9) 调用模型生成
        # 如果 realtime_mol=True 但 batch size >= 1，逐个处理
        if realtime_mol and len(prompts) >= 1:
            logger.info(f"ℹ️  realtime_mol 仅支持 batch=1，当前 batch={len(prompts)}，将逐个处理")
            # 逐个处理每个 prompt
            results = []
            raw_outputs = []
            for i, prompt in enumerate(prompts):
                # 重新编码单个 prompt
                single_prompt_enc = self._encode_prompts([prompt], system_msg, add_dialog_wrapper)
                single_input_ids = single_prompt_enc["input_ids"].to(model_device)
                single_attention_mask = single_prompt_enc.get("attention_mask", torch.ones_like(single_input_ids)).to(model_device)
                
                # 处理单个 prompt 的 extra_embeddings（如果有）
                single_extra_inputs_embeds = None
                
                if extra_embeddings is not None:
                    # 如果 extra_embeddings 是列表，取对应的元素；否则使用同一个
                    if isinstance(extra_embeddings, list):
                        single_extra_emb = extra_embeddings[i] if i < len(extra_embeddings) else extra_embeddings[0]
                    else:
                        single_extra_emb = extra_embeddings
                    
                    single_extra_inputs_embeds = self._prepare_extra_embeddings(
                        single_extra_emb, model_device, 1
                    )
                
                single_prompt_len = single_input_ids.shape[1]
                
                # 调用模型生成（同时传入 input_ids 和 inputs_embeds）
                single_out_ids = self._call_model_generate(
                    input_ids=single_input_ids,
                    input_embeds=single_extra_inputs_embeds,  # 额外的 embedding（类似 GVP 虚拟步）
                    attention_mask=single_attention_mask,
                    realtime_mol=realtime_mol,  # 保持 realtime_mol=True
                    verbose_logging=verbose_logging,
                    max_text_length_for_detection=max_text_length_for_detection,
                    gen_kwargs=gen_kwargs,
                )
                
                # 解码单个结果
                single_results, single_raw_outputs = self._decode_generated_ids(
                    single_out_ids, [single_prompt_len], skip_special_tokens
                )
                results.append(single_results[0] if single_results else "")
                raw_outputs.append(single_raw_outputs[0] if single_raw_outputs else "")
        else:
            # batch_size=1 或 realtime_mol=False，正常处理
            # 同时传入 input_ids 和 inputs_embeds（如果有额外的 embedding）
            out_ids = self._call_model_generate(
                input_ids=input_ids,
                input_embeds=extra_inputs_embeds,  # 额外的 embedding（类似 GVP 虚拟步）
                attention_mask=attention_mask,
                realtime_mol=realtime_mol,
                verbose_logging=verbose_logging,
                max_text_length_for_detection=max_text_length_for_detection,
                gen_kwargs=gen_kwargs,
            )
            
            # 10) 解码结果
            results, raw_outputs = self._decode_generated_ids(
                out_ids, prompt_lens, skip_special_tokens
            )
        
        # 11) 保存 last_raw_outputs
        self._last_raw_outputs = raw_outputs
        
        # 12) 处理特殊任务
        if not is_batched:
            results = self._postprocess_special_tasks(
                results, task_type, realtime_mol, verbose_logging, src_smiles
            )
        
        # 返回结果
        return results if is_batched else results[0]
    
    @torch.no_grad()
    def generate_with_layer2(
        self,
        prompt: str,
        *,
        add_dialog_wrapper: bool = True,
        realtime_mol: bool = True,  # 两轮生成都使用此参数
        max_new_tokens: int = 256,
        max_tokens: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 3,
        task_type: Optional[str] = None,
        extract_reactant_fn: Optional[callable] = None,  # 自定义提取反应物 SMILES 的函数
        return_intermediate: bool = False,  # 是否返回中间结果
        force_use_layer2: Optional[bool] = None,  # 强制使用 Layer2（None 时根据 task_type 自动判断）
    ) -> str | Dict[str, Any]:
        """
        Layer2 Pipeline: query -> LLM -> Layer2 -> LLM
        
        Pipeline 流程:
        1. 第一轮 LLM 生成，提取反应物 SMILES 和 amount_info
        2. GVP 获取对应 embedding
        3. Layer2 预测 [yield_bin, embedding]（仅当任务需要时）
        4. 将 embedding 追加到 prompt 后（像 GVP 一样），构建增强 prompt（包含 yield_bin 信息）
        5. 第二轮 LLM 生成，输出最终结果
        
        任务类型支持:
        - "yield_prediction": 产率预测 - 使用 Layer2
        - "product_yield_prediction": 产物+产率预测 - 使用 Layer2
        - "product_prediction": 产物预测 - 不使用 Layer2
        - "reaction_prediction": 反应预测 - 不使用 Layer2
        - 其他或 None: 根据 force_use_layer2 参数决定
        
        Args:
            prompt: 输入查询
            extract_reactant_fn: 自定义函数，从第一轮输出中提取反应物信息
                                函数签名: (text: str) -> Dict[str, Any]
                                返回格式: {
                                    "reactant_smiles": str | List[str],
                                    "amount_info": Optional[Dict | List[Dict]]
                                }
            return_intermediate: 是否返回中间结果
            force_use_layer2: 强制使用 Layer2（None 时根据 task_type 自动判断）
            ... (其他参数同 generate 方法)
        
        Returns:
            如果 return_intermediate=False: 返回最终生成的文本
            如果 return_intermediate=True: 返回字典 {
                "first_response": str,
                "layer2_info": {
                    "yield_bin": int,
                    "yield_reg": float,
                    "embedding": torch.Tensor,
                } | None,  # 如果未使用 Layer2 则为 None
                "final_response": str
            }
        """
        assert self.model is not None and self.tokenizer is not None, "请先调用 load(config) 完成初始化。"
        
        # 如果强制不使用 Layer2，回退到标准生成
        if force_use_layer2 is False:
            logger.info(f"强制不使用 Layer2，使用标准生成模式")
            return self.generate(
                prompt,
                add_dialog_wrapper=add_dialog_wrapper,
                realtime_mol=realtime_mol,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                task_type=task_type,
            )
        
        # 如果 Layer2 未启用，回退到标准生成
        if not self.use_layer2:
            logger.warning("Layer2 未启用，回退到标准生成模式")
            return self.generate(
                prompt,
                add_dialog_wrapper=add_dialog_wrapper,
                realtime_mol=realtime_mol,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                task_type=task_type,
            )
        
        logger.info(f"🔄 开始 Layer2 Pipeline (任务类型: {task_type})")
        
        # ===== 第一阶段：第一轮 LLM 生成，专门输出 JSON 格式的反应物信息 =====
        logger.info("📝 第一阶段：LLM 生成（JSON 格式提取反应物信息）")
        
        # 构建专门用于提取反应物信息的 prompt（要求输出 JSON）
        extraction_prompt = self._build_layer2_extraction_prompt(prompt)
        
        first_response = self.generate(
            extraction_prompt,
            add_dialog_wrapper=add_dialog_wrapper,
            max_new_tokens=max_new_tokens // 2,  # 第一轮生成较短
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            task_type=None,  # 第一轮不使用特殊任务类型，专注于 JSON 提取
            realtime_mol=False,  # 第一轮不需要 realtime_mol，只提取信息
        )
        
        if isinstance(first_response, list):
            first_response = first_response[0]
        
        logger.info(f"第一轮输出 (JSON): {first_response[:200]}...")
        
        # 提取分子信息和角色
        if extract_reactant_fn is not None:
            # 使用自定义提取函数
            extracted_info = extract_reactant_fn(first_response)
            if isinstance(extracted_info, dict):
                # 兼容新格式
                if "molecules" in extracted_info:
                    molecules_info = extracted_info["molecules"]
                else:
                    # 兼容旧格式
                    reactant_smiles = extracted_info.get("reactant_smiles", extracted_info.get("smiles"))
                    if isinstance(reactant_smiles, str):
                        reactant_smiles = [reactant_smiles]
                    molecules_info = [
                        {
                            "smiles": smi,
                            "role": "REACTANT",
                            "amount_info": extracted_info.get("amount_info")
                        }
                        for smi in reactant_smiles
                    ]
            else:
                # 兼容旧接口：直接返回 SMILES 字符串
                if isinstance(extracted_info, str):
                    extracted_info = [extracted_info]
                molecules_info = [
                    {"smiles": smi, "role": "REACTANT"}
                    for smi in extracted_info
                ]
        else:
            # 优先从 JSON 中解析
            parsed_json = self._parse_json_response(first_response)
            if parsed_json is not None:
                molecules_info = parsed_json.get("molecules", [])
            else:
                # JSON 解析失败，回退到默认提取逻辑
                logger.warning("⚠️  JSON 解析失败，使用默认提取逻辑")
                reactant_smiles = self._extract_reactant_smiles(first_response)
                if isinstance(reactant_smiles, str):
                    reactant_smiles = [reactant_smiles]
                molecules_info = [
                    {"smiles": smi, "role": "REACTANT"}
                    for smi in reactant_smiles
                ]
        
        if not molecules_info:
            logger.warning("⚠️  未能从第一轮输出中提取分子信息，回退到标准生成")
            if return_intermediate:
                return {
                    "first_response": first_response,
                    "layer2_info": None,
                    "final_response": first_response,
                }
            return first_response
        
        # 根据任务类型和角色信息，决定需要预测的内容
        # 如果是预测反应物，应该 mask 反应物部分；如果是预测产物，应该 mask 产物部分
        # 这里我们需要根据 task_type 来判断
        target_role = None
        if task_type:
            task_type_lower = task_type.lower()
            if "reactant" in task_type_lower or "reaction" in task_type_lower:
                # 预测反应物，需要 mask 反应物
                target_role = "REACTANT"
            elif "product" in task_type_lower:
                # 预测产物，需要 mask 产物
                target_role = "PRODUCT"
            else:
                # 默认预测反应物
                target_role = "REACTANT"
        else:
            # 默认预测反应物
            target_role = "REACTANT"
        
        logger.info(f"✅ 提取到 {len(molecules_info)} 个分子，目标角色: {target_role}")
        for i, mol in enumerate(molecules_info):
            logger.info(f"  分子 {i+1}: {mol.get('smiles', 'N/A')} (角色: {mol.get('role', 'N/A')})")
        
        # ===== 第二阶段：GVP + Layer2 预测 =====
        logger.info("🔬 第二阶段：GVP + Layer2 预测")
        
        try:
            # 根据角色筛选需要预测的分子
            # 如果目标是预测反应物，我们需要所有已知的分子（包括产物）来预测反应物
            # 如果目标是预测产物，我们需要所有已知的分子（包括反应物）来预测产物
            # 这里我们先使用所有分子，Layer2 会根据角色信息来处理
            
            # 提取所有分子的 SMILES 和相关信息
            all_smiles = [mol["smiles"] for mol in molecules_info]
            all_roles = [mol.get("role", "REACTANT") for mol in molecules_info]
            all_amount_info = [mol.get("amount_info") for mol in molecules_info]
            
            # 获取所有分子的 GVP embeddings
            gvp_embeddings = []
            for smi in all_smiles:
                gvp_emb = self.model.gvp_encoder.forward_from_smiles(smi)
                if gvp_emb is None:
                    raise ValueError(f"GVP encoder 返回 None for SMILES: {smi}")
                gvp_embeddings.append(gvp_emb.squeeze(0))  # [D]
            
            # 如果只有一个分子，保持兼容性
            if len(gvp_embeddings) == 1:
                gvp_embedding = gvp_embeddings[0]
                amount_info = all_amount_info[0]
            else:
                gvp_embedding = gvp_embeddings
                amount_info = all_amount_info if any(ai is not None for ai in all_amount_info) else None
            
            # Layer2 预测
            # 根据任务类型，决定哪些分子需要被 mask（需要预测的）
            # 如果任务是预测反应物，则 mask 反应物；如果任务是预测产物，则 mask 产物
            # 这里我们只传入需要预测的分子（被 mask 的），其他作为上下文
            
            # 筛选需要预测的分子（根据 target_role）
            target_molecules = [mol for mol in molecules_info if mol.get("role") == target_role]
            context_molecules = [mol for mol in molecules_info if mol.get("role") != target_role]
            
            if not target_molecules:
                logger.warning(f"⚠️  没有找到角色为 {target_role} 的分子，使用所有分子作为反应物")
                target_molecules = molecules_info
                target_smiles = all_smiles
                target_gvp_embeddings = gvp_embeddings
                target_amount_info = amount_info
            else:
                # 只使用目标角色的分子进行预测
                target_smiles = [mol["smiles"] for mol in target_molecules]
                target_indices = [i for i, mol in enumerate(molecules_info) if mol.get("role") == target_role]
                target_gvp_embeddings = [gvp_embeddings[i] for i in target_indices]
                target_amount_info = [all_amount_info[i] for i in target_indices]
                if len(target_amount_info) == 1:
                    target_amount_info = target_amount_info[0]
                elif not any(ai is not None for ai in target_amount_info):
                    target_amount_info = None
            
            logger.info(f"📊 预测目标: {len(target_molecules)} 个 {target_role} 分子")
            if context_molecules:
                logger.info(f"📊 上下文: {len(context_molecules)} 个其他角色分子")
            
            # 如果只有一个目标分子，保持兼容性
            if len(target_gvp_embeddings) == 1:
                target_gvp_embedding = target_gvp_embeddings[0]
            else:
                target_gvp_embedding = target_gvp_embeddings
            
            # Layer2 预测（传入需要预测的分子）
            layer2_output = self.model.layer2_inferer.predict(
                reactant_smiles=target_smiles,  # 传入需要预测的 SMILES
                gvp_embedding=target_gvp_embedding,
                amount_info=target_amount_info,
            )
            
            yield_bin = layer2_output['yield_bin']
            yield_reg = layer2_output['yield_reg']
            task_embedding_raw = layer2_output['embedding']  # 原始维度（256）
            
            # 通过 mol_adapter 将 embedding 映射到 LLM 维度（与 GVP 返回的格式一致）
            if hasattr(self.model, 'mol_adapter') and self.model.mol_adapter is not None:
                # 确保 dtype 一致
                mol_adapter_dtype = next(self.model.mol_adapter.parameters()).dtype
                task_embedding_raw = task_embedding_raw.to(dtype=mol_adapter_dtype)
                task_embedding = self.model.mol_adapter(task_embedding_raw)  # [LLM_hidden_size]
            else:
                # 如果没有 mol_adapter，使用原始 embedding
                task_embedding = task_embedding_raw
            
            # 计算 yield_bin 对应的产率范围
            yield_bin_percent_min = yield_bin * 10
            yield_bin_percent_max = (yield_bin + 1) * 10
            yield_bin_range = f"{yield_bin_percent_min}%-{yield_bin_percent_max}%"
            yield_reg_percent = yield_reg * 100
            
            logger.info(f"✅ Layer2 预测完成: yield_bin={yield_bin} (区间: {yield_bin_range}), yield_reg={yield_reg:.3f} ({yield_reg_percent:.1f}%)")
            
        except Exception as e:
            logger.error(f"❌ Layer2 预测失败: {e}，回退到标准生成")
            import traceback
            traceback.print_exc()
            return first_response
        
        # ===== 第三阶段：构建增强 prompt，第二轮生成 =====
        logger.info("📝 第三阶段：构建增强 prompt，第二轮生成")
        
        # 构建增强 prompt，包含：
        # 1. 原始输入（prompt）
        # 2. Layer2 预测信息（yield_bin, yield_reg，如果任务需要）
        # 注意：embedding 将通过特殊标记追加到序列末尾（像 GVP 一样），而不是文本化
        # 注意：第一轮的 JSON 输出只是中间结果，不需要在 prompt 中展示
        
        # 根据任务类型决定是否包含 yield 信息
        include_yield_info = False
        if task_type:
            task_type_lower = task_type.lower()
            if task_type_lower in ["yield_prediction", "product_yield_prediction"]:
                include_yield_info = True
        
        if include_yield_info:
            # 包含 yield 信息的 prompt
            enhanced_prompt = f"""{prompt}

[Layer2 Prediction]
- Yield Bin: {yield_bin}/9 (predicted yield range: {yield_bin*10}%-{(yield_bin+1)*10}%)
- Yield Value: {yield_reg:.1%}

Based on the original input and Layer2 prediction, please provide your final answer:"""
        else:
            # 不包含 yield 信息，但 Layer2 embedding 会通过 extra_embeddings 传递
            enhanced_prompt = f"""{prompt}

[Layer2 Information]
Layer2 has analyzed the reaction and provided additional context (embedded in the input sequence).

Based on the original input and Layer2 analysis, please provide your final answer:"""
        
        # 第二轮生成
        # 将 Layer2 的 embedding 追加到序列末尾（像 GVP 一样）
        final_response = self.generate(
            enhanced_prompt,
            add_dialog_wrapper=add_dialog_wrapper,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            task_type=task_type,
            realtime_mol=realtime_mol,  # 使用传入的参数
            extra_embeddings=task_embedding,  # 追加 Layer2 embedding 到序列末尾
        )
        
        if isinstance(final_response, list):
            final_response = final_response[0]
        
        logger.info("✅ Layer2 Pipeline 完成")
        
        # 返回结果
        if return_intermediate:
            return {
                "first_response": first_response,
                "layer2_info": {
                    "yield_bin": yield_bin,
                    "yield_reg": yield_reg,
                    "embedding": task_embedding,  # 已经是 LLM 维度
                },
                "final_response": final_response,
            }
        return final_response
    
    def _build_layer2_extraction_prompt(self, original_prompt: str) -> str:
        """
        构建第一轮生成的 prompt，要求模型输出 JSON 格式的反应物/产物信息。
        
        Returns:
            用于第一轮生成的 prompt
        """
        # 检查 prompt 中是否已经包含反应 SMILES
        has_rxn = "Reaction SMILES:" in original_prompt or "Reactants SMILES:" in original_prompt or "Target product SMILES:" in original_prompt
        
        if has_rxn:
            # 如果 prompt 中已经包含反应信息，直接要求提取
            return f"""{original_prompt}

Extract the molecule information from the above question in JSON format. Output only a valid JSON object with the following structure:
{{
    "molecules": [
        {{
            "smiles": "SMILES string",
            "role": "REACTANT" | "PRODUCT" | "REAGENT" | "SOLVENT" | "CATALYST" | "BYPRODUCT" | "OTHER",
            "amount_info": {{
                "moles": float (optional),
                "mass": float (optional),
                "volume": float (optional)
            }} (optional)
        }},
        ...
    ]
}}

Important: Extract ALL molecules mentioned in the question, including reactants, products, reagents, and solvents.
Output only the JSON, no additional text:"""
        else:
            # 如果 prompt 中没有明确的反应信息，要求更详细的提取
            return f"""{original_prompt}

Extract the molecule information from the question in JSON format. Output only a valid JSON object with the following structure:
{{
    "molecules": [
        {{
            "smiles": "SMILES string",
            "role": "REACTANT" | "PRODUCT" | "REAGENT" | "SOLVENT" | "CATALYST" | "BYPRODUCT" | "OTHER",
            "amount_info": {{
                "moles": float (optional),
                "mass": float (optional),
                "volume": float (optional)
            }} (optional)
        }},
        ...
    ]
}}

Example:
{{
    "molecules": [
        {{
            "smiles": "CCO",
            "role": "REACTANT",
            "amount_info": {{
                "moles": 1.0,
                "mass": 46.07
            }}
        }},
        {{
            "smiles": "CC(=O)O",
            "role": "REACTANT"
        }},
        {{
            "smiles": "CC(=O)OCC",
            "role": "PRODUCT"
        }}
    ]
}}

Important: Extract ALL molecules mentioned in the question. If the question contains a reaction SMILES (with >> or >), parse it to extract all reactants and products.
Output only the JSON, no additional text:"""
    
    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """
        从文本中解析 JSON 格式的分子信息（包含角色）。
        使用 json_repair 来修复可能的 JSON 格式错误。
        
        Returns:
            解析后的字典，包含 molecules 列表，每个元素包含 smiles, role, amount_info
            如果解析失败返回 None
        """
        import json
        import re
        
        # 尝试导入 json_repair（使用类级别的缓存避免重复导入检查）
        if not hasattr(MolAwareGenerator2, '_json_repair_available'):
            try:
                import json_repair
                MolAwareGenerator2._json_repair_available = True
                MolAwareGenerator2._json_repair_module = json_repair
            except ImportError:
                MolAwareGenerator2._json_repair_available = False
                MolAwareGenerator2._json_repair_module = None
                # 只在第一次警告
                if not hasattr(MolAwareGenerator2, '_json_repair_warned'):
                    logger.warning("⚠️  json_repair 未安装，将尝试直接解析 JSON。建议安装: pip install json-repair")
                    MolAwareGenerator2._json_repair_warned = True
        
        HAS_JSON_REPAIR = MolAwareGenerator2._json_repair_available
        json_repair = MolAwareGenerator2._json_repair_module if HAS_JSON_REPAIR else None
        
        # 尝试提取 JSON 部分（可能在代码块中）
        # 方法0: 如果包含 _type 和 _content，提取 _content
        if "_type" in text and "_content" in text:
            try:
                # 先解析外层 JSON，提取 _content
                outer_parsed = json.loads(text)
                if isinstance(outer_parsed, dict) and "_content" in outer_parsed:
                    content = outer_parsed["_content"]
                    # _content 可能是字符串（需要再次解析）或已经是字典
                    if isinstance(content, str):
                        text = content  # 继续处理
                        logger.info("📝 从 _content 字段提取 JSON 字符串")
                    elif isinstance(content, dict):
                        # 如果已经是字典，直接返回
                        parsed = content
                        if "molecules" in parsed:
                            return parsed
                        # 否则继续处理
                        text = json.dumps(content)
                        logger.info("📝 从 _content 字段提取 JSON 对象")
                else:
                    # 如果解析失败，尝试正则提取
                    content_match = re.search(r'"_content"\s*:\s*"([^"]+)"', text, re.DOTALL)
                    if content_match:
                        # 处理转义字符
                        import codecs
                        text = codecs.decode(content_match.group(1), 'unicode_escape')
                        logger.info("📝 从 _content 字段（转义）提取 JSON 内容")
            except Exception as e:
                logger.debug(f"解析 _content 失败: {e}，继续使用原始文本")
        
        # 方法1: 查找 ```json ... ``` 代码块
        json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_block_pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # 方法2: 查找第一个 { ... } 块
            brace_match = re.search(r'\{.*\}', text, re.DOTALL)
            if brace_match:
                json_str = brace_match.group(0)
            else:
                # 方法3: 直接使用整个文本
                json_str = text.strip()
        
        # 清理可能的尾随逗号等
        json_str = json_str.strip()
        if json_str.endswith(','):
            json_str = json_str[:-1]
        
        # 尝试直接解析
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            # 如果直接解析失败，尝试使用 json_repair
            if HAS_JSON_REPAIR:
                try:
                    logger.info("🔧 使用 json_repair 修复 JSON...")
                    repaired_json_str = json_repair.repair_json(json_str)
                    parsed = json.loads(repaired_json_str)
                    logger.info("✅ JSON 修复成功")
                except Exception as repair_e:
                    logger.warning(f"⚠️  json_repair 修复失败: {repair_e}")
                    logger.debug(f"尝试解析的文本: {json_str[:200]}...")
                    return None
            else:
                logger.warning(f"⚠️  JSON 解析失败: {e}")
                logger.debug(f"尝试解析的文本: {json_str[:200]}...")
                return None
        
        # 验证必需字段
        if "molecules" not in parsed:
            # 兼容旧格式：如果只有 reactant_smiles，转换为新格式
            if "reactant_smiles" in parsed:
                logger.info("🔄 检测到旧格式 JSON，转换为新格式...")
                reactant_smiles = parsed["reactant_smiles"]
                if isinstance(reactant_smiles, str):
                    reactant_smiles = [reactant_smiles]
                
                molecules = []
                for smi in reactant_smiles:
                    molecules.append({
                        "smiles": smi,
                        "role": "REACTANT",
                        "amount_info": parsed.get("amount_info")
                    })
                parsed = {"molecules": molecules}
            else:
                logger.warning("⚠️  JSON 中缺少 molecules 或 reactant_smiles 字段")
                return None
        
        # 验证 molecules 格式
        if not isinstance(parsed["molecules"], list):
            logger.warning("⚠️  molecules 字段必须是列表")
            return None
        
        # 验证每个分子对象
        for i, mol in enumerate(parsed["molecules"]):
            if not isinstance(mol, dict):
                logger.warning(f"⚠️  molecules[{i}] 必须是字典")
                return None
            if "smiles" not in mol:
                logger.warning(f"⚠️  molecules[{i}] 缺少 smiles 字段")
                return None
            if "role" not in mol:
                # 默认角色为 REACTANT
                mol["role"] = "REACTANT"
                logger.info(f"ℹ️  molecules[{i}] 缺少 role 字段，默认为 REACTANT")
        
        return parsed
    
    def _extract_reactant_smiles(self, text: str) -> Optional[str | List[str]]:
        """
        从文本中提取反应物 SMILES。
        优先从 <mol>...</mol> 标签中提取，否则使用正则表达式。
        支持提取多个反应物。
        """
        import re
        
        # 方法1: 从 <mol>...</mol> 标签中提取（可能多个）
        mol_pattern = r'<mol>(.*?)</mol>'
        matches = re.findall(mol_pattern, text, re.DOTALL)
        if matches:
            candidates = []
            for match in matches:
                candidate = match.strip()
                # 简单验证：如果看起来像 SMILES（包含常见原子符号）
                if any(c in candidate for c in ['C', 'N', 'O', 'S', '(', ')', '=', '[', ']']):
                    candidates.append(candidate)
            if candidates:
                # 如果只有一个，返回字符串；多个则返回列表
                return candidates[0] if len(candidates) == 1 else candidates
        
        # 方法2: 使用正则表达式提取可能的 SMILES
        # SMILES 模式：包含原子符号、括号、数字等
        simple_pattern = r'[CNOScnos()=\[\]#@0-9]+'
        matches = re.findall(simple_pattern, text)
        if matches:
            # 取最长的匹配
            candidate = max(matches, key=len)
            if len(candidate) >= 5:  # 至少5个字符才可能是有效的 SMILES
                return candidate
        
        return None
    
    def _extract_amount_info(self, text: str) -> Optional[Dict[str, float] | List[Dict[str, float]]]:
        """
        从文本中提取 amount_info（量信息）。
        尝试从文本中解析 moles、mass、volume 等信息。
        
        返回格式:
        - 单个: {"moles": float, "mass": float, "volume": float}
        - 多个: [{"moles": float, ...}, ...]
        """
        import re
        
        # 简单的提取逻辑：查找数字和单位
        # 例如: "1.0 mol", "2.5 g", "100 mL"
        amount_info = {}
        
        # 提取 moles
        moles_pattern = r'(\d+\.?\d*)\s*(?:mol|mole|moles)'
        moles_match = re.search(moles_pattern, text, re.IGNORECASE)
        if moles_match:
            amount_info["moles"] = float(moles_match.group(1))
        
        # 提取 mass
        mass_pattern = r'(\d+\.?\d*)\s*(?:g|gram|grams|mg|milligram)'
        mass_match = re.search(mass_pattern, text, re.IGNORECASE)
        if mass_match:
            amount_info["mass"] = float(mass_match.group(1))
        
        # 提取 volume
        volume_pattern = r'(\d+\.?\d*)\s*(?:ml|mL|milliliter|liters?|L)'
        volume_match = re.search(volume_pattern, text, re.IGNORECASE)
        if volume_match:
            amount_info["volume"] = float(volume_match.group(1))
        
        return amount_info if amount_info else None

   
if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--ckpt_dir", default=os.environ.get("CHECKPOINT_DIR", ""), help="Path to model checkpoint")
    _p.add_argument("--token_classifier_path", default=os.environ.get("TOKEN_CLASSIFIER_PATH", ""), help="Path to token classifier .pt file")
    _p.add_argument("--device", default="cuda:0")
    _p.add_argument("--multi_gpu", action="store_true", help="Use device_map=auto for multi-GPU")
    _p.add_argument("--dtype", default="bf16")
    _args = _p.parse_args()

    CONFIG = {
        "ckpt_dir": _args.ckpt_dir,
        "device": None if _args.multi_gpu else _args.device,
        "device_map": "auto" if _args.multi_gpu else None,
        "dtype": _args.dtype,
        "debug": True,
        "token_classifier_path": _args.token_classifier_path,
    }

    gen = MolAwareGenerator2()
    gen.load(CONFIG)

    prompt = "Describe this molecule: CCCCCCC(O)C/C=C\\CCCCCCCC(=O)[O-]\nPlease only output the answer."
    # prompt = "How's the weather in Beijing?"
    text = gen.generate(
        prompt,
        add_dialog_wrapper=True,
        realtime_mol=False,
        max_new_tokens=1024,
        max_tokens=16384,
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.05,
        skip_special_tokens=True,
    )
    print("\n=== Generated Text ===")
    print(text)
    print("\n=== Raw Output (first 500 chars) ===")
    if hasattr(gen, '_last_raw_outputs') and gen._last_raw_outputs:
        print(gen._last_raw_outputs[0][:500] if gen._last_raw_outputs[0] else "N/A")
    else:
        print("N/A")

