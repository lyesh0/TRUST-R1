# Search-R1 项目复现完整实验手册

本手册旨在帮助你在 AutoDL 云服务器上，从零开始完整复现 Search-R1 项目的 3B 模型基线实验。

---

## 目录

1. [实验概述](#一实验概述)
2. [前期准备](#二前期准备)
3. [AutoDL 实例配置](#三autodl-实例配置)
4. [环境搭建](#四环境搭建)
5. [数据准备](#五数据准备)
6. [模型下载](#六模型下载)
7. [检索服务部署](#七检索服务部署)
8. [配置训练脚本](#八配置训练脚本)
9. [运行训练](#九运行训练)
10. [监控与调优](#十监控与调优)
11. [常见问题](#十一常见问题)
12. [预期结果](#十二预期结果)

---

## 一、实验概述

### 1.1 实验目标

| 项目 | 说明 |
|------|------|
| **目标模型** | Qwen2.5-3B-Instruct 或 Llama-3.2-3B-Instruct |
| **训练算法** | GRPO（组相对策略优化） |
| **任务类型** | NQ Search（自然问题搜索问答） |
| **数据集规模** | 10,000 条训练数据 |
| **训练步数** | 500-1000 步 |
| **预期 EM 提升** | 从基线 ~30% 提升到 ~50%+ |

### 1.2 硬件需求

| 配置项 | 最低要求 | 推荐配置 |
|--------|---------|---------|
| **GPU** | A100 40GB | A100 80GB |
| **GPU 数量** | 1 张 | 1 张 |
| **CPU** | 8 核 | 16 核 |
| **内存** | 64GB | 128GB |
| **系统盘** | 100GB | 100GB |
| **数据盘** | 200GB | 500GB |

### 1.3 时间与成本估算

| 阶段 | 时间 | 费用（按 A100 40GB ¥6/时） |
|------|------|---------------------------|
| 环境配置与调试 | 1-2 小时 | ¥6-12 |
| 模型下载 | 0.5-1 小时 | ¥3-6 |
| 数据下载处理 | 0.5 小时 | ¥3 |
| 训练（500步） | 8-12 小时 | ¥48-72 |
| **总计** | **10-15 小时** | **¥60-93** |

### 1.4 实验流程图

```
┌────────────────────────────────────────────────────────────────────────┐
│                           实验流程总览                                  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐           │
│  │ 1. 注册 AutoDL │ ──► │ 2. 租用实例  │ ──► │ 3. 配置环境  │           │
│  └──────────────┘     └──────────────┘     └──────────────┘           │
│         │                                         │                    │
│         │          ┌──────────────────────────────┘                    │
│         │          ▼                                                    │
│         │     ┌──────────────┐     ┌──────────────┐     ┌───────────┐ │
│         │     │ 4. 下载模型  │ ──► │ 5. 准备数据   │ ──► │ 6. 构建索引│ │
│         │     └──────────────┘     └──────────────┘     └───────────┘ │
│         │                                                        │      │
│         │          ┌──────────────────────────────────────────────┘    │
│         │          ▼                                                    │
│         │     ┌──────────────┐     ┌──────────────┐     ┌───────────┐ │
│         │     │ 7. 启动检索  │ ──► │ 8. 运行训练   │ ──► │ 9. 监控   │ │
│         │     └──────────────┘     └──────────────┘     └───────────┘ │
│         │                                                        │      │
│         │          ┌──────────────────────────────────────────────┘    │
│         │          ▼                                                    │
│         │     ┌──────────────┐     ┌──────────────┐                   │
│         └────►│ 10. 结果分析 │ ──► │ 11. 总结报告 │                   │
│               └──────────────┘     └──────────────┘                   │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 二、前期准备

### 2.1 必要账号注册

#### 2.1.1 AutoDL 账号

1. 访问 [autodl.com](https://www.autodl.com)
2. 点击「注册」按钮
3. 使用手机号或邮箱注册
4. 完成实名认证（云服务器必需）

#### 2.1.2 HuggingFace 账号（获取模型权限）

1. 访问 [huggingface.co](https://huggingface.co)
2. 注册账号
3. 申请 Llama 模型权限：
   - 访问 [Llama 3.2 3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
   - 点击「Request access」申请权限
4. 生成 Access Token：
   - 进入 Settings → Access Tokens
   - 创建新 Token，权限选择「Read」
   - 复制 Token 备用

#### 2.1.3 WandB 账号（可选，用于监控）

1. 访问 [wandb.ai](https://wandb.ai)
2. 注册账号
3. 获取 API Key：
   - 进入 Settings → API Keys
   - 复制 API Key 备用

### 2.2 本地准备工作

```bash
# 1. 确保本地已安装 SSH 客户端
# macOS/Linux 自带，Windows 可使用 PowerShell 或安装 Git Bash

# 2. 测试 SSH 连接
ssh -V

# 3. 配置 SSH 密钥（可选，但更方便）
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub  # 复制公钥到 AutoDL
```

### 2.3 准备本地文件传输工具

```bash
# macOS 安装 scp（自带）
# 或使用 rsync（更高效）
brew install rsync  # macOS

# 示例：
rsync -avz -e ssh local_folder/ user@server:/remote/path/
```

---

## 三、AutoDL 实例配置

### 3.1 登录 AutoDL

1. 访问 [autodl.com](https://www.autodl.com)
2. 登录账号
3. 进入控制台

### 3.2 选择资源配置

#### 3.2.1 基础配置

| 配置项 | 选择 |
|--------|------|
| **区域** | 西部（乌兰察布/保定）- 价格较低 |
| **计费方式** | 按量付费（先用按量，确认OK后可选包月） |
| **镜像** | PyTorch 2.1.0 + CUDA 11.8 |

#### 3.2.2 GPU 选择

| GPU 型号 | 显存 | 价格参考 | 推荐度 |
|---------|------|---------|--------|
| **RTX 4090** | 24GB | ¥1.5-2/时 | ⭐⭐（便宜但可能不够） |
| **A100-PCIE-40GB** | 40GB | ¥5-7/时 | ⭐⭐⭐⭐（性价比高） |
| **A100-SXM-80GB** | 80GB | ¥8-12/时 | ⭐⭐⭐⭐⭐（推荐，稳定） |
| **H100** | 80GB | ¥15-20/时 | ⭐⭐（太贵，不推荐） |

**推荐选择**：1x A100-PCIE-40GB 或 1x A100-SXM-80GB

#### 3.2.3 资源配置详情

```
实例规格配置：
├── GPU: A100-PCIE-40GB × 1
├── CPU: 14 核（Intel Xeon）
├── 内存: 110 GB
├── 系统盘: 50 GB
├── 数据盘: 100 GB（推荐挂载）
└── 镜像: PyTorch 2.1.0 + CUDA 11.8 + Python 3.10
```

### 3.3 创建实例

1. 点击「租用新实例」
2. 按上述配置选择
3. 点击「立即租用」
4. 等待实例创建完成（通常 1-3 分钟）

### 3.4 连接实例

```bash
# 方法1：使用 AutoDL 网页终端
# 在控制台点击「Terminal」直接使用

# 方法2：使用本地 SSH 连接
# 在 AutoDL 控制台查看 SSH 命令，通常格式为：
ssh -p 端口号 root@机器IP

# 示例：
ssh -p 12345 root@123.456.78.90

# 首次连接需要设置密码
```

### 3.5 初始化数据盘（如果挂载了数据盘）

```bash
# 查看磁盘情况
df -h

# 格式化并挂载数据盘（如果未挂载）
# 假设数据盘是 /dev/vdb
sudo mkfs.ext4 /dev/vdb
sudo mkdir /mnt/data
sudo mount /dev/vdb /mnt/data

# 设置自动挂载（可选）
echo '/dev/vdb /mnt/data ext4 defaults 0 0' | sudo tee -a /etc/fstab
```

---

## 四、环境搭建

### 4.1 SSH 连接后基础配置

```bash
# 1. 更新系统包
apt-get update && apt-get upgrade -y

# 2. 安装基础工具
apt-get install -y git wget curl vim htop tmux screen

# 3. 检查 Python 版本
python3 --version  # 应该是 3.10.x
```

### 4.2 创建项目目录

```bash
# 在数据盘创建项目目录
mkdir -p /mnt/data/Search-R1
cd /mnt/data/Search-R1

# 创建数据目录
mkdir -p data/models
mkdir -p data/corpus
mkdir -p data/index
mkdir -p checkpoints
mkdir -p logs
```

### 4.3 配置 conda 环境

```bash
# 1. 如果没有 conda，先安装
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/.bashrc

# 2. 创建虚拟环境
conda create -n searchr1 python=3.10 -y
conda activate searchr1

# 3. 验证
python --version  # 应该显示 Python 3.10.x
```

### 4.4 安装 PyTorch

```bash
# 激活环境
conda activate searchr1

# 安装 PyTorch 2.1.0 (CUDA 11.8)
pip install torch==2.1.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 4.5 安装核心依赖

```bash
# 1. Flash Attention（加速 LLM 推理）
pip install flash-attn --no-build-isolation

# 2. HuggingFace 生态
pip install transformers accelerate bitsandbytes sentencepiece

# 3. vLLM（高效推理引擎）
pip install vllm==0.4.0.post1

# 4. 分布式训练框架
pip install "ray[default]" omegaconf

# 5. 数据处理
pip install pandas pyarrow datasets tqdm

# 6. 检索相关
pip install faiss-cpu pyserini rank-bm25

# 7. 其他工具
pip install wandb rich
```

### 4.6 克隆项目代码

```bash
cd /mnt/data/Search-R1

# 克隆 Search-R1
git clone https://github.com/你的用户名/Search-R1.git .
# 或直接克隆原仓库（如果你的代码已同步）
git clone https://github.com/AI-MO/Search-R1.git .

# 安装项目本身
pip install -e .
```

### 4.7 安装 veRL

```bash
# veRL 是底层训练框架
pip install verl

# 或从源码安装（如果需要最新版本）
cd /mnt/data
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
```

### 4.8 验证环境

```bash
# 逐个验证各组件
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')"
python -c "import verl; print(f'✓ veRL installed')"
python -c "import vllm; print(f'✓ vLLM: {vllm.__version__}')"
python -c "import ray; print(f'✓ Ray: {ray.__version__}')"

# 检查 GPU
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

---

## 五、数据准备

### 5.1 数据集说明

Search-R1 使用 NQ（Natural Questions）数据集，包含：
- 真实用户问题
- Wikipedia 作为知识来源
- 标准答案用于 EM（精确匹配）评估

### 5.2 下载训练数据

```bash
cd /mnt/data/Search-R1

# 注意：scripts/download.py 只负责准备 wiki-18 检索语料和 e5 index，
# 不负责下载 NQ 训练数据。NQ 训练数据可直接从 HuggingFace 下载。

# 直接从 HuggingFace 下载
python -c "
from datasets import load_dataset

# 加载 NQ 数据集
train_ds = load_dataset('natural_questions', split='train[:10000]')  # 先下载10000条
test_ds = load_dataset('natural_questions', split='validation[:1000]')

print(f'Train samples: {len(train_ds)}')
print(f'Test samples: {len(test_ds)}')

# 保存为 parquet 格式
train_ds.to_parquet('./data/nq_train.parquet')
test_ds.to_parquet('./data/nq_test.parquet')

print('Data saved!')
"
```

### 5.3 处理数据格式

Search-R1 需要特定格式的数据。创建处理脚本：

```python
# scripts/process_nq_data.py
import json
import pandas as pd
from datasets import load_dataset

def process_nq_data():
    """处理 NQ 数据集为 Search-R1 格式"""
    
    # 加载数据
    train_ds = load_dataset('natural_questions', split='train[:10000]')
    test_ds = load_dataset('natural_questions', split='validation[:1000]')
    
    def format_sample(example):
        """转换为 Search-R1 格式"""
        question = example['question']['text']
        # 提取短答案（如果有）
        short_answers = example.get('annotations', [{}])[0].get('short_answers', [])
        if short_answers:
            answer = short_answers[0]['text']
        else:
            # 使用长答案作为备选
            long_answers = example.get('annotations', [{}])[0].get('long_answers', [])
            if long_answers:
                answer = long_answers[0]['text']
            else:
                answer = ""
        
        return {
            'data_source': 'nq',
            'prompt': [
                {
                    'role': 'user',
                    'content': f'Question: {question}\nPlease search for the answer and provide your final answer in <answer> tags.\nAnswer: '
                }
            ],
            'ability': 'fact-reasoning',
            'reward_model': {
                'style': 'rule',
                'ground_truth': {
                    'target': [answer] if answer else []
                }
            },
            'extra_info': {
                'split': example.get('_split', 'train'),
                'index': example.get('id', 0)
            }
        }
    
    # 处理训练集
    train_data = [format_sample(ex) for ex in train_ds]
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet('./data/nq_search/train.parquet')
    
    # 处理测试集
    test_data = [format_sample(ex) for ex in test_ds]
    test_df = pd.DataFrame(test_data)
    test_df.to_parquet('./data/nq_search/test.parquet')
    
    print(f"Processed {len(train_data)} train samples")
    print(f"Processed {len(test_data)} test samples")

if __name__ == '__main__':
    process_nq_data()
```

```bash
# 运行处理脚本
mkdir -p data/nq_search
python scripts/process_nq_data.py
```

### 5.4 下载 wiki-18 检索语料和 e5 index（用于检索）

```bash
cd /root/autodl-tmp/TRUST-R1

# wiki-18 检索语料和 index 较大，应在 AutoDL 上执行，不要在本地 Mac 下载。
SEARCH_DATA_ROOT=/root/autodl-fs
python scripts/download.py --data-root "$SEARCH_DATA_ROOT"

# 查看检索数据
ls -lh /root/autodl-fs/data/wiki-18.jsonl
ls -lh /root/autodl-fs/indexes/wiki-18/e5_Flat.index
```

---

## 六、模型下载

### 6.1 配置 HuggingFace 访问

```bash
# 1. 登录 HuggingFace
huggingface-cli login

# 2. 输入你的 Access Token
# Token: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 3. 验证登录
huggingface-cli whoami
```

### 6.2 下载模型

```bash
cd /mnt/data/Search-R1/data/models

# 推荐模型1：Qwen2.5-3B-Instruct（推荐，较小且效果好）
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir Qwen2.5-3B-Instruct

# 推荐模型2：Llama-3.2-3B-Instruct（需要申请权限）
# huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir Llama-3.2-3B-Instruct

# 备用模型（更小，适合测试）
# huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir Qwen2.5-1.5B-Instruct

# 查看已下载的模型
ls -la /mnt/data/Search-R1/data/models/
```

### 6.3 验证模型

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = '/mnt/data/Search-R1/data/models/Qwen2.5-3B-Instruct'

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f'Vocab size: {len(tokenizer)}')

print('Loading model (CPU for test)...')
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
print(f'Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B')
print('Model loaded successfully!')
"
```

---

## 七、检索服务部署

### 7.1 构建检索索引

Search-R1 支持多种检索方式，推荐使用 BM25（简单快速）：

```bash
cd /mnt/data/Search-R1

# 创建索引保存目录
mkdir -p data/index/bm25

# 如果有 Wikipedia 语料，运行索引构建
# python -m search_r1.search.index_builder \
#     --corpus_path data/wiki18.jsonl \
#     --index_path data/index/bm25 \
#     --retriever_type bm25
```

### 7.2 启动检索服务

创建一个启动脚本 `start_retriever.sh`：

```bash
#!/bin/bash
# start_retriever.sh

cd /mnt/data/Search-R1

# 设置参数
CORPUS_PATH="/mnt/data/Search-R1/data/wiki18.jsonl"
INDEX_PATH="/mnt/data/Search-R1/data/index/bm25"
PORT=8000

# 启动 BM25 检索服务
python search_r1/search/retrieval_server.py \
    --corpus_path $CORPUS_PATH \
    --index_path $INDEX_PATH \
    --topk 3 \
    --retriever_name bm25 \
    --port $PORT
```

```bash
# 赋予执行权限
chmod +x start_retriever.sh

# 后台启动检索服务
nohup ./start_retriever.sh > logs/retriever.log 2>&1 &

# 等待几秒让服务启动
sleep 5

# 检查服务是否启动
ps aux | grep retrieval_server
curl -s http://127.0.0.1:8000/docs || echo "服务可能需要更多时间启动"
```

### 7.3 测试检索服务

```bash
# 测试检索功能
curl -X POST http://127.0.0.1:8000/retrieve \
    -H "Content-Type: application/json" \
    -d '{"queries": ["Who is the president of the United States?"], "topk": 3}'

# 预期输出格式：
# {"result": [[{"document": {...}, "score": ...}, ...]]}
```

---

## 八、配置训练脚本

### 8.1 创建训练配置

创建专门用于 AutoDL 的训练脚本 `train_grpo_autodl.sh`：

```bash
#!/bin/bash
# train_grpo_autodl.sh - AutoDL 专用训练脚本

# ==================== 基本配置 ====================
# GPU 配置
export CUDA_VISIBLE_DEVICES=0

# 模型路径
export BASE_MODEL='/mnt/data/Search-R1/data/models/Qwen2.5-3B-Instruct'
# 如果使用 Llama：
# export BASE_MODEL='/mnt/data/Search-R1/data/models/Llama-3.2-3B-Instruct'

# 数据路径
export DATA_DIR='/mnt/data/Search-R1/data/nq_search'
export CORPUS_DIR='/mnt/data/Search-R1/data/wiki18.jsonl'
export INDEX_DIR='/mnt/data/Search-R1/data/index/bm25'

# 实验名称
export EXPERIMENT_NAME="search-r1-grpo-qwen2.5-3b-nq"
export WAND_PROJECT="Search-R1-AutoDL"

# ==================== 训练参数 ====================
# 批处理大小（根据 GPU 显存调整）
# A100 40GB: 256
# A100 80GB: 512
export TRAIN_BATCH_SIZE=256
export VAL_BATCH_SIZE=128

# 训练步数
# 验证概念: 100步
# 小规模: 500步
# 完整训练: 1000步
export TRAIN_STEPS=500

# 模型配置
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=500
export MAX_START_LENGTH=2048
export MAX_OBS_LENGTH=500
export MAX_TURNS=2

# ==================== vLLM 配置 ====================
# GPU 利用率（留一些给其他组件）
# A100 40GB: 0.65
# A100 80GB: 0.85
export GPU_MEMORY_UTILIZATION=0.65

# ==================== 开始训练 ====================
cd /mnt/data/Search-R1

python -m verl.trainer.main_ppo \
    # 数据配置
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=10000 \
    data.val_data_num=1000 \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.max_start_length=$MAX_START_LENGTH \
    data.max_obs_length=$MAX_OBS_LENGTH \
    data.shuffle_train_dataloader=True \
    
    # 算法配置（使用 GRPO）
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    
    # Actor/Rollout/Ref 配置
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    
    # 训练器配置
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=$TRAIN_STEPS \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=/mnt/data/Search-R1/checkpoints/$EXPERIMENT_NAME \
    
    # 检索配置
    max_turns=$MAX_TURNS \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    
    2>&1 | tee logs/$EXPERIMENT_NAME.log
```

### 8.2 配置 WandB（可选）

```bash
# 在 AutoDL 实例上登录 WandB
wandb login
# 输入你的 WandB API Key

# 或者使用离线模式（不记录到 WandB）
# 修改脚本：trainer.logger=[] \
```

### 8.3 创建监控脚本

创建 `monitor.sh` 用于监控训练进度：

```bash
#!/bin/bash
# monitor.sh - 监控训练进程

echo "========== Search-R1 训练监控 =========="
echo "时间: $(date)"
echo ""

# 检查进程
echo "训练进程:"
ps aux | grep "verl.trainer.main_ppo" | grep -v grep || echo "  未发现训练进程"
echo ""

# GPU 使用情况
echo "GPU 状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo ""

# 磁盘使用
echo "磁盘使用:"
df -h /mnt/data
echo ""

# 查看最新日志
echo "最近日志 (最后 20 行):"
if [ -f "logs/current.log" ]; then
    tail -n 20 logs/current.log
fi
echo ""

# 查看 wandb 链接（如果有）
echo "WandB 链接:"
grep -i "wandb" logs/*.log 2>/dev/null | tail -1 || echo "  无 WandB 链接"
```

```bash
chmod +x monitor.sh
```

---

## 九、运行训练

### 9.1 启动前的最后检查

```bash
cd /mnt/data/Search-R1

# 1. 确认检索服务运行
curl -s http://127.0.0.1:8000/docs | head -1
# 预期：{"openapi":"3.0.2",...

# 2. 确认模型文件存在
ls -la data/models/Qwen2.5-3B-Instruct/ | head -5

# 3. 确认数据文件存在
ls -la data/nq_search/

# 4. 确认 GPU 可用
nvidia-smi
```

### 9.2 使用 screen/tmux 保持会话

```bash
# 创建新的 screen 会话
screen -S searchr1

# 或者使用 tmux
tmux new -s searchr1
```

### 9.3 开始训练

```bash
# 激活 conda 环境
conda activate searchr1

# 给脚本执行权限
chmod +x train_grpo_autodl.sh
chmod +x start_retriever.sh

# 如果检索服务还没启动，先启动它
./start_retriever.sh &
sleep 3

# 开始训练
./train_grpo_autodl.sh
```

### 9.4 训练过程中的常用操作

```bash
# 在另一个终端中监控
watch -n 5 nvidia-smi

# 查看训练日志
tail -f logs/search-r1-grpo-qwen2.5-3b-nq.log

# 查看 GPU 使用率
nvidia-smi -l 1

# 如果需要中断训练
# Ctrl+C 或
pkill -f "verl.trainer.main_ppo"

# 恢复被中断的 screen 会话
screen -r searchr1

# 如果会话 detached
screen -ls
screen -D -R searchr1
```

### 9.5 预期训练输出

训练开始后，你会看到类似以下的输出：

```
[Epoch 1/15] [Step 1/500]
  batch_size: 256
  generating responses...
  
[Epoch 1/15] [Step 10/500]
  rewards/mean: 0.32
  rewards/max: 1.0
  kl_loss: 0.002
  policy_loss: -0.15
  grad_norm: 1.23
  
  [Validation]
  test_score/mean: 0.31
  response_length/mean: 245.3
  timing_s/total: 45.2
  
[Epoch 1/15] [Step 50/500]
  rewards/mean: 0.45
  rewards/max: 1.0
  kl_loss: 0.003
  policy_loss: -0.22
  grad_norm: 1.15
  
  [Validation]
  test_score/mean: 0.48
  response_length/mean: 312.7
```

---

## 十、监控与调优

### 10.1 WandB 监控

如果配置了 WandB，可以：

1. 访问 [wandb.ai](https://wandb.ai)
2. 进入你的项目
3. 查看实时训练曲线

关键监控指标：

| 指标 | 说明 | 期望趋势 |
|------|------|---------|
| `rewards/mean` | 平均奖励 | 逐渐上升 |
| `test_score/mean` | 验证集 EM | 逐渐上升 |
| `kl_loss` | KL 散度损失 | 稳定在 0.01-0.05 |
| `policy_loss` | 策略损失 | 逐渐接近 0 |
| `grad_norm` | 梯度范数 | 稳定在 0.5-2.0 |

### 10.2 本地监控脚本

```bash
# 运行监控脚本
./monitor.sh

# 或持续监控
while true; do
    clear
    ./monitor.sh
    sleep 10
done
```

### 10.3 超参数调优建议

如果训练效果不佳，可以调整以下参数：

| 参数 | 当前值 | 调优建议 |
|------|--------|---------|
| `actor_rollout_ref.actor.optim.lr` | 1e-6 | 如果 loss 不下降，尝试 5e-7 或 2e-6 |
| `algorithm.kl_ctrl.kl_coef` | 0.001 | 如果策略变化太大，增大到 0.01 |
| `actor_rollout_ref.rollout.n_agent` | 5 | 增加样本数可提高稳定性 |
| `actor_rollout_ref.rollout.temperature` | 1 | 如果生成过于重复，降低到 0.7 |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 128 | 如果 OOM，减少到 64 |

---

## 十一、常见问题

### 11.1 启动问题

#### Q1: 检索服务启动失败

```
Error: Connection refused on port 8000
```

**解决方案**：
```bash
# 1. 检查服务是否启动
ps aux | grep retrieval_server

# 2. 查看错误日志
tail -f logs/retriever.log

# 3. 手动启动并查看错误
python search_r1/search/retrieval_server.py --corpus_path /path/to/corpus.jsonl --index_path /path/to/index --topk 3
```

#### Q2: 模型加载失败

```
Error: Could not find the model files
```

**解决方案**：
```bash
# 1. 确认模型路径
ls -la /mnt/data/Search-R1/data/models/

# 2. 重新下载（如果文件损坏）
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir /mnt/data/Search-R1/data/models/Qwen2.5-3B-Instruct --local-dir-use-symlinks=False
```

### 11.2 内存问题

#### Q3: CUDA Out of Memory (OOM)

```
RuntimeError: CUDA out of memory. Tried to allocate...
```

**解决方案**：
```bash
# 1. 减小 batch size
# 修改 train_grpo_autodl.sh：
export TRAIN_BATCH_SIZE=128  # 从 256 减到 128

# 2. 减小 vLLM 显存占用
export GPU_MEMORY_UTILIZATION=0.5  # 从 0.65 减到 0.5

# 3. 启用模型卸载
actor_rollout_ref.actor.fsdp_config.param_offload=true
actor_rollout_ref.actor.fsdp_config.grad_offload=true
actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
```

#### Q4: 系统内存不足

```bash
# 1. 检查内存使用
free -h

# 2. 增加 swap（如果有空间）
sudo fallocate -l 64G /mnt/swap
sudo chmod 600 /mnt/swap
sudo mkswap /mnt/swap
sudo swapon /mnt/swap
```

### 11.3 训练问题

#### Q5: 训练 loss 不下降

**可能原因及解决方案**：

1. **奖励函数问题**：检查 reward_model 配置
```bash
# 查看 reward 是否正常计算
grep -i "reward" logs/*.log
```

2. **学习率问题**：尝试调整
```bash
actor_rollout_ref.actor.optim.lr=5e-7  # 减小
# 或
actor_rollout_ref.actor.optim.lr=2e-6  # 增大
```

3. **数据问题**：检查数据格式
```python
# 验证数据格式
import pandas as pd
df = pd.read_parquet('./data/nq_search/train.parquet')
print(df.head())
print(df.columns)
```

#### Q6: 训练过程中断

**解决方案**：

1. 使用 screen/tmux 保持会话
2. 定期检查 checkpoint
```bash
ls -la checkpoints/
```

3. 修改保存频率
```bash
trainer.save_freq=50  # 每 50 步保存
```

### 11.4 网络问题

#### Q7: HuggingFace 下载慢

```bash
# 使用镜像站
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-3B-Instruct
```

#### Q8: WandB 连接失败

```bash
# 检查网络
curl https://wandb.ai

# 或使用离线模式（不推荐，无法监控）
# 修改脚本：
trainer.logger=[]  # 移除 wandb
```

---

## 十二、预期结果

### 12.1 训练结果预期

| 训练阶段 | 步数 | 预期 EM 分数 |
|---------|------|-------------|
| 初始 | 0 | ~0.30-0.35 |
| 早期 | 50 | ~0.35-0.40 |
| 中期 | 200 | ~0.42-0.48 |
| 后期 | 500 | ~0.48-0.55 |
| 充分训练 | 1000 | ~0.50-0.60 |

### 12.2 训练时间参考

| 配置 | 模型 | 步数 | 实际时间 |
|------|------|------|---------|
| A100 40GB | Qwen2.5-3B | 100步 | ~1.5小时 |
| A100 40GB | Qwen2.5-3B | 500步 | ~8-10小时 |
| A100 40GB | Qwen2.5-3B | 1000步 | ~16-20小时 |
| A100 80GB | Qwen2.5-3B | 1000步 | ~12-14小时 |

### 12.3 成功标志

训练成功完成的标志：

1. ✅ WandB 曲线显示 rewards/mean 持续上升
2. ✅ test_score/mean 从 ~0.30 提升到 ~0.50+
3. ✅ kl_loss 稳定在合理范围（0.01-0.05）
4. ✅ grad_norm 稳定在 0.5-2.0
5. ✅ 最终 checkpoint 保存成功

### 12.4 结果分析

训练完成后，分析最佳 checkpoint：

```python
# 查看 wandb 记录确定最佳步数
# 在 wandb.ai 项目页面查看 val/test_score 曲线

# 测试最佳 checkpoint
python evaluate.py \
    --model_path checkpoints/best_model \
    --test_data data/nq_search/test.parquet \
    --retriever_url http://127.0.0.1:8000/retrieve
```

---

## 附录 A：完整命令清单

```bash
# ===== 第一步：连接服务器 =====
ssh -p 端口号 root@机器IP
cd /mnt/data/Search-R1

# ===== 第二步：启动环境 =====
conda activate searchr1

# ===== 第三步：启动检索服务 =====
nohup python search_r1/search/retrieval_server.py \
    --corpus_path /mnt/data/Search-R1/data/wiki18.jsonl \
    --index_path /mnt/data/Search-R1/data/index/bm25 \
    --topk 3 \
    --retriever_name bm25 > logs/retriever.log 2>&1 &

# 等待服务启动
sleep 5

# 验证服务
curl -X POST http://127.0.0.1:8000/retrieve \
    -H "Content-Type: application/json" \
    -d '{"queries": ["test"], "topk": 1}'

# ===== 第四步：开始训练 =====
screen -S searchr1
chmod +x train_grpo_autodl.sh
./train_grpo_autodl.sh

# ===== 第五步：监控（在新终端） =====
watch -n 10 nvidia-smi
tail -f logs/search-r1-grpo-qwen2.5-3b-nq.log
```

---

## 附录 B：费用优化建议

### 节省策略

| 策略 | 节省比例 | 说明 |
|------|---------|------|
| 先用按量付费测试 | - | 确认 OK 后可选包月 |
| 选择西部区域 | 10-20% | 乌兰察布/保定价格较低 |
| 使用抢占式实例 | 30-50% | 价格低但可能被中断 |
| 减少训练步数 | 线性节省 | 1000步 vs 500步 |
| 早点保存退出 | 随时 | 观察结果决定是否继续 |

### 推荐的省钱组合

```
低预算方案（¥50-80）：
├── GPU: RTX 4090 24GB
├── 数据量: 5000条
├── 训练步数: 300步
└── 预期效果: 验证概念

中等预算方案（¥100-150）：
├── GPU: A100 40GB
├── 数据量: 10000条
├── 训练步数: 500步
└── 预期效果: 初步基线

高预算方案（¥200-300）：
├── GPU: A100 80GB
├── 数据量: 20000条
├── 训练步数: 1000步
└── 预期效果: 完整基线
```

---

## 附录 C：检查清单

### 启动前检查清单

- [ ] AutoDL 实例已创建并连接
- [ ] conda 环境已激活
- [ ] 所有依赖已安装
- [ ] 模型文件已下载
- [ ] 数据文件已准备
- [ ] 检索服务已启动并测试
- [ ] WandB 已登录（可选）
- [ ] screen/tmux 会话已创建
- [ ] 日志目录已创建

### 训练中监控清单

- [ ] GPU 使用率 > 80%
- [ ] 训练 loss 在下降
- [ ] rewards/mean 在上升
- [ ] test_score/mean 在上升
- [ ] 无 OOM 错误
- [ ] checkpoint 正常保存

---

**手册版本**：v1.0
**最后更新**：2025年4月
**适用版本**：Search-R1 main branch