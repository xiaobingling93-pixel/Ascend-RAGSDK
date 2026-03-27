## 用户查询解析优化模块

### 微调

#### FineTuneArguments

##### 类功能

**功能描述**

模型微调参数类，用于配置模型微调相关参数，包括模型路径、训练数据文件路径和最大序列长度等。

**函数原型**

```python
from mx_rag.corag import FineTuneArguments
FineTuneArguments(model_name_or_path, train_file, max_len)
```

**输入参数说明**

| 参数名                   | 数据类型           | 可选/必选 | 说明                                                       |
| --------------------- | -------------- | ----- | -------------------------------------------------------- |
| model\_name\_or\_path | str            | 可选    | 预训练模型路径，仅支持本地模型，默认值为"Qwen/Qwen2.5-7B-Instruct"。 |
| train\_file           | Optional\[str] | 可选    | 训练数据文件路径（jsonl格式），默认值为"data/aligned\_train.jsonl"。       |
| max\_len              | int            | 可选    | 分词后的最大输入序列长度，默认值为2048。                                   |

#### SubqueryFineTuner

##### 类功能

**功能描述**

子查询微调器类，用于微调模型以优化子查询生成。支持NPU加速，使用前需要调用`torch.npu.set_device`设置NPU设备。

**函数原型**

```python
from mx_rag.corag import SubqueryFineTuner
SubqueryFineTuner(finetune_args, train_args)
```

**输入参数说明**

| 参数名            | 数据类型              | 可选/必选 | 说明                                      |
| -------------- | ----------------- | ----- | --------------------------------------- |
| finetune\_args | FineTuneArguments | 必选    | 模型微调参数。                                 |
| train\_args    | TrainingArguments | 必选    | 训练参数，来自transformers库的TrainingArguments。 |

**核心方法**

##### train

**功能描述**

训练模型，执行模型准备、数据准备、训练器初始化，然后执行训练并保存模型。

**函数原型**

```python
def train(self)
```

**返回值说明**

无返回值，训练完成后会保存模型和分词器到指定目录。


#### 使用示例

**基本使用示例**

```python
from mx_rag.corag import SubqueryFineTuner, FineTuneArguments
from transformers import TrainingArguments
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

# 设置NPU设备
torch.npu.set_device(0)

# 配置微调参数
finetune_args = FineTuneArguments(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    train_file="data/aligned_train.jsonl",
    max_len=2048
)

# 配置训练参数
train_args = TrainingArguments(
    output_dir="./output",
    do_train=True,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    logging_dir="./logs",
    learning_rate=1e-5,
    logging_steps=10,
    save_steps=500,
    remove_unused_columns=False
)

# 创建微调器实例
tuner = SubqueryFineTuner(finetune_args, train_args)

# 执行训练
tuner.train()
```

