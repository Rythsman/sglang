# 性能指标提取工具

## 概述

提供了两种方法从JSON文件中提取性能指标：
- Python脚本：`extract_metrics.py`
- Bash脚本：`extract_metrics.sh`

## 提取的指标

- `batch_size` - 批次大小
- `input_len` - 输入长度
- `output_len` - 输出长度
- `latency` - 延迟
- `ttft` - Time to First Token
- `input_throughput` - 输入吞吐量
- `output_throughput` - 输出吞吐量

## 使用方法

### Python 方法（推荐）

#### 基本用法（输出到控制台）
```bash
python3 extract_metrics.py <json_file>
```

示例：
```bash
python3 extract_metrics.py sample_metrics.json
```

#### 保存到CSV文件
```bash
python3 extract_metrics.py <json_file> <output_csv>
```

示例：
```bash
python3 extract_metrics.py sample_metrics.json output_metrics.csv
```

### Bash 方法（需要jq工具）

```bash
./extract_metrics.sh <json_file>
```

示例：
```bash
./extract_metrics.sh sample_metrics.json
```

## 输入文件格式

JSON文件每行应包含一个JSON对象，格式如下：

```json
{"summary_info": {"batch_size": 32, "input_len": 1934, "output_len": 133205, "latency": 610.8065051138401, "ttft": 1.5396787747740746, "input_throughput": 1256.106164276887, "output_throughput": 218.63163107106286}, "memory_info": {...}}
```

## 输出示例

### 控制台输出
```
Extracted Metrics:
--------------------------------------------------------------------------------
batch_size	input_len	output_len	latency	ttft	input_throughput	output_throughput
--------------------------------------------------------------------------------
32	1934	133205	610.8065051138401	1.5396787747740746	1256.106164276887	218.63163107106286
```

### CSV输出
```csv
batch_size,input_len,output_len,latency,ttft,input_throughput,output_throughput
32,1934,133205,610.8065051138401,1.5396787747740746,1256.106164276887,218.63163107106286
```

## 依赖

### Python 脚本
- Python 3.x
- 标准库（json, sys, csv）

### Bash 脚本
- Bash shell
- `jq` 工具（JSON处理器）

安装jq：
```bash
# Ubuntu/Debian
sudo apt-get install jq

# CentOS/RHEL
sudo yum install jq
```

## 文件列表

- `extract_metrics.py` - Python提取脚本
- `extract_metrics.sh` - Bash提取脚本
- `sample_metrics.json` - 示例JSON文件
- `METRICS_EXTRACTION_README.md` - 本说明文档
