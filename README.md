# AI 文本分析工具

[![CI](https://github.com/your-username/ai-text-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/ai-text-analyzer/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个功能全面的命令行 AI 文本分析工具套件，支持情感分析、文本分类、关键词提取和自动摘要等功能。

## 功能特性

| 功能 | 命令 | 说明 |
|------|------|------|
| 情感分析 | `sentiment` | 基于 VADER / TextBlob / 规则引擎的多模型情感分析 |
| 文本分类 | `classify` | 基于关键词和规则的多类别文本分类器 |
| 关键词提取 | `keywords` | TF-IDF / 词频统计关键词提取 |
| 文本摘要 | `summarize` | 基于句子评分（关键词密度 + 位置 + 长度）的抽取式摘要 |
| 综合分析 | `analyze` | 一键执行以上全部四个分析功能 |

### 架构优势

- **多模型自动降级**：优先调用 VADER / TextBlob / scikit-learn，库缺失时自动切换到基于规则的实现
- **中英文双语支持**：分词、停用词、情感词典涵盖中英文
- **美观的终端输出**：带进度条和彩色标识的格式化结果
- **完善的日志系统**：优先使用 loguru，降级到 logging

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/ai-text-analyzer.git
cd ai-text-analyzer

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -e ".[dev]"
```

### 使用示例

**情感分析：**

```bash
python main.py sentiment "今天天气真好，心情非常愉快"
```

输出示例：
```
============================================================
  [ 情感分析结果 ]
  ----------------------------------------
  情感倾向 : POSITIVE       ^_^
  极性分数 : +0.7717    (范围 -1.0 ~ 1.0)
  主观性   : 0.7717    (范围 0.0 ~ 1.0)
  分析方法 : vaderSentiment
============================================================
```

**文本分类：**

```bash
python main.py classify "这个产品很好用，推荐购买" --categories 好评,差评,中性
```

**关键词提取：**

```bash
python main.py keywords "自然语言处理是人工智能的一个重要分支，它研究如何让计算机理解和生成人类语言" --top 10
```

**文本摘要：**

```bash
python main.py summarize "长文本内容..." --sentences 5
```

**综合分析（全部功能一次输出）：**

```bash
python main.py analyze "文本内容" --categories 好评,差评,中性 --top 10 --sentences 5
```

## 技术实现

### 情感分析 (sentiment)

使用三层分析策略，逐级降级：

1. **VADER** (vaderSentiment) - 擅长中英文混合社交文本，考虑上下文强度
2. **TextBlob** - 基于模式的情感词典，提供极性 + 主观性双维度
3. **规则引擎** - 内置中英文情感词典的统计方法（终极保底）

### 文本分类 (classify)

基于关键词匹配的分类器：
- 内置预定义类别：好评、差评、中性、technology、business、education
- 支持用户自定义任意类别
- 短语匹配权重加倍，提高长关键词的准确性
- 输出各类别详细得分和可视化柱状图

### 关键词提取 (keywords)

- **主要方法**：scikit-learn TfidfVectorizer（支持 1-2 元词组）
- **降级方法**：基于词频 (TF) 的统计，含停用词过滤
- 结果按权重降序排列，带可视化条

### 文本摘要 (summarize)

抽取式摘要，综合三个维度的句子评分：

| 维度 | 权重 | 说明 |
|------|------|------|
| 关键词密度 | 50% | 包含高频关键词越多的句子得分越高 |
| 位置权重 | 30% | 首句 (1.8x) 和末句 (1.4x) 权重更高 |
| 长度得分 | 20% | 句子长度适中（约 20 词）得分最高 |

## 项目结构

```
ai-text-analyzer/
├── main.py              # 主程序入口（包含所有分析逻辑）
├── requirements.txt     # 生产依赖
├── pyproject.toml       # 项目配置（pytest / flake8）
├── README.md            # 本文件
├── .gitignore           # Git 忽略规则
├── .github/
│   └── workflows/
│       └── ci.yml       # GitHub Actions CI 配置
└── tests/
    ├── __init__.py
    └── test_main.py     # 单元测试（20+ 测试用例）
```

## 开发指南

### 运行测试

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行全部测试
pytest

# 运行测试并生成覆盖率报告
pytest --cov=main --cov-report=term-missing

# 运行特定测试类
pytest tests/test_main.py -k TestSentiment -v
```

### 代码检查

```bash
# 静态检查
flake8 main.py tests/
```

### 依赖关系

- **textblob** >= 0.17 - TextBlob 情感分析引擎
- **nltk** >= 3.8 - 自然语言工具包（分词、停用词）
- **vaderSentiment** >= 3.3 - VADER 情感分析器
- **scikit-learn** >= 1.0 - TF-IDF 关键词提取
- **loguru** >= 0.7 - 结构化日志

> **注意**：所有依赖均为可选，任意库缺失时工具会自动降级到基于规则的实现，功能不受影响。

## 持续集成

项目配置了 GitHub Actions CI，在每次推送时自动执行：

1. Python 3.10 / 3.11 / 3.12 矩阵测试
2. flake8 代码风格检查
3. pytest 单元测试（含覆盖率报告）
4. Codecov 覆盖率上传

详情见 [ci.yml](.github/workflows/ci.yml)。

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。
