#!/usr/bin/env python3
"""
AI 文本分析工具 - 命令行多模型文本分析套件

支持功能：
  - 情感分析 (sentiment)
  - 文本分类 (classify)
  - 关键词提取 (keywords)
  - 文本摘要 (summarize)
  - 综合分析 (analyze)

使用示例：
  python main.py sentiment "今天天气真好，心情非常愉快"
  python main.py classify "这个产品很好用" --categories 好评,差评,中性
  python main.py keywords "自然语言处理是人工智能的重要分支" --top 10
  python main.py summarize "长文本..." --sentences 5
  python main.py analyze "文本"
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 日志配置：优先使用 loguru，不可用时降级为 logging
# ---------------------------------------------------------------------------
try:
    from loguru import logger as _loguru_logger

    _loguru_logger.remove()
    _loguru_logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="DEBUG",
    )
    logger = _loguru_logger
    _HAS_LOGURU = True
except ImportError:
    _HAS_LOGURU = False
    logger = logging.getLogger("ai_text_analyzer")
    logger.setLevel(logging.DEBUG)
    _console = logging.StreamHandler(sys.stderr)
    _console.setLevel(logging.DEBUG)
    _console.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s - %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_console)

# ---------------------------------------------------------------------------
# 第三方库延迟导入，失败时自动降级
# ---------------------------------------------------------------------------
_HAS_TEXTBLOB = False
_HAS_VADER = False
_HAS_NLTK = False
_HAS_SKLEARN = False

try:
    from textblob import TextBlob as _TextBlob

    _HAS_TEXTBLOB = True
except ImportError:
    pass

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VaderAnalyzer

    _HAS_VADER = True
except ImportError:
    pass

try:
    import nltk

    _HAS_NLTK = True
    # 按需下载数据（静默，仅首次）
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
except ImportError:
    pass

try:
    from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer

    _HAS_SKLEARN = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# 模块级缓存
# ---------------------------------------------------------------------------
_STOP_WORDS: set[str] = set()
_VADER_INSTANCE = None


# ===================================================================
# 工具函数
# ===================================================================


def _ensure_nltk_resources() -> None:
    """确保 NLTK 必要资源已下载（静默模式）。"""
    if not _HAS_NLTK:
        return
    for resource_id in ("tokenizers/punkt_tab", "corpora/stopwords", "tokenizers/averaged_perceptron_tagger_eng"):
        try:
            nltk.data.find(resource_id)
        except LookupError:
            nltk.download(resource_id.rsplit("/", 1)[-1], quiet=True)


def _get_stop_words(language: str = "english") -> set[str]:
    """获取停用词集合，降级时返回内置的常见停用词。

    Args:
        language: NLTK 停用词语种名（默认 english）。

    Returns:
        停用词集合。
    """
    global _STOP_WORDS
    if _STOP_WORDS:
        return _STOP_WORDS
    if _HAS_NLTK:
        try:
            _STOP_WORDS = set(nltk.corpus.stopwords.words(language))
        except Exception:
            pass
    # 内置后备停用词
    if not _STOP_WORDS:
        _STOP_WORDS = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "by", "with", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "can", "could", "shall", "should", "may", "might", "must", "i", "you",
            "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
            "my", "your", "his", "its", "our", "their", "this", "that", "these",
            "those", "am", "not", "no", "nor", "so", "very", "just", "also",
            "的", "了", "在", "是", "我", "有", "不", "这", "他", "她", "它",
            "们", "和", "就", "也", "都", "要", "而", "会", "可", "对", "上",
            "下", "去", "能", "等", "之", "与", "及", "被", "让", "从", "到",
        }
    return _STOP_WORDS


def _tokenize(text: str) -> List[str]:
    """分词，优先使用 nltk，降级为简单正则。

    Args:
        text: 输入文本。

    Returns:
        分词后的 token 列表。
    """
    if _HAS_NLTK:
        try:
            return nltk.word_tokenize(text.lower())
        except Exception:
            pass
    # 降级：支持中英文混合分词
    # \w+ 匹配英文单词，[\u4e00-\u9fff] 匹配 CJK 中文字符
    tokens: List[str] = []
    for match in re.finditer(r"\w+|[^\w\s]", text.lower()):
        token = match.group()
        # CJK 统一表意文字区：每个字单独作为一个 token
        if re.match(r"[\u4e00-\u9fff]", token):
            tokens.extend(list(token))
        else:
            tokens.append(token)
    return tokens


def _split_sentences(text: str) -> List[str]:
    """分句，优先使用 nltk，降级为简单正则。

    Args:
        text: 输入文本。

    Returns:
        句子列表。
    """
    if _HAS_NLTK:
        try:
            _ensure_nltk_resources()
            return nltk.sent_tokenize(text)
        except Exception:
            pass
    # 降级：按句号/问号/感叹号/换行切分
    sentences = re.split(r"(?<=[.!?。！？])\s*|\n+", text)
    return [s.strip() for s in sentences if s.strip()]


# ===================================================================
# 1. 情感分析
# ===================================================================


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """对输入文本进行情感分析。

    优先级：VADER > TextBlob > 基于规则的情感词典。

    Args:
        text: 要进行情感分析的文本。

    Returns:
        包含以下字段的字典：
        - polarity: 极性分数（-1.0 ~ 1.0）
        - subjectivity: 主观性分数（0.0 ~ 1.0），仅 TextBlob/规则模式
        - sentiment: 情感标签（"positive", "negative", "neutral"）
        - scores: 详细分数
        - method: 使用的分析方法名称
    """
    text = text.strip()
    if not text:
        logger.warning("输入文本为空，返回中性结果")
        return {"polarity": 0.0, "subjectivity": 0.0, "sentiment": "neutral", "scores": {}, "method": "empty_input"}

    logger.info("开始情感分析，文本长度=%d", len(text))

    # --- 方法1：VADER（最强，支持中英文混合） ---
    if _HAS_VADER:
        try:
            global _VADER_INSTANCE
            if _VADER_INSTANCE is None:
                _VADER_INSTANCE = _VaderAnalyzer()
            vader_scores = _VADER_INSTANCE.polarity_scores(text)
            compound = vader_scores["compound"]
            if compound >= 0.05:
                sentiment = "positive"
            elif compound <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            logger.debug("VADER 情感分析完成: compound=%s", compound)
            return {
                "polarity": compound,
                "subjectivity": abs(compound),  # VADER 无主观性维度，用绝对极性近似
                "sentiment": sentiment,
                "scores": vader_scores,
                "method": "vaderSentiment",
            }
        except Exception as exc:
            logger.warning("VADER 分析失败: %s，降级到下一方法", exc)

    # --- 方法2：TextBlob ---
    if _HAS_TEXTBLOB:
        try:
            blob = _TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            if polarity > 0.05:
                sentiment = "positive"
            elif polarity < -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            logger.debug("TextBlob 情感分析完成: polarity=%s", polarity)
            return {
                "polarity": polarity,
                "subjectivity": subjectivity,
                "sentiment": sentiment,
                "scores": {"polarity": polarity, "subjectivity": subjectivity},
                "method": "textblob",
            }
        except Exception as exc:
            logger.warning("TextBlob 分析失败: %s，降级到基于规则的实现", exc)

    # --- 方法3：基于规则的简易情感词典（终极降级） ---
    logger.info("使用基于规则的降级情感分析")
    positive_words: set[str] = {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic", "happy", "love",
        "beautiful", "nice", "perfect", "best", "awesome", "joy", "delight", "pleased",
        "positive", "fortunate", "correct", "better", "super", "brilliant", "splendid",
        "好", "棒", "赞", "优秀", "开心", "快乐", "喜欢", "爱", "美好", "完美", "精彩",
        "漂亮", "喜欢", "高兴", "愉快", "舒适", "满意", "享受", "幸福", "幸运",
    }
    negative_words: set[str] = {
        "bad", "terrible", "awful", "horrible", "hate", "ugly", "worst", "poor",
        "sad", "angry", "upset", "disappointed", "terrible", "dreadful", "worse",
        "negative", "unfortunate", "wrong", "inferior", "broken", "damaged", "faulty",
        "坏", "差", "烂", "糟糕", "讨厌", "恶心", "伤心", "生气", "失望", "痛苦",
        "难受", "差劲", "失败", "崩溃", "绝望", "难受", "烦恼", "厌恶", "痛恨",
    }

    tokens = _tokenize(text)
    pos_count = sum(1 for w in tokens if w in positive_words)
    neg_count = sum(1 for w in tokens if w in negative_words)
    total = pos_count + neg_count
    if total == 0:
        polarity = 0.0
        sentiment = "neutral"
    else:
        polarity = (pos_count - neg_count) / total
        sentiment = "positive" if polarity > 0.2 else "negative" if polarity < -0.2 else "neutral"

    # 用正/负词占比近似主观性
    subjectivity = min(total / max(len(tokens), 1) * 2, 1.0)

    logger.debug("规则情感分析完成: polarity=%s, pos=%d, neg=%d", polarity, pos_count, neg_count)
    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "sentiment": sentiment,
        "scores": {"positive_words": pos_count, "negative_words": neg_count, "total_words": len(tokens)},
        "method": "rule_based",
    }


# ===================================================================
# 2. 文本分类
# ===================================================================


def classify_text(text: str, categories: Optional[List[str]] = None) -> Dict[str, Any]:
    """对文本进行分类。

    基于关键词/规则的文本分类器。当用户指定 categories 时按给定类别判断，
    否则使用预定义的通用类别。

    Args:
        text: 要分类的文本。
        categories: 可选的类别列表（逗号分隔），如 ["好评", "差评", "中性"]。

    Returns:
        包含以下字段的字典：
        - category: 分类标签
        - confidence: 置信度（0.0 ~ 1.0）
        - scores: 各类别得分字典
        - method: 使用的分类方法名称
    """
    text_lower = text.lower().strip()
    if not text_lower:
        logger.warning("输入文本为空，返回 'unknown'")
        return {"category": "unknown", "confidence": 0.0, "scores": {}, "method": "empty_input"}

    logger.info("开始文本分类，文本长度=%d, 类别=%s", len(text), categories)

    # 定义关键词规则（每个类别 -> 关键词列表）
    default_rules: Dict[str, List[str]] = {
        "好评": [
            "好", "棒", "赞", "优秀", "推荐", "喜欢", "满意", "不错", "值得",
            "good", "great", "excellent", "recommend", "love", "perfect", "nice",
            "方便", "实用", "性价比", "质量好", "效果好", "体验好",
        ],
        "差评": [
            "差", "烂", "糟糕", "垃圾", "后悔", "失望", "不好", "太差",
            "bad", "terrible", "awful", "horrible", "disappointed", "worst",
            "难用", "卡顿", "崩溃", "退款", "投诉", "质量问题", "体验差",
        ],
        "中性": [
            "一般", "还行", "可以", "普通", "正常", "中规中矩",
            "ok", "okay", "average", "normal", "moderate", "decent",
            "中等", "凑合", "过得去",
        ],
        "technology": [
            "computer", "software", "hardware", "programming", "code", "algorithm",
            "data", "network", "server", "database", "api", "web", "app",
            "技术", "软件", "硬件", "编程", "代码", "算法", "数据", "网络",
            "计算机", "互联网", "人工智能", "机器学习", "深度学习",
        ],
        "business": [
            "market", "finance", "investment", "revenue", "profit", "growth",
            "strategy", "management", "customer", "product", "service", "brand",
            "市场", "商业", "投资", "收入", "利润", "增长", "战略", "管理",
            "客户", "品牌", "营销", "销售", "融资", "上市",
        ],
        "education": [
            "learn", "study", "education", "school", "university", "course",
            "training", "teaching", "student", "teacher", "knowledge", "skill",
            "学习", "教育", "学校", "大学", "课程", "培训", "教学", "学生",
            "老师", "知识", "技能", "考试", "考研", "留学",
        ],
    }

    # 如果提供了自定义类别，构建临时规则（所有未知类别视为中性）
    if categories:
        rules: Dict[str, List[str]] = {}
        for cat in categories:
            if cat in default_rules:
                rules[cat] = default_rules[cat]
            else:
                # 对自定义类别用其类别名本身作为关键词
                rules[cat] = [cat.lower()]
    else:
        rules = default_rules

    # 计算每个类别的匹配得分
    tokens = _tokenize(text)
    token_set = set(tokens)
    scores: Dict[str, float] = {}
    total_matches = 0

    for category, keywords in rules.items():
        match_count = sum(1 for kw in keywords if kw in text_lower or kw in token_set)
        # 也检查在原文中的直接匹配（处理多词短语）
        phrase_matches = sum(1 for kw in keywords if " " in kw and kw in text_lower)
        score = match_count + phrase_matches * 2  # 短语匹配权重加倍
        scores[category] = score
        total_matches += score

    if total_matches == 0:
        # 完全无匹配时归入第一个类别或 "unknown"
        category = categories[0] if categories else "unknown"
        confidence = 0.0
        logger.info("未匹配到任何关键词，默认分类=%s", category)
    else:
        best_category = max(scores, key=scores.get)  # type: ignore
        category = best_category
        confidence = scores[best_category] / total_matches if total_matches > 0 else 0.0
        confidence = min(confidence, 0.95)  # 上限 0.95，避免过度自信
        logger.info("分类完成: %s (置信度=%.2f)", category, confidence)

    return {
        "category": category,
        "confidence": round(confidence, 4),
        "scores": scores,
        "method": "keyword_rule_based",
    }


# ===================================================================
# 3. 关键词提取
# ===================================================================


def extract_keywords(text: str, top_n: int = 10) -> Dict[str, Any]:
    """从文本中提取关键词。

    优先级：scikit-learn TfidfVectorizer > 基于 TF 的统计方法。

    Args:
        text: 输入文本。
        top_n: 返回的关键词数量（默认 10）。

    Returns:
        包含以下字段的字典：
        - keywords: (关键词, 权重) 列表，按权重降序
        - method: 使用的提取方法名称
        - total_tokens: 文本中唯一的 token 总数
    """
    text = text.strip()
    if not text:
        logger.warning("输入文本为空，返回空关键词列表")
        return {"keywords": [], "method": "empty_input", "total_tokens": 0}

    logger.info("开始关键词提取，文本长度=%d, top_n=%d", len(text), top_n)

    # --- 方法1：scikit-learn TfidfVectorizer ---
    if _HAS_SKLEARN:
        try:
            stop_words = _get_stop_words()
            # 对于较短的文本，使用字符 n-gram 提升效果
            vectorizer = _TfidfVectorizer(
                max_features=500,
                stop_words=list(stop_words) if stop_words else None,
                ngram_range=(1, 2),
                token_pattern=r"(?u)\b\w+\b",
            )
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            # 取 top_n 个非零权重的词
            row = tfidf_matrix[0]
            coo = row.tocoo()
            if coo.data.size == 0:
                logger.warning("TF-IDF 未提取到有效特征，使用降级方法")
                raise ValueError("No features extracted")

            word_weight: List[Tuple[str, float]] = []
            for idx, val in sorted(zip(coo.col, coo.data), key=lambda x: -x[1]):
                term = feature_names[idx]
                if len(term) > 1:  # 过滤单字
                    word_weight.append((term, round(float(val), 6)))
            keywords = word_weight[:top_n]
            logger.debug("TF-IDF 提取完成: %d 个关键词", len(keywords))
            return {
                "keywords": keywords,
                "method": "tfidf_sklearn",
                "total_tokens": len(feature_names),
            }
        except Exception as exc:
            logger.warning("TF-IDF 提取失败: %s，使用降级方法", exc)

    # --- 方法2：基于 TF（词频）的统计方法 ---
    logger.info("使用基于词频的关键词提取降级方法")
    tokens = _tokenize(text)
    stop_words = _get_stop_words()

    # 判断文本是否以中文为主（CJK 字符占比 > 30%）
    cjk_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    is_cjk_dominant = cjk_chars > len(text) * 0.3

    # 过滤停用词和单字符（英文场景下单字符多为噪声）
    filtered = [t for t in tokens if t not in stop_words and len(t) > 1]
    if not filtered:
        # 保底：允许单字符
        filtered = [t for t in tokens if t not in stop_words]
        if not filtered:
            filtered = tokens  # 终极保底

    # 对中文为主的文本，生成字符 bigram（双字词）以提升可读性
    if is_cjk_dominant and _HAS_NLTK:
        try:
            # 提取纯中文文本的相邻双字组合
            raw_chars = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
            bigrams = [raw_chars[i] + raw_chars[i + 1] for i in range(len(raw_chars) - 1)]
            # 将 bigram 加入词频统计
            counter = Counter(filtered)
            for bg in bigrams:
                counter[bg] += 1  # 每个 bigram 出现一次
            total = len(filtered) + len(bigrams)
        except Exception:
            counter = Counter(filtered)
            total = len(filtered)
    else:
        counter = Counter(filtered)
        total = len(filtered)

    keywords = [
        (word, round(count / total, 6))
        for word, count in counter.most_common(top_n)
    ]
    logger.debug("词频提取完成: %d 个关键词", len(keywords))
    return {
        "keywords": keywords,
        "method": "term_frequency",
        "total_tokens": total,
    }


# ===================================================================
# 4. 文本摘要
# ===================================================================


def summarize_text(text: str, num_sentences: int = 5) -> Dict[str, Any]:
    """对文本进行抽取式摘要。

    基于句子评分：结合 TF 关键词密度、句子长度、位置权重。

    Args:
        text: 输入文本。
        num_sentences: 摘要包含的句子数量（默认 5）。

    Returns:
        包含以下字段的字典：
        - summary: 摘要文本
        - sentences: 原始句子评分列表
        - method: 使用的摘要方法名称
        - compression_ratio: 压缩比（摘要句子数 / 原文句子数）
    """
    text = text.strip()
    if not text:
        logger.warning("输入文本为空，返回空摘要")
        return {"summary": "", "sentences": [], "method": "empty_input", "compression_ratio": 0.0}

    logger.info("开始文本摘要，文本长度=%d, 目标句数=%d", len(text), num_sentences)

    sentences = _split_sentences(text)
    if len(sentences) <= num_sentences:
        logger.info("原文句子数(%d) <= 目标句数(%d)，直接返回全文", len(sentences), num_sentences)
        # 返回全部句子
        scored = [{"sentence": s, "score": round(1.0 / len(sentences), 6), "position": i}
                  for i, s in enumerate(sentences)]
        return {
            "summary": text,
            "sentences": scored,
            "method": "full_text",
            "compression_ratio": 1.0,
        }

    # 计算词频（用于关键词密度评分）
    tokens = _tokenize(text)
    stop_words = _get_stop_words()
    filtered_tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    word_freq = Counter(filtered_tokens)
    max_freq = max(word_freq.values()) if word_freq else 1

    # 对每个句子评分
    scored_sentences: List[Dict[str, Any]] = []
    num_sent = len(sentences)

    for i, sent in enumerate(sentences):
        sent_lower = sent.lower().strip()
        sent_tokens = _tokenize(sent_lower)
        sent_len = len(sent_tokens)

        # 过短的句子降权
        if sent_len < 3:
            continue

        # 1) 关键词密度得分
        keyword_score = sum(word_freq.get(t, 0) for t in sent_tokens if t in word_freq)
        keyword_density = keyword_score / (sent_len * max_freq) if sent_len > 0 else 0

        # 2) 位置权重：首句和末句有更高权重
        position_ratio = i / num_sent
        position_weight = 1.0
        if i == 0:
            position_weight = 1.8  # 首句
        elif i == num_sent - 1:
            position_weight = 1.4  # 末句
        else:
            # 靠近开头和结尾的有稍高权重
            position_weight = 1.0 + 0.4 * (1 - abs(position_ratio - 0.5) * 2)

        # 3) 句子长度得分（适中长度更好）
        if sent_len < 5:
            length_score = 0.3
        elif sent_len > 50:
            length_score = 0.6
        else:
            length_score = 1.0 - 0.4 * abs(sent_len - 20) / 30

        # 综合评分
        total_score = keyword_density * 0.5 + position_weight * 0.3 + length_score * 0.2
        # 给标题级句子（较短且包含高频词）额外加分
        if sent_len <= 8 and keyword_density > 0.1:
            total_score *= 1.2

        scored_sentences.append({
            "sentence": sent,
            "score": round(total_score, 6),
            "position": i,
        })

    # 按分数降序排列，取 top N
    scored_sentences.sort(key=lambda x: -x["score"])
    top_sentences = scored_sentences[:num_sentences]
    # 按原文顺序重新排列
    top_sentences.sort(key=lambda x: x["position"])

    summary_text = " ".join(s["sentence"] for s in top_sentences)
    compression_ratio = round(len(top_sentences) / len(scored_sentences), 4)

    logger.info("摘要完成: %d 句 (%d -> %d, 压缩率=%.2f%%)",
                len(top_sentences), len(scored_sentences), len(top_sentences), compression_ratio * 100)

    # 所有评分句按位置返回（含低分句的引用信息）
    all_scored = sorted(scored_sentences, key=lambda x: x["position"])

    return {
        "summary": summary_text,
        "sentences": all_scored,
        "method": "extractive_scoring",
        "compression_ratio": compression_ratio,
    }


# ===================================================================
# 5. 综合分析
# ===================================================================


def analyze_full(text: str, **kwargs: Any) -> Dict[str, Any]:
    """对文本执行所有分析功能。

    Args:
        text: 输入文本。
        **kwargs: 传递给子功能的额外参数（如 top_n、num_sentences、categories）。

    Returns:
        包含情感分析、分类、关键词、摘要等结果的字典。
    """
    logger.info("开始全量分析，文本长度=%d", len(text))

    result: Dict[str, Any] = {
        "text_preview": text[:200] + "..." if len(text) > 200 else text,
        "text_length": len(text),
    }

    # 情感分析
    result["sentiment"] = analyze_sentiment(text)

    # 文本分类
    categories = kwargs.get("categories")
    result["classification"] = classify_text(text, categories=categories)

    # 关键词提取
    top_n = kwargs.get("top_n", 10)
    result["keywords"] = extract_keywords(text, top_n=top_n)

    # 文本摘要（仅当文本较长时）
    num_sentences = kwargs.get("num_sentences", 5)
    if len(text) > 150:
        result["summary"] = summarize_text(text, num_sentences=num_sentences)
    else:
        result["summary"] = {"summary": text, "method": "text_too_short", "compression_ratio": 1.0}

    logger.info("全量分析完成")
    return result


# ===================================================================
# 输出格式化
# ===================================================================


SEPARATOR = "=" * 60
SUB_SEPARATOR = "-" * 40


def _format_sentiment(result: Dict[str, Any]) -> str:
    """格式化情感分析结果。"""
    sent = result["sentiment"]
    emoji_map = {"positive": "^_^", "negative": "T_T", "neutral": "-_-"}
    emoji = emoji_map.get(sent, "?")
    lines = [
        SEPARATOR,
        "  [ 情感分析结果 ]",
        SUB_SEPARATOR,
        f"  情感倾向 : {sent.upper():<12} {emoji}",
        f"  极性分数 : {result['polarity']:<+.4f}    (范围 -1.0 ~ 1.0)",
        f"  主观性   : {result['subjectivity']:.4f}    (范围 0.0 ~ 1.0)",
        f"  分析方法 : {result['method']}",
    ]
    if result.get("scores") and result["method"] == "vaderSentiment":
        s = result["scores"]
        lines.extend([
            SUB_SEPARATOR,
            f"  VADER 详细分数:",
            f"    positive: {s.get('pos', 0):.4f}",
            f"    negative: {s.get('neg', 0):.4f}",
            f"    neutral : {s.get('neu', 0):.4f}",
            f"    compound: {s.get('compound', 0):+.4f}",
        ])
    lines.append(SEPARATOR)
    return "\n".join(lines)


def _format_classification(result: Dict[str, Any]) -> str:
    """格式化分类结果。"""
    lines = [
        SEPARATOR,
        "  [ 文本分类结果 ]",
        SUB_SEPARATOR,
        f"  分类结果 : {result['category']}",
        f"  置信度   : {result['confidence']:.2%}",
        f"  分类方法 : {result['method']}",
    ]
    if result.get("scores"):
        lines.append(SUB_SEPARATOR)
        lines.append("  各类别得分:")
        for cat, score in sorted(result["scores"].items(), key=lambda x: -x[1]):
            bar = "█" * min(int(score * 4), 30) if score > 0 else ""
            lines.append(f"    {cat:<10} {score:>4}  {bar}")
    lines.append(SEPARATOR)
    return "\n".join(lines)


def _format_keywords(result: Dict[str, Any]) -> str:
    """格式化关键词提取结果。"""
    lines = [
        SEPARATOR,
        "  [ 关键词提取结果 ]",
        SUB_SEPARATOR,
    ]
    if result.get("total_tokens"):
        lines.append(f"  候选词总数: {result['total_tokens']}")
        lines.append(f"  提取方法  : {result['method']}")
        lines.append(SUB_SEPARATOR)
        lines.append("  关键词 TOP:")
        keywords = result.get("keywords", [])
        for i, (word, weight) in enumerate(keywords, 1):
            bar_len = min(int(weight * 1000), 30)
            bar = "▓" * bar_len
            lines.append(f"  {i:>2}. {word:<15} {weight:.6f}  {bar}")
        if not keywords:
            lines.append("  (未提取到有效关键词)")
    lines.append(SEPARATOR)
    return "\n".join(lines)


def _format_summary(result: Dict[str, Any]) -> str:
    """格式化摘要结果。"""
    lines = [
        SEPARATOR,
        "  [ 文本摘要结果 ]",
        SUB_SEPARATOR,
    ]
    summary = result.get("summary", "")
    if result.get("compression_ratio") is not None:
        lines.append(f"  压缩比率 : {result['compression_ratio']:.1%}")
    lines.append(f"  摘要方法 : {result.get('method', 'N/A')}")
    lines.append(SUB_SEPARATOR)
    # 分句显示
    if summary:
        summary_sents = _split_sentences(summary)
        for i, s in enumerate(summary_sents, 1):
            lines.append(f"  [{i}] {s.strip()}")
    else:
        lines.append("  (摘要为空)")
    lines.append(SEPARATOR)
    return "\n".join(lines)


def _format_full_analysis(result: Dict[str, Any]) -> str:
    """格式化全量分析结果。"""
    parts = [
        SEPARATOR,
        "  \u2601  AI 文本综合分析报告",
        SEPARATOR,
        f"  文本预览: {result.get('text_preview', 'N/A')}",
        f"  文本长度: {result.get('text_length', 0)} 字符",
        "",
    ]
    # 情感
    if "sentiment" in result:
        parts.append(_format_sentiment(result["sentiment"]))
        parts.append("")
    # 分类
    if "classification" in result:
        parts.append(_format_classification(result["classification"]))
        parts.append("")
    # 关键词
    if "keywords" in result:
        parts.append(_format_keywords(result["keywords"]))
        parts.append("")
    # 摘要
    if "summary" in result:
        parts.append(_format_summary(result["summary"]))
    parts.append(SEPARATOR)
    return "\n".join(parts)


# ===================================================================
# CLI 入口
# ===================================================================


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        配置好的 ArgumentParser 实例。
    """
    parser = argparse.ArgumentParser(
        prog="ai-text-analyzer",
        description="多模型AI文本分析工具 - 支持情感分析、文本分类、关键词提取与摘要",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "使用示例:\n"
            "  %(prog)s sentiment \"今天天气真好\"\n"
            "  %(prog)s classify \"这个产品很好用\" --categories 好评,差评,中性\n"
            "  %(prog)s keywords \"自然语言处理是人工智能的重要分支\" --top 10\n"
            "  %(prog)s summarize \"长文本...\" --sentences 5\n"
            "  %(prog)s analyze \"文本\"\n"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # sentiment
    p_sent = subparsers.add_parser("sentiment", help="情感分析")
    p_sent.add_argument("text", type=str, help="要分析的文本")

    # classify
    p_cls = subparsers.add_parser("classify", help="文本分类")
    p_cls.add_argument("text", type=str, help="要分类的文本")
    p_cls.add_argument("--categories", type=str, default=None,
                       help="自定义类别，用逗号分隔（默认: 好评,差评,中性,technology,business,education）")

    # keywords
    p_kw = subparsers.add_parser("keywords", help="关键词提取")
    p_kw.add_argument("text", type=str, help="要提取关键词的文本")
    p_kw.add_argument("--top", type=int, default=10, help="返回的关键词数量（默认: 10）")

    # summarize
    p_sum = subparsers.add_parser("summarize", help="文本摘要")
    p_sum.add_argument("text", type=str, help="要摘要的文本")
    p_sum.add_argument("--sentences", type=int, default=5, help="摘要句子数量（默认: 5）")

    # analyze (综合)
    p_all = subparsers.add_parser("analyze", help="综合分析（情感+分类+关键词+摘要）")
    p_all.add_argument("text", type=str, help="要分析的文本")
    p_all.add_argument("--categories", type=str, default=None, help="自定义类别列表，逗号分隔")
    p_all.add_argument("--top", type=int, default=10, help="关键词数量（默认: 10）")
    p_all.add_argument("--sentences", type=int, default=5, help="摘要句子数量（默认: 5）")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """主入口函数。

    Args:
        argv: 命令行参数列表，为 None 时从 sys.argv 读取。

    Returns:
        退出码（0 表示成功）。
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    text = args.text.strip()
    if not text:
        logger.error("文本内容为空，请提供有效的文本")
        print("错误：文本内容不能为空")
        return 1

    try:
        if args.command == "sentiment":
            result = analyze_sentiment(text)
            print(_format_sentiment(result))

        elif args.command == "classify":
            categories = args.categories.split(",") if args.categories else None
            result = classify_text(text, categories=categories)
            print(_format_classification(result))

        elif args.command == "keywords":
            result = extract_keywords(text, top_n=args.top)
            print(_format_keywords(result))

        elif args.command == "summarize":
            result = summarize_text(text, num_sentences=args.sentences)
            print(_format_summary(result))

        elif args.command == "analyze":
            categories = args.categories.split(",") if args.categories else None
            result = analyze_full(
                text,
                categories=categories,
                top_n=args.top,
                num_sentences=args.sentences,
            )
            print(_format_full_analysis(result))

        return 0

    except Exception as exc:
        logger.exception("执行命令 %s 时发生错误", args.command)
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
