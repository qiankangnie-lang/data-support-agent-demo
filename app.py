from __future__ import annotations

import os
import json
import re
import html
from pathlib import Path
from typing import Any
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


load_dotenv()

APP_TITLE = "数据支持智能分诊与问数助手"
PROMPT_DIR = Path("prompts")
LOG_DIR = Path("conversation_logs")
REQUIRED_SHEET = "问题记录数据"
API_KEY_MISSING_MESSAGE = "当前未配置 OpenAI API Key，无法调用运行时大模型。请在 .env 文件中配置 OPENAI_API_KEY。"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")

QUESTION_TYPES = {
    "权限申请类",
    "新增数据需求类",
    "数据异常类",
    "口径确认类",
    "使用咨询类",
    "数据分析问数类",
    "其他",
}

SUPPORT_PROMPTS = {
    "权限申请类": "permission_prompt.txt",
    "新增数据需求类": "new_requirement_prompt.txt",
    "数据异常类": "anomaly_prompt.txt",
    "口径确认类": "caliber_prompt.txt",
    "使用咨询类": "consultation_prompt.txt",
}

ANALYSIS_INTENTS = {
    "issue_type_count",
    "department_count",
    "avg_resolve_hours_by_type",
    "sensitive_permission_count",
    "avg_missing_fields_by_type",
    "unsupported",
}

COLUMN_MAPPING = {
    "问题编号": "issue_id",
    "提交日期": "submit_date",
    "提交部门": "department",
    "问题类型": "issue_type",
    "涉及对象": "object_name",
    "处理状态": "status",
    "优先级": "priority",
    "处理角色": "handler_role",
    "缺失信息数量": "missing_fields_count",
    "解决时长": "resolve_hours",
    "是否涉及敏感数据": "is_sensitive",
    "满意度": "satisfaction",
}

PROMPT_LABELS = {
    "意图识别 Prompt": "intent_classification_prompt.txt",
    "权限申请 Prompt": "permission_prompt.txt",
    "新增数据需求 Prompt": "new_requirement_prompt.txt",
    "数据异常 Prompt": "anomaly_prompt.txt",
    "口径确认 Prompt": "caliber_prompt.txt",
    "使用咨询 Prompt": "consultation_prompt.txt",
    "问数分析 Prompt": "data_analysis_prompt.txt",
}

BUILT_IN_TESTS = [
    ("我需要导出会员手机号和收货地址，用于短信触达活动。", "权限申请类"),
    ("我想新增一个复购率字段，用在运营日报里。", "新增数据需求类"),
    ("今天运营日报没有更新。", "数据异常类"),
    ("复购率这个指标到底怎么算？运营和数据团队说法不一样。", "口径确认类"),
    ("DAU 是什么意思？", "使用咨询类"),
    ("哪类问题最多？", "数据分析问数类"),
]


def load_prompt(prompt_name: str) -> str:
    prompt_path = PROMPT_DIR / prompt_name
    if not prompt_path.exists():
        return "该 Prompt 文件暂未找到"
    return prompt_path.read_text(encoding="utf-8")


def call_llm(system_prompt: str, user_prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError(API_KEY_MISSING_MESSAGE)

    try:
        client_kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            client_kwargs["base_url"] = OPENAI_BASE_URL
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as exc:
        raise RuntimeError(f"OpenAI API 调用失败：{exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"大模型调用失败：{exc}") from exc


def normalize_question_type(raw_type: str) -> str:
    cleaned = raw_type.strip().replace("`", "").replace("：", "").replace(":", "")
    for question_type in QUESTION_TYPES:
        if question_type in cleaned:
            return question_type
    return "其他"


def normalize_analysis_intent(raw_intent: str) -> str:
    cleaned = raw_intent.strip().replace("`", "")
    for intent in ANALYSIS_INTENTS:
        if intent in cleaned:
            return intent
    return "unsupported"


def load_issue_sheet(uploaded_file: Any) -> tuple[pd.DataFrame | None, str | None]:
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        if REQUIRED_SHEET not in excel_file.sheet_names:
            return None, "未找到 ‘问题记录数据’ sheet，请检查上传文件。"
        df = pd.read_excel(uploaded_file, sheet_name=REQUIRED_SHEET)
        return normalize_columns(df), None
    except Exception as exc:
        return None, f"Excel 读取失败：{exc}"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.rename(columns=COLUMN_MAPPING).copy()
    for column in ["resolve_hours", "missing_fields_count", "satisfaction"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if "submit_date" in normalized.columns:
        normalized["submit_date"] = pd.to_datetime(normalized["submit_date"], errors="coerce")
    return normalized


def build_recent_context() -> str:
    turns = st.session_state.get("conversation", [])[-3:]
    if not turns:
        return "暂无历史上下文。"

    lines = []
    for index, turn in enumerate(turns, start=1):
        lines.append(
            "\n".join(
                [
                    f"最近第 {index} 轮：",
                    f"用户输入：{turn.get('user_input', '')}",
                    f"意图识别结果：{turn.get('question_type', '')}",
                    f"助手回答摘要：{turn.get('assistant_output', '')[:500]}",
                ]
            )
        )
    return "\n\n".join(lines)


def build_contextual_user_prompt(question: str, instruction: str = "") -> str:
    context = build_recent_context()
    parts = [
        "以下是最近 3 轮会话上下文。用户可能在当前问题中补充上一轮信息，请结合上下文理解，但仍以当前输入为主。",
        context,
        f"当前用户输入：{question}",
    ]
    if instruction:
        parts.append(instruction)
    return "\n\n".join(parts)


def classify_with_llm(question: str) -> tuple[str, str, str]:
    prompt_name = "intent_classification_prompt.txt"
    system_prompt = load_prompt(prompt_name)
    raw_result = call_llm(system_prompt, build_contextual_user_prompt(question, "只输出分类标签本身。"))
    return normalize_question_type(raw_result), raw_result, prompt_name


def process_support_question(question: str, question_type: str) -> tuple[str, str]:
    prompt_name = SUPPORT_PROMPTS[question_type]
    system_prompt = load_prompt(prompt_name)
    user_prompt = build_contextual_user_prompt(question, f"当前问题已被识别为：{question_type}。请输出结构化处理结果。")
    return call_llm(system_prompt, user_prompt), prompt_name


def identify_analysis_intent(question: str) -> tuple[str, str, str]:
    prompt_name = "data_analysis_prompt.txt"
    system_prompt = load_prompt(prompt_name)
    user_prompt = "\n\n".join(
        [
            "请只输出一个分析意图标签，不要输出解释。",
            "可选标签：issue_type_count, department_count, avg_resolve_hours_by_type, sensitive_permission_count, avg_missing_fields_by_type, unsupported。",
            build_contextual_user_prompt(question, "请只输出一个分析意图标签。"),
        ]
    )
    raw_intent = call_llm(system_prompt, user_prompt)
    return normalize_analysis_intent(raw_intent), raw_intent, prompt_name


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"上传数据缺少必要字段：{', '.join(missing)}")


def run_pandas_analysis(intent: str, df: pd.DataFrame) -> dict[str, Any]:
    if intent == "issue_type_count":
        ensure_columns(df, ["issue_type"])
        counts = df["issue_type"].value_counts().sort_values(ascending=False)
        return {
            "analysis_name": "问题类型数量分布",
            "top_item": counts.index[0] if not counts.empty else "无",
            "top_value": int(counts.iloc[0]) if not counts.empty else 0,
            "result_table": counts.reset_index().rename(columns={"issue_type": "问题类型", "count": "数量"}),
            "result_dict": counts.to_dict(),
        }

    if intent == "department_count":
        ensure_columns(df, ["department"])
        counts = df["department"].value_counts().sort_values(ascending=False)
        return {
            "analysis_name": "部门提交问题数量分布",
            "top_item": counts.index[0] if not counts.empty else "无",
            "top_value": int(counts.iloc[0]) if not counts.empty else 0,
            "result_table": counts.reset_index().rename(columns={"department": "提交部门", "count": "数量"}),
            "result_dict": counts.to_dict(),
        }

    if intent == "avg_resolve_hours_by_type":
        ensure_columns(df, ["issue_type", "resolve_hours"])
        grouped = df.groupby("issue_type")["resolve_hours"].mean().sort_values(ascending=False).round(2)
        return {
            "analysis_name": "各问题类型平均解决时长",
            "top_item": grouped.index[0] if not grouped.empty else "无",
            "top_value": float(grouped.iloc[0]) if not grouped.empty else 0,
            "result_table": grouped.reset_index().rename(columns={"issue_type": "问题类型", "resolve_hours": "平均解决时长"}),
            "result_dict": grouped.to_dict(),
        }

    if intent == "sensitive_permission_count":
        ensure_columns(df, ["issue_type", "is_sensitive"])
        filtered = df[(df["issue_type"] == "权限申请类") & (df["is_sensitive"] == "是")]
        return {
            "analysis_name": "涉及敏感数据的权限申请数量",
            "top_item": "敏感权限申请",
            "top_value": int(len(filtered)),
            "result_table": pd.DataFrame([{"指标": "涉及敏感数据的权限申请数量", "数量": int(len(filtered))}]),
            "result_dict": {"涉及敏感数据的权限申请数量": int(len(filtered))},
        }

    if intent == "avg_missing_fields_by_type":
        ensure_columns(df, ["issue_type", "missing_fields_count"])
        grouped = df.groupby("issue_type")["missing_fields_count"].mean().sort_values(ascending=False).round(2)
        return {
            "analysis_name": "各问题类型平均缺失信息数量",
            "top_item": grouped.index[0] if not grouped.empty else "无",
            "top_value": float(grouped.iloc[0]) if not grouped.empty else 0,
            "result_table": grouped.reset_index().rename(columns={"issue_type": "问题类型", "missing_fields_count": "平均缺失信息数量"}),
            "result_dict": grouped.to_dict(),
        }

    return {
        "analysis_name": "暂不支持的问数意图",
        "top_item": "unsupported",
        "top_value": 0,
        "result_table": pd.DataFrame([{"说明": "当前 MVP 暂不支持该问数问题"}]),
        "result_dict": {"unsupported": "当前 MVP 暂不支持该问数问题"},
    }


def explain_analysis_with_llm(question: str, intent: str, stats: dict[str, Any]) -> str:
    system_prompt = load_prompt("data_analysis_prompt.txt")
    user_prompt = (
        f"最近 3 轮会话上下文：\n{build_recent_context()}\n\n"
        "请基于以下 Pandas 真实统计结果，输出最终分析结果。\n"
        "必须使用以下格式：\n"
        "分析主题：\n统计结果：\n业务解释：\n优化建议：\n\n"
        f"用户问题：{question}\n"
        f"分析意图：{intent}\n"
        f"Pandas 统计结果：{stats['result_dict']}\n"
        f"最高项：{stats['top_item']}\n"
        f"最高值：{stats['top_value']}"
    )
    return call_llm(system_prompt, user_prompt)


def process_analysis_question(question: str, df: pd.DataFrame) -> dict[str, Any]:
    intent, raw_intent, prompt_name = identify_analysis_intent(question)
    stats = run_pandas_analysis(intent, df)
    llm_explanation = explain_analysis_with_llm(question, intent, stats)
    return {
        "prompt_name": prompt_name,
        "intent": intent,
        "raw_intent": raw_intent,
        "stats": stats,
        "llm_explanation": llm_explanation,
    }


def process_other_question(question: str) -> str:
    system_prompt = (
        "你是企业数据支持场景中的澄清助手。用户问题未能归入既定分类。"
        "请用简洁中文提示用户补充问题背景、涉及对象、期望结果或异常现象。"
    )
    user_prompt = build_contextual_user_prompt(question, "请生成澄清建议。")
    return call_llm(system_prompt, user_prompt)


def sanitize_sensitive_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    sanitized = value
    sanitized = re.sub(
        r"\b[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b",
        "【身份证号】",
        sanitized,
    )
    sanitized = re.sub(r"\b\d{16,19}\b", "【银行卡号】", sanitized)
    sanitized = re.sub(r"1[3-9]\d{9}", "【手机号】", sanitized)
    sanitized = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "【邮箱】", sanitized)
    return sanitized


def sanitize_log_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: sanitize_sensitive_text(value) for key, value in entry.items()}


def generate_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def generate_session_title(question_type: str, first_question: str) -> str:
    prefixes = {
        "权限申请类": "权限申请",
        "新增数据需求类": "新增需求",
        "数据异常类": "数据异常",
        "口径确认类": "口径确认",
        "数据分析问数类": "问数分析",
    }
    cleaned_question = re.sub(r"\s+", " ", first_question).strip()
    short_question = cleaned_question[:20] or "新会话"
    prefix = prefixes.get(question_type)
    if prefix:
        return f"{prefix}：{short_question}"
    return short_question


def reset_current_session() -> None:
    now = datetime.now().isoformat(timespec="seconds")
    st.session_state.session_id = generate_session_id()
    st.session_state.session_title = "新会话"
    st.session_state.session_created_at = now
    st.session_state.session_updated_at = now
    st.session_state.conversation = []
    st.session_state.execution_chain = {}
    st.session_state.result_text = ""
    st.session_state.last_saved_hint = "未保存"
    st.session_state.view = "home"
    st.session_state.current_session_file = None
    clear_uploaded_excel_state()


def clear_uploaded_excel_state() -> None:
    for key in ["uploaded_df", "uploaded_df_session_id", "uploaded_excel_chat", "uploaded_excel_home"]:
        if key in st.session_state:
            del st.session_state[key]


def save_conversation_log() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"session_{timestamp}.json"
    messages = []
    for turn in st.session_state.get("conversation", []):
        messages.append(
            sanitize_log_entry(
                {
                    "timestamp": turn.get("timestamp", ""),
                    "user_input": turn.get("user_input", ""),
                    "question_type": turn.get("question_type", ""),
                    "execution_path": turn.get("execution_path", ""),
                    "assistant_output": turn.get("assistant_output", ""),
                }
            )
        )

    payload = {
        "session_id": st.session_state.get("session_id", generate_session_id()),
        "title": sanitize_sensitive_text(st.session_state.get("session_title", "新会话")),
        "created_at": st.session_state.get("session_created_at", now.isoformat(timespec="seconds")),
        "updated_at": now.isoformat(timespec="seconds"),
        "messages": messages,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    st.session_state.session_updated_at = payload["updated_at"]
    st.session_state.last_saved_hint = now.strftime("%Y/%m/%d %H:%M:%S")
    st.session_state.current_session_file = str(path)
    return path


def add_conversation_turn(question: str, chain: dict[str, Any], answer: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    if not st.session_state.get("session_title") or st.session_state.session_title == "新会话":
        st.session_state.session_title = generate_session_title(chain.get("question_type", ""), question)
    st.session_state.session_updated_at = now
    st.session_state.conversation.append(
        {
            "timestamp": now,
            "user_input": question,
            "question_type": chain.get("question_type", ""),
            "execution_path": chain.get("path", ""),
            "assistant_output": answer,
            "execution_chain": chain,
        }
    )


def list_recent_session_files(limit: int = 5) -> list[Path]:
    if not LOG_DIR.exists():
        return []
    files = sorted(LOG_DIR.glob("session_*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[:limit]


def load_session_from_file(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    messages = []
    for message in payload.get("messages", []):
        chain = {
            "question": message.get("user_input", ""),
            "question_type": message.get("question_type", ""),
            "path": message.get("execution_path", ""),
            "llm_result": message.get("assistant_output", ""),
            "mode": "loaded",
        }
        messages.append(
            {
                "timestamp": message.get("timestamp", ""),
                "user_input": message.get("user_input", ""),
                "question_type": message.get("question_type", ""),
                "execution_path": message.get("execution_path", ""),
                "assistant_output": message.get("assistant_output", ""),
                "execution_chain": chain,
            }
        )

    st.session_state.session_id = payload.get("session_id", generate_session_id())
    st.session_state.session_title = payload.get("title", path.stem)
    st.session_state.session_created_at = payload.get("created_at", "")
    st.session_state.session_updated_at = payload.get("updated_at", "")
    st.session_state.conversation = messages
    st.session_state.execution_chain = messages[-1]["execution_chain"] if messages else {}
    st.session_state.result_text = messages[-1]["assistant_output"] if messages else ""
    st.session_state.view = "chat" if messages else "home"
    st.session_state.last_saved_hint = payload.get("updated_at", "已加载")
    st.session_state.current_session_file = str(path)
    clear_uploaded_excel_state()


def clear_current_session(delete_history_file: bool = False) -> None:
    current_file = st.session_state.get("current_session_file")
    if delete_history_file and current_file:
        path = Path(current_file)
        try:
            resolved_log_dir = LOG_DIR.resolve()
            resolved_path = path.resolve()
            if resolved_path.parent == resolved_log_dir and resolved_path.exists():
                resolved_path.unlink()
        except OSError as exc:
            st.warning(f"删除历史会话失败：{exc}")

    now = datetime.now().isoformat(timespec="seconds")
    st.session_state.conversation = []
    st.session_state.execution_chain = {}
    st.session_state.result_text = ""
    st.session_state.session_id = generate_session_id()
    st.session_state.session_title = "新会话"
    st.session_state.session_created_at = now
    st.session_state.session_updated_at = now
    st.session_state.last_saved_hint = "未保存"
    st.session_state.current_session_file = None
    st.session_state.view = "home"
    clear_uploaded_excel_state()


def render_sidebar_session_manager() -> None:
    with st.sidebar:
        st.markdown("### 会话管理")
        st.caption(st.session_state.get("session_title", "新会话"))

        if st.button("新建会话", use_container_width=True):
            reset_current_session()
            st.rerun()

        if st.button("保存当前会话", use_container_width=True):
            if not st.session_state.get("conversation"):
                st.warning("当前没有可保存的会话。")
            else:
                saved_path = save_conversation_log()
                st.success(f"已保存：{saved_path.name}")

        if st.button("清空当前会话", use_container_width=True):
            clear_current_session(delete_history_file=True)
            st.rerun()

        st.markdown("### 最近历史会话")
        recent_files = list_recent_session_files()
        if not recent_files:
            st.caption("暂无历史会话")
        for path in recent_files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                title = payload.get("title", path.stem)
            except Exception:
                title = path.stem
            if st.button(title[:28], key=f"history_{path.name}", use_container_width=True):
                load_session_from_file(path)
                st.rerun()


def process_agent_turn(question: str, df: pd.DataFrame | None) -> tuple[str, dict[str, Any]]:
    question_type, raw_intent, intent_prompt = classify_with_llm(question)
    chain = {
        "question": question,
        "question_type": question_type,
        "raw_intent_result": raw_intent,
    }

    if question_type in SUPPORT_PROMPTS:
        result, prompt_name = process_support_question(question, question_type)
        chain.update(
            {
                "mode": "support",
                "path": f"读取 {intent_prompt} → LLM 意图识别 → 读取 {prompt_name} → LLM 生成结构化分诊结果",
                "prompt_name": prompt_name,
                "llm_result": result,
            }
        )
        return result, chain

    if question_type == "数据分析问数类":
        if df is None:
            raise ValueError("请先上传包含“问题记录数据”sheet 的 Excel 文件，再进行问数分析。")
        analysis_result = process_analysis_question(question, df)
        result = analysis_result["llm_explanation"]
        chain.update(
            {
                "mode": "analysis",
                "path": "读取 intent_classification_prompt.txt → LLM 意图识别 → 读取 data_analysis_prompt.txt → LLM 识别分析意图 → Pandas 真实统计 → LLM 生成业务解释与优化建议",
                "prompt_name": analysis_result["prompt_name"],
                "analysis_intent": analysis_result["intent"],
                "raw_analysis_intent": analysis_result["raw_intent"],
                "stats": analysis_result["stats"],
                "llm_result": result,
            }
        )
        return result, chain

    result = process_other_question(question)
    chain.update(
        {
            "mode": "other",
            "path": "读取 intent_classification_prompt.txt → LLM 意图识别为其他 → LLM 生成澄清建议",
            "llm_result": result,
        }
    )
    return result, chain


def render_text_result(text: str) -> None:
    st.markdown(text.replace("\n", "  \n"))


def render_execution_chain(chain: dict[str, Any]) -> None:
    st.subheader("执行链路展示区")
    if not chain:
        st.info("提交问题后将在这里展示 Agent 执行链路。")
        return

    st.markdown("**用户原始问题**")
    st.write(chain.get("question", ""))
    st.markdown("**LLM 意图识别结果**")
    st.write(chain.get("question_type", ""))
    st.markdown("**执行路径**")
    st.write(chain.get("path", ""))

    if chain.get("mode") == "analysis":
        st.markdown("**LLM 解析出的分析意图**")
        st.write(chain.get("analysis_intent", ""))
        st.markdown("**Pandas 统计结果**")
        stats = chain.get("stats")
        if stats:
            st.dataframe(stats["result_table"], use_container_width=True)
            st.json(stats["result_dict"])
        st.markdown("**LLM 解释结果**")
        render_text_result(chain.get("llm_result", ""))
    elif chain.get("mode") == "support":
        st.markdown("**读取的 Prompt 文件**")
        st.write(chain.get("prompt_name", ""))
        st.markdown("**LLM 输出结果**")
        render_text_result(chain.get("llm_result", ""))
    elif chain.get("mode") == "other":
        st.markdown("**LLM 澄清建议**")
        render_text_result(chain.get("llm_result", ""))


def render_data_preview(df: pd.DataFrame | None) -> None:
    st.subheader("上传数据预览区")
    if df is None:
        st.info("上传 Excel 后将展示“问题记录数据”sheet 的前 10 行。")
        return
    st.dataframe(df.head(10), use_container_width=True)


def render_charts(df: pd.DataFrame | None) -> None:
    st.subheader("基础图表展示区")
    if df is None:
        st.info("上传 Excel 后将展示基础统计图表。")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**问题类型分布**")
        if "issue_type" in df.columns:
            st.bar_chart(df["issue_type"].value_counts())
        else:
            st.warning("缺少 issue_type / 问题类型 字段。")

    with col2:
        st.markdown("**部门问题数量**")
        if "department" in df.columns:
            st.bar_chart(df["department"].value_counts())
        else:
            st.warning("缺少 department / 提交部门 字段。")

    with col3:
        st.markdown("**平均解决时长**")
        if {"issue_type", "resolve_hours"}.issubset(df.columns):
            st.bar_chart(df.groupby("issue_type")["resolve_hours"].mean().sort_values(ascending=False))
        else:
            st.warning("缺少 issue_type 或 resolve_hours 字段。")


def run_builtin_tests() -> pd.DataFrame:
    prompt = load_prompt("intent_classification_prompt.txt")
    rows = []
    for question, expected in BUILT_IN_TESTS:
        try:
            raw_result = call_llm(prompt, question)
            actual = normalize_question_type(raw_result)
            passed = "通过" if actual == expected else "未通过"
        except RuntimeError as exc:
            actual = str(exc)
            passed = "未执行"
        rows.append(
            {
                "测试问题": question,
                "期望分类": expected,
                "实际分类": actual,
                "是否通过": passed,
            }
        )
    return pd.DataFrame(rows)


def render_builtin_tests() -> None:
    st.subheader("内置测试集验证区")
    st.caption("点击后会调用 OpenAI API 做真实意图识别。")
    if st.button("运行六条内置测试"):
        with st.spinner("正在调用大模型验证内置测试集..."):
            st.dataframe(run_builtin_tests(), use_container_width=True)


def render_prompt_expander() -> None:
    st.subheader("Prompt 规则查看")
    with st.expander("查看全部 Prompt 规则"):
        for label, filename in PROMPT_LABELS.items():
            st.markdown(f"**{label}**")
            prompt_text = load_prompt(filename)
            if prompt_text == "该 Prompt 文件暂未找到":
                st.warning(prompt_text)
            else:
                st.text_area(label, prompt_text, height=220, key=f"prompt_{filename}")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #1f2a44;
            --muted: #6d7690;
            --line: rgba(113, 128, 150, 0.18);
            --panel: rgba(255, 255, 255, 0.72);
            --panel-strong: rgba(255, 255, 255, 0.88);
            --accent: #3c6df0;
            --accent-soft: rgba(60, 109, 240, 0.10);
        }

        .stApp {
            background:
                linear-gradient(145deg, rgba(232, 222, 255, 0.84) 0%, rgba(232, 241, 255, 0.76) 38%, rgba(218, 244, 247, 0.88) 100%),
                linear-gradient(180deg, #f9fbff 0%, #eef6fb 100%);
            color: var(--ink);
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stAppViewContainer"] > .main {
            padding-top: 0;
        }

        .block-container {
            max-width: 980px;
            padding-top: 76px;
            padding-bottom: 56px;
        }

        .assistant-shell {
            margin: 0 auto;
        }

        .hero-mark {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 22px;
        }

        .bot-icon {
            width: 46px;
            height: 46px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            background: linear-gradient(135deg, #ff6d8f, #fb3e72);
            color: white;
            font-size: 24px;
            box-shadow: 0 16px 36px rgba(245, 68, 119, 0.22);
        }

        .hero-title {
            font-size: 34px;
            line-height: 1.18;
            font-weight: 760;
            letter-spacing: 0;
            color: var(--ink);
            margin: 0;
        }

        .hero-subtitle {
            margin: 8px 0 0 60px;
            color: var(--muted);
            font-size: 15px;
            line-height: 1.8;
        }

        .assistant-panel {
            margin-top: 24px;
            border: 1px solid rgba(255, 255, 255, 0.72);
            background: var(--panel);
            box-shadow: 0 28px 80px rgba(45, 60, 95, 0.12);
            backdrop-filter: blur(18px);
            border-radius: 18px;
            padding: 10px 10px 22px;
        }

        .panel-inner {
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 22px;
        }

        .question-block {
            margin-top: 26px;
            margin-bottom: 22px;
        }

        .capability-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin: 4px 0 14px;
        }

        .capability {
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 14px 15px;
            background: rgba(255, 255, 255, 0.72);
        }

        .capability.active {
            border-color: rgba(60, 109, 240, 0.22);
            background: var(--accent-soft);
        }

        .capability-title {
            color: var(--ink);
            font-weight: 700;
            font-size: 15px;
            margin-bottom: 5px;
        }

        .capability-text {
            color: var(--muted);
            font-size: 13px;
            line-height: 1.55;
        }

        .example-list {
            background: rgba(255, 255, 255, 0.60);
            border: 1px solid var(--line);
            border-radius: 12px;
            margin: 10px 0 16px;
            overflow: hidden;
        }

        .example-line {
            padding: 12px 16px;
            color: #43506d;
            font-size: 14px;
            border-bottom: 1px solid rgba(113, 128, 150, 0.13);
        }

        .example-line:last-child {
            border-bottom: 0;
        }

        .example-line span {
            color: var(--accent);
            margin-right: 8px;
        }

        .upload-note {
            color: var(--muted);
            font-size: 13px;
            margin: 4px 0 10px;
        }

        .upload-title {
            color: #34405b;
            font-size: 14px;
            font-weight: 680;
            margin: 14px 0 2px;
        }

        .result-card {
            margin-top: 20px;
            padding: 20px 22px;
            border-radius: 14px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 18px 48px rgba(45, 60, 95, 0.08);
        }

        .result-title {
            font-weight: 760;
            color: var(--ink);
            margin-bottom: 10px;
        }

        .chat-topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            margin-bottom: 18px;
            border-bottom: 1px solid rgba(113, 128, 150, 0.18);
            padding-bottom: 14px;
        }

        .chat-title {
            font-weight: 760;
            color: var(--ink);
            font-size: 18px;
        }

        .chat-subtitle {
            color: var(--muted);
            font-size: 13px;
            margin-top: 3px;
        }

        .chat-window {
            max-width: 820px;
            margin: 0 auto 18px;
        }

        .message {
            display: flex;
            margin: 18px 0;
        }

        .message.user {
            justify-content: flex-end;
        }

        .bubble {
            max-width: 78%;
            border-radius: 16px;
            padding: 14px 16px;
            line-height: 1.78;
            font-size: 15px;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        .user .bubble {
            background: linear-gradient(135deg, #2f6cf6, #5b8cff);
            color: white;
            box-shadow: 0 18px 34px rgba(60, 109, 240, 0.18);
        }

        .assistant .bubble {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(113, 128, 150, 0.18);
            color: #24304a;
            box-shadow: 0 18px 42px rgba(45, 60, 95, 0.08);
        }

        .chat-input-wrap {
            max-width: 820px;
            margin: 22px auto 0;
        }

        .meta-pill {
            display: inline-flex;
            align-items: center;
            border: 1px solid rgba(113, 128, 150, 0.18);
            background: rgba(255, 255, 255, 0.72);
            color: #64708a;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 12px;
            margin-bottom: 8px;
        }

        div[data-testid="stFileUploader"] {
            padding: 2px 0 8px;
        }

        div[data-testid="stForm"] {
            border: 1px solid rgba(113, 128, 150, 0.24);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.40);
            box-shadow: 0 20px 54px rgba(45, 60, 95, 0.08);
            padding: 18px 18px 20px;
        }

        div[data-testid="stTextArea"] textarea {
            border-radius: 16px;
            border: 1px solid rgba(113, 128, 150, 0.24);
            min-height: 112px;
            font-size: 15px;
            background: rgba(255, 255, 255, 0.86);
            box-shadow: inset 0 1px 2px rgba(30, 41, 59, 0.03);
        }

        .stButton > button,
        [data-testid="stFormSubmitButton"] button {
            width: auto;
            border-radius: 14px;
            min-height: 46px;
            padding: 0 24px;
            border: 0;
            background: linear-gradient(135deg, #2f6cf6, #5b8cff);
            color: #fff;
            font-weight: 720;
            box-shadow: 0 18px 36px rgba(60, 109, 240, 0.24);
        }

        .footer-note {
            text-align: center;
            color: #9aa3b6;
            font-size: 12px;
            margin-top: 18px;
        }

        @media (max-width: 760px) {
            .block-container {
                padding-top: 36px;
            }
            .hero-title {
                font-size: 25px;
            }
            .hero-subtitle {
                margin-left: 0;
            }
            .capability-row {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_minimal_header() -> None:
    st.markdown(
        """
        <div class="assistant-shell">
            <div class="hero-mark">
                <div class="bot-icon">✦</div>
                <div>
                    <h1 class="hero-title">数据支持智能分诊与问数助手</h1>
                </div>
            </div>
            <div class="hero-subtitle">
                面向企业内部数据支持场景，统一处理数据分诊、历史问数和基础问答。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_capabilities() -> None:
    st.markdown(
        """
        <div class="capability-row">
            <div class="capability active">
                <div class="capability-title">数据分诊</div>
                <div class="capability-text">权限申请、新增需求、数据异常、口径确认与使用咨询。</div>
            </div>
            <div class="capability">
                <div class="capability-title">历史问数</div>
                <div class="capability-text">基于上传 Excel，用 Pandas 真实统计问题类型、部门和处理效率。</div>
            </div>
            <div class="capability">
                <div class="capability-title">基础问答</div>
                <div class="capability-text">解释常见指标含义、使用说明，并给出下一步补充建议。</div>
            </div>
        </div>
        <div class="example-list">
            <div class="example-line"><span>◆</span>我需要导出会员手机号和收货地址，用于短信触达活动。</div>
            <div class="example-line"><span>◆</span>今天运营日报没有更新。</div>
            <div class="example-line"><span>◆</span>哪类问题平均解决时长最长？</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compact_status(df: pd.DataFrame | None, error: str | None) -> None:
    if not OPENAI_API_KEY:
        st.error(API_KEY_MISSING_MESSAGE)
    elif error:
        st.error(error)
    elif df is not None:
        st.success(f"已读取“{REQUIRED_SHEET}”sheet，共 {len(df)} 行数据。")


def render_result_card() -> None:
    if not st.session_state.result_text:
        return
    st.markdown('<div class="result-card"><div class="result-title">分析结果</div>', unsafe_allow_html=True)
    render_text_result(st.session_state.result_text)
    st.markdown("</div>", unsafe_allow_html=True)


def render_debug_details() -> None:
    if not st.session_state.execution_chain:
        return
    with st.expander("查看执行详情"):
        render_execution_chain(st.session_state.execution_chain)


def render_chat_messages() -> None:
    st.markdown('<div class="chat-window">', unsafe_allow_html=True)
    for index, turn in enumerate(st.session_state.get("conversation", []), start=1):
        user_text = html.escape(turn.get("user_input", ""))
        assistant_text = html.escape(turn.get("assistant_output", ""))
        assistant_text = assistant_text.replace("\n", "<br>")
        question_type = html.escape(turn.get("question_type", ""))

        st.markdown(
            f'<div class="message user"><div class="bubble">{user_text}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="message assistant"><div><div class="meta-pill">{question_type}</div>'
            f'<div class="bubble">{assistant_text}</div></div></div>',
            unsafe_allow_html=True,
        )
        with st.expander(f"查看第 {index} 轮执行详情", expanded=False):
            render_execution_chain(turn.get("execution_chain", {}))
    st.markdown("</div>", unsafe_allow_html=True)


def render_conversation_controls() -> None:
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("清空当前会话"):
            st.session_state.conversation = []
            st.session_state.execution_chain = {}
            st.session_state.result_text = ""
            st.session_state.view = "home"
            st.rerun()
    with col2:
        if st.button("保存本次会话记录"):
            if not st.session_state.get("conversation"):
                st.warning("当前没有可保存的会话记录。")
            else:
                saved_path = save_conversation_log()
                st.success(f"已保存到：{saved_path}")


def render_chat_header() -> None:
    last_saved = st.session_state.get("last_saved_hint", "未保存")
    title = html.escape(st.session_state.get("session_title", "当前对话"))
    st.markdown(
        f"""
        <div class="chat-topbar">
            <div>
                <div class="chat-title">{title}</div>
                <div class="chat-subtitle">最近保存：{last_saved}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_styles()

    if "execution_chain" not in st.session_state:
        st.session_state.execution_chain = {}
    if "result_text" not in st.session_state:
        st.session_state.result_text = ""
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "view" not in st.session_state:
        st.session_state.view = "chat" if st.session_state.conversation else "home"
    if "last_saved_hint" not in st.session_state:
        st.session_state.last_saved_hint = "未保存"
    if "session_id" not in st.session_state:
        st.session_state.session_id = generate_session_id()
    if "session_title" not in st.session_state:
        st.session_state.session_title = "新会话"
    if "session_created_at" not in st.session_state:
        st.session_state.session_created_at = datetime.now().isoformat(timespec="seconds")
    if "session_updated_at" not in st.session_state:
        st.session_state.session_updated_at = st.session_state.session_created_at
    if "current_session_file" not in st.session_state:
        st.session_state.current_session_file = None

    render_sidebar_session_manager()

    uploaded_file = None
    df = None
    upload_error = None

    def render_upload_area(compact: bool = False) -> pd.DataFrame | None:
        nonlocal uploaded_file, upload_error
        if compact:
            with st.expander("Excel 数据文件", expanded=False):
                uploaded_file = st.file_uploader(
                    "上传 Excel 数据文件",
                    type=["xlsx"],
                    help="需要包含“问题记录数据”sheet。问数类问题会基于该 sheet 做真实统计。",
                    key="uploaded_excel_chat",
                )
                st.caption("问数分析需要上传 Excel；普通分诊和基础问答可以直接提问。")
        else:
            st.markdown('<div class="upload-title">上传 Excel 数据文件</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "上传 Excel 数据文件",
                type=["xlsx"],
                help="需要包含“问题记录数据”sheet。问数类问题会基于该 sheet 做真实统计。",
                label_visibility="collapsed",
                key="uploaded_excel_home",
            )
            st.markdown('<div class="upload-note">问数分析需要上传 Excel；普通分诊和基础问答可以直接提问。</div>', unsafe_allow_html=True)

        loaded_df = None
        if uploaded_file is not None:
            loaded_df, upload_error = load_issue_sheet(uploaded_file)
            if loaded_df is not None:
                st.session_state.uploaded_df = loaded_df
                st.session_state.uploaded_df_session_id = st.session_state.session_id
        elif (
            "uploaded_df" in st.session_state
            and st.session_state.get("uploaded_df_session_id") == st.session_state.session_id
        ):
            loaded_df = st.session_state.uploaded_df
        render_compact_status(loaded_df, upload_error)
        return loaded_df

    def render_thinking_state(container: Any) -> None:
        with container.container():
            st.markdown(
                """
                <div class="chat-window">
                    <div class="message assistant">
                        <div>
                            <div class="meta-pill">正在处理</div>
                            <div class="bubble">正在理解问题并生成回答...</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def handle_submit(question: str, loaded_df: pd.DataFrame | None, thinking_slot: Any | None = None) -> None:
        st.session_state.execution_chain = {}
        st.session_state.result_text = ""

        if not question.strip():
            st.warning("请输入问题后再开始分析。")
            return
        if not OPENAI_API_KEY:
            st.error(API_KEY_MISSING_MESSAGE)
            return

        try:
            if thinking_slot is not None:
                render_thinking_state(thinking_slot)
                spinner_context = thinking_slot.container()
            else:
                spinner_context = st.container()

            with spinner_context:
                with st.spinner("正在理解问题并生成回答..."):
                    result, chain = process_agent_turn(question, loaded_df)
            if thinking_slot is not None:
                thinking_slot.empty()
            st.session_state.result_text = result
            st.session_state.execution_chain = chain
            add_conversation_turn(question, chain, result)
            st.session_state.view = "chat"
            st.rerun()
        except (RuntimeError, ValueError) as exc:
            if thinking_slot is not None:
                thinking_slot.empty()
            st.error(str(exc))

    if st.session_state.view == "chat" and st.session_state.conversation:
        render_chat_header()
        df = render_upload_area(compact=True)
        render_chat_messages()
        thinking_slot = st.empty()

        st.markdown('<div class="chat-input-wrap">', unsafe_allow_html=True)
        with st.form("chat_form", clear_on_submit=True):
            question = st.text_area(
                "继续输入问题",
                placeholder="在此输入任何您想查询或分析的问题",
                height=112,
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("发送")
        st.markdown('<div class="footer-note">内容由 AI 生成，仅供参考</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            handle_submit(question, df, thinking_slot)
        return

    render_minimal_header()

    st.markdown('<div class="question-block">', unsafe_allow_html=True)
    with st.form("agent_form", clear_on_submit=False):
        question = st.text_area(
            "请输入你的数据支持问题或问数问题",
            placeholder="请输入你的数据支持问题或问数问题",
            height=128,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("开始分析")
    st.markdown("</div>", unsafe_allow_html=True)
    thinking_slot = st.empty()

    render_capabilities()
    df = render_upload_area(compact=False)

    st.markdown('<div class="footer-note">内容由 AI 生成，仅供参考</div>', unsafe_allow_html=True)

    if submitted:
        handle_submit(question, df, thinking_slot)


if __name__ == "__main__":
    main()
