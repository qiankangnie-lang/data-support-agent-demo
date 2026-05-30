# 数据支持智能分诊与问数助手

这是一个运行时接入 OpenAI API 的 Streamlit AI Agent Demo。用户通过一个统一输入入口提交数据支持问题或历史问题问数问题，系统调用大模型完成意图识别，再按意图路由到对应处理节点。

## 项目结构

```text
app.py
requirements.txt
README.md
.env.example
prompts/
  intent_classification_prompt.txt
  permission_prompt.txt
  new_requirement_prompt.txt
  anomaly_prompt.txt
  caliber_prompt.txt
  consultation_prompt.txt
  data_analysis_prompt.txt
```

## 环境配置

复制 `.env.example` 为 `.env`，并填写真实 API Key：

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=deepseek-v4-pro
OPENAI_BASE_URL=https://api.deepseek.com/v1
```

如果未配置 `OPENAI_API_KEY`，页面会提示无法调用运行时大模型，并且不会回退到本地规则版。

## 安装与运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

默认访问地址通常是：

```text
http://localhost:8501
```

## 使用方式

1. 上传 `.xlsx` 文件。
2. 系统优先读取 sheet：`问题记录数据`。
3. 在统一输入框中输入数据支持问题或问数问题。
4. 点击“开始分析”。
5. 查看结果展示区与执行链路展示区。

## Agent 闭环链路

```text
用户输入
↓
读取 intent_classification_prompt.txt
↓
调用 OpenAI API 做意图识别
↓
根据 question_type 路由
↓
分诊类：读取对应 Prompt → 调用 OpenAI API → 输出结构化处理结果
↓
问数类：读取问数 Prompt → LLM 识别分析意图 → Pandas 真实统计 → LLM 生成业务解释和优化建议
```

## 内置测试集

页面提供“内置测试集验证区”，点击按钮后会对 6 条样例调用 OpenAI API 做真实意图识别，并展示期望分类、实际分类和是否通过。
