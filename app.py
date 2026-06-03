import os
import traceback

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.0-flash"

RESUME_PROMPT_TEMPLATE = """你是一位資深人力資源顧問與履歷撰寫專家。請根據以下使用者提供的資訊，撰寫一份專業、具說服力的履歷內容。

## 使用者資訊
- **目標職位**：{target_role}
- **核心技能**：{core_skills}
- **經歷簡述**：{experience_summary}

## 撰寫要求（務必嚴格遵守）
1. **語言**：全文使用「繁體中文」。
2. **STAR 原則**：擴寫工作經歷時，每一項成就須盡量涵蓋 Situation（情境）、Task（任務）、Action（行動）、Result（成果），以具體數據或成果收尾。
3. **結構**：輸出須包含以下區塊，並使用 Markdown 標題（##）區分：
   - ## 專業總結
   - ## 核心技能
   - ## 工作經歷（依 STAR 原則擴寫，條列清楚）
4. **風格**：專業、精煉、針對目標職位客製化，避免空泛形容詞。
5. **格式與輸出限制**：
   - 請直接輸出履歷內容，**絕對不要**包含任何問候語（如「好的」、「這是一份履歷」等）或結語。
   - 請確保輸出為純淨的 Markdown 格式。
   - 僅輸出履歷 Markdown 正文，禁止前言、後記、「以下是履歷」等說明文字。

從第一個 Markdown 標題（##）開始輸出，不要有任何其他文字。"""


def _get_gemini_model():
    """Configure and return the Gemini model, or raise with a clear message."""
    if not GEMINI_API_KEY or GEMINI_API_KEY.strip() in ("", "your_api_key_here"):
        raise ValueError(
            "未設定有效的 GEMINI_API_KEY。請在 .env 檔案中設定您的 Google Gemini API 金鑰。"
        )

    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY.strip())
    return genai.GenerativeModel(MODEL_NAME)


def _build_prompt(target_role: str, core_skills: str, experience_summary: str) -> str:
    return RESUME_PROMPT_TEMPLATE.format(
        target_role=target_role.strip() or "（未提供）",
        core_skills=core_skills.strip() or "（未提供）",
        experience_summary=experience_summary.strip() or "（未提供）",
    )


def _extract_response_text(response) -> str:
    """Extract text from Gemini response; raise ValueError if empty or blocked."""
    try:
        text = response.text
        if text and text.strip():
            return text.strip()
    except (ValueError, AttributeError):
        pass

    if getattr(response, "candidates", None):
        parts = []
        for candidate in response.candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(part_text)
        if parts:
            return "\n".join(parts).strip()

    block_reason = None
    if getattr(response, "prompt_feedback", None):
        block_reason = getattr(response.prompt_feedback, "block_reason", None)
    if block_reason:
        raise ValueError("輸入內容可能觸發安全政策，請調整描述後再試。")

    raise ValueError("AI 未回傳有效內容，請稍後再試。")


def _gemini_error_message(exc: Exception) -> str:
    """Map exceptions to user-facing JSON error messages."""
    message = str(exc).lower()
    if "api key" in message or "api_key" in message or "permission" in message:
        return "API 金鑰無效或權限不足，請確認 GEMINI_API_KEY 是否正確。"
    if "quota" in message or "rate" in message or "429" in message:
        return "API 配額已用盡或請求過於頻繁，請稍後再試。"
    if "timeout" in message or "deadline" in message:
        return "請求逾時，請檢查網路連線後再試。"
    if "404" in message or "not found" in message:
        return f"模型 {MODEL_NAME} 無法使用，請確認 API 是否支援此模型。"
    return f"生成履歷時發生錯誤：{exc}"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate_resume():
    try:
        data = request.get_json(silent=True) or {}
        target_role = (data.get("target_role") or "").strip()
        core_skills = (data.get("core_skills") or "").strip()
        experience_summary = (data.get("experience_summary") or "").strip()

        if not target_role:
            return jsonify({"success": False, "error": "請填寫目標職位。"}), 400
        if not core_skills:
            return jsonify({"success": False, "error": "請填寫核心技能。"}), 400
        if not experience_summary:
            return jsonify({"success": False, "error": "請填寫經歷簡述。"}), 400

        try:
            model = _get_gemini_model()
            prompt = _build_prompt(target_role, core_skills, experience_summary)
            response = model.generate_content(prompt)
        except ValueError as exc:
            return jsonify({"success": False, "error": str(exc)}), 500
        except Exception as exc:
            app.logger.error("Gemini API call failed: %s\n%s", exc, traceback.format_exc())
            return jsonify(
                {"success": False, "error": _gemini_error_message(exc)}
            ), 502

        try:
            content = _extract_response_text(response)
        except ValueError as exc:
            return jsonify({"success": False, "error": str(exc)}), 502

        return jsonify({"success": True, "content": content})

    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 500
    except Exception as exc:
        app.logger.error("Unexpected error: %s\n%s", exc, traceback.format_exc())
        return jsonify(
            {"success": False, "error": _gemini_error_message(exc)}
        ), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Render."""
    key_ok = bool(
        GEMINI_API_KEY
        and GEMINI_API_KEY.strip() not in ("", "your_api_key_here")
    )
    return jsonify({"status": "ok", "api_key_configured": key_ok})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
