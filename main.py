import gradio as gr
import dashscope
from dashscope import MultiModalConversation
from PIL import Image
import base64
import io
import re
import sqlite3
from sqlite3 import Error
import os
from datetime import datetime
import uuid
import markdown
from bs4 import BeautifulSoup

# ====== 配置 ======
dashscope.api_key = "sk-0482b028c90a46b99047ec5f5206df55"
MAX_IMAGE_SIZE = 2048
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ====== 数据库操作类 ======
class ChartQADatabase:
    def __init__(self, db_file="chart_qa.db"):
        self.db_file = db_file
        self.conn = None
        self._initialize_db()

    def _initialize_db(self):
        try:
            self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    session_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    image_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    image_path TEXT NOT NULL,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            ''')
            self.conn.commit()
        except Error as e:
            print(f"数据库初始化错误: {e}")
            self.conn.rollback()

    def get_or_create_user(self, username):
        cursor = self.conn.cursor()
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if row:
            return row[0]
        user_id = str(uuid.uuid4())
        self.conn.execute("INSERT INTO users (user_id, username) VALUES (?, ?)", (user_id, username))
        self.conn.commit()
        return user_id

    def create_session(self, user_id, session_name="新会话"):
        session_id = str(uuid.uuid4())
        self.conn.execute("INSERT INTO sessions (session_id, user_id, session_name) VALUES (?, ?, ?)",
                          (session_id, user_id, session_name))
        self.conn.commit()
        return session_id

    def get_user_sessions(self, user_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT session_id, session_name FROM sessions WHERE user_id = ? ORDER BY created_at DESC",
                       (user_id,))
        return cursor.fetchall()

    def add_conversation(self, session_id, question, answer):
        conversation_id = str(uuid.uuid4())
        self.conn.execute(
            "INSERT INTO conversations (conversation_id, session_id, question, answer) VALUES (?, ?, ?, ?)",
            (conversation_id, session_id, question, answer))
        self.conn.commit()

    def add_image(self, session_id, image_path, description=None):
        image_id = str(uuid.uuid4())
        self.conn.execute(
            "INSERT INTO images (image_id, session_id, image_path, description) VALUES (?, ?, ?, ?)",
            (image_id, session_id, image_path, description))
        self.conn.commit()

    def get_session_history(self, session_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT question, answer FROM conversations WHERE session_id = ? ORDER BY timestamp ASC",
                       (session_id,))
        return cursor.fetchall()

    def close(self):
        self.conn.close()


# ====== 初始化数据库 ======
db = ChartQADatabase()
DEFAULT_USER_ID = db.get_or_create_user("default_user")
CURRENT_SESSION_ID = db.create_session(DEFAULT_USER_ID, "默认会话")


# ====== 工具函数 ======
def pil_image_to_base64_str(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"


def save_uploaded_image(image, session_id):
    image_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{image_id}.png")
    image.save(path)
    db.add_image(session_id, path)
    return path


def clean_markdown(text):
    # 将 Markdown 转换为 HTML
    html = markdown.markdown(text, output_format='html')
    # 使用 BeautifulSoup 清理 HTML 标签，只保留文本
    soup = BeautifulSoup(html, features="html.parser")
    plain_text = soup.get_text(separator="\n")  # 保留段落结构
    # 去除多余空行、空格
    plain_text = re.sub(r'\n{3,}', '\n\n', plain_text)
    plain_text = re.sub(r'[ \t]{2,}', ' ', plain_text)
    return plain_text.strip()


def convert_history_to_messages(history):
    """将数据库格式 (question, answer) 转换为 Chatbot 支持的消息格式"""
    messages = []
    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    return messages


def answer_with_image(image, question, session_id, history_state):
    if not image:
        return "请上传图像。", history_state
    if not question.strip():
        return "请输入问题。", history_state

    try:
        save_uploaded_image(image, session_id)
        img_b64 = pil_image_to_base64_str(image)
        response = MultiModalConversation.call(
            model="qwen-vl-plus",
            messages=[
                {"role": "user", "content": [{"image": img_b64}, {"text": question}]}
            ]
        )

        output = response.get("output")
        if not output or "choices" not in output:
            raise ValueError(f"output 字段缺失或为空，返回值为：{response}")

        choices = output.get("choices")
        if not choices or not isinstance(choices, list):
            raise ValueError(f"choices 字段结构不合法，返回值为：{response}")

        message = choices[0].get("message")
        if not message or "content" not in message:
            raise ValueError(f"message 内容缺失，返回值为：{response}")

        content = message["content"]
        if not content:
            raise ValueError("content 字段为空")

        if isinstance(content, list):
            answer_text = ''.join([item.get("text", "") for item in content])
        elif isinstance(content, str):
            answer_text = content
        else:
            raise ValueError("content 格式未知，预期为 list 或 str")

        cleaned = clean_markdown(answer_text)
        db.add_conversation(session_id, question, cleaned)
        history_state = history_state or []
        history_state.append((question, cleaned))
        return cleaned, history_state

    except Exception as e:
        return f"❌ 错误: 模型输出结构异常：{str(e)}", history_state


def get_session_history(session_id):
    history = db.get_session_history(session_id)
    return convert_history_to_messages(history)


def create_new_session(session_name):
    global CURRENT_SESSION_ID
    CURRENT_SESSION_ID = db.create_session(DEFAULT_USER_ID, session_name)
    sessions = db.get_user_sessions(DEFAULT_USER_ID)
    session_choices = [(name, sid) for sid, name in sessions]  # ✅ 纠正顺序
    dropdown_update = gr.Dropdown.update(choices=session_choices, value=CURRENT_SESSION_ID)
    history = db.get_session_history(CURRENT_SESSION_ID)
    history_as_messages = convert_history_to_messages(history)
    return dropdown_update, history_as_messages


def refresh_dropdown():
    return gr.Dropdown.update(
        choices=[(name, sid) for sid, name in db.get_user_sessions(DEFAULT_USER_ID)],
        value=CURRENT_SESSION_ID
    )


# ====== Gradio UI ======
with gr.Blocks(title="图表问答系统") as demo:
    current_session = gr.State(CURRENT_SESSION_ID)
    chat_history = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 会话管理")
            session_dropdown = gr.Dropdown(
                choices=[(name, sid) for sid, name in db.get_user_sessions(DEFAULT_USER_ID)],
                value=CURRENT_SESSION_ID,
                label="选择会话"
            )
            new_session_name = gr.Textbox(label="新建会话名", value="新会话")
            create_session_btn = gr.Button("创建新会话")
            history_display = gr.Chatbot(label="历史问答", height=300, type="messages")

        with gr.Column(scale=2):
            gr.Markdown("### 上传图像并提问")
            image_input = gr.Image(type="pil", label="上传图表图像")
            question_input = gr.Textbox(lines=2, label="请输入你的问题")
            submit_btn = gr.Button("提交")
            answer_output = gr.Textbox(label="模型回答", interactive=False)
            current_chat = gr.Chatbot(label="当前会话对话", height=300, type="messages")

    submit_btn.click(
        answer_with_image,
        inputs=[image_input, question_input, current_session, chat_history],
        outputs=[answer_output, chat_history]
    ).then(
        lambda h: convert_history_to_messages(h),
        inputs=[chat_history],
        outputs=[current_chat]
    )

    session_dropdown.change(
        lambda sid: (sid, get_session_history(sid)),
        inputs=[session_dropdown],
        outputs=[current_session, history_display]
    )

    create_session_btn.click(
        create_new_session,
        inputs=[new_session_name],
        outputs=[session_dropdown, history_display]
    )

# ====== 启动服务 ======
if __name__ == "__main__":
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            auth=("zjr", "123")
        )
    finally:
        db.close()
