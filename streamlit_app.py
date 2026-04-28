import streamlit as st
from openai import OpenAI
import json
import os
import time
import glob
from datetime import datetime
from typing import List, Dict, Any, Generator


# 页面配置
st.set_page_config(
    page_title="AI 对话助手",
    page_icon="🤖",
    layout="wide"
)


# 初始化会话状态中的消息列表
def init_session_state():
    """初始化 session_state 中的必要变量"""
    if "messages" not in st.session_state:
        # 默认系统消息和欢迎消息
        st.session_state.messages = [
            {"role": "system",
             "content": "你是一个有用的AI助手，请用markdown格式回答问题，包括代码块时请使用正确的语言标识。"},
            {"role": "assistant",
             "content": "你好！我是AI助手，有什么可以帮你的吗？\n\n我可以帮你解答问题、编写代码等。你可以输入任何问题开始对话。"}
        ]
    if "sessions_dir" not in st.session_state:
        st.session_state.sessions_dir = "chat_sessions"
        # 确保会话保存目录存在
        if not os.path.exists(st.session_state.sessions_dir):
            os.makedirs(st.session_state.sessions_dir)


def save_session_to_file(messages: List[Dict[str, str]]) -> str:
    """保存当前对话会话到JSON文件，返回文件路径"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{timestamp}.json"
    filepath = os.path.join(st.session_state.sessions_dir, filename)
    session_data = {
        "timestamp": timestamp,
        "messages": messages,
        "message_count": len(messages)
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)
    return filepath


def load_session_from_file(filepath: str) -> List[Dict[str, str]]:
    """从JSON文件加载会话消息列表"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            session_data = json.load(f)
            messages = session_data.get("messages", [])
            if isinstance(messages, list) and messages:
                return messages
            else:
                st.error("会话文件格式无效，缺少有效的消息列表")
                return None
    except Exception as e:
        st.error(f"加载会话失败: {str(e)}")
        return None


def get_list_sessions() -> List[Dict[str, Any]]:
    """获取所有保存的会话信息，返回列表包含文件名和预览"""
    session_files = glob.glob(os.path.join(st.session_state.sessions_dir, "session_*.json"))
    sessions = []
    for filepath in session_files:
        filename = os.path.basename(filepath)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                timestamp = data.get("timestamp", filename.replace("session_", "").replace(".json", ""))
                msg_count = data.get("message_count", len(data.get("messages", [])))
                messages = data.get("messages", [])
                preview = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        preview = msg.get("content", "")[:50]
                        break
                if not preview and messages:
                    preview = messages[-1].get("content", "")[:50] if messages else ""
                sessions.append({
                    "filename": filename,
                    "filepath": filepath,
                    "timestamp": timestamp,
                    "msg_count": msg_count,
                    "preview": preview
                })
        except:
            continue
    sessions.sort(key=lambda x: x["timestamp"], reverse=True)
    return sessions


def delete_session_file(filepath: str):
    """删除会话文件"""
    try:
        os.remove(filepath)
        return True
    except Exception as e:
        st.error(f"删除失败: {str(e)}")
        return False


def reset_conversation(keep_system: bool = True):
    """重置当前对话"""
    if keep_system:
        system_msg = None
        for msg in st.session_state.messages:
            if msg.get("role") == "system":
                system_msg = msg
                break
        st.session_state.messages = [system_msg] if system_msg else [
            {"role": "system", "content": "你是一个有用的AI助手"}]
        st.session_state.messages.append({"role": "assistant", "content": "对话已重置，有什么我可以帮助你的吗？"})
    else:
        st.session_state.messages = [
            {"role": "system",
             "content": "你是一个有用的AI助手，请用markdown格式回答问题，包括代码块时请使用正确的语言标识。"},
            {"role": "assistant",
             "content": "你好！我是AI助手，有什么可以帮你的吗？\n\n我可以帮你解答问题、编写代码等。你可以输入任何问题开始对话。"}
        ]
    st.rerun()


# ---- 流式调用 AI API ----
def stream_ai_response(messages: List[Dict[str, str]], api_key: str, model: str, temperature: float) -> Generator[
    str, None, None]:
    """流式调用OpenAI API，逐块生成回复内容"""
    if not api_key:
        yield "⚠️ 请先在侧边栏配置您的 API Key。"
        return

    clean_messages = []
    for msg in messages:
        if msg.get("content") and msg.get("role") in ["system", "user", "assistant"]:
            clean_messages.append(msg)

    if not clean_messages:
        yield "没有有效的消息内容。"
        return

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model=model,
            messages=clean_messages,
            temperature=temperature,
            stream=True  # 开启流式
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        print(e)
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "auth" in error_msg.lower():
            yield "❌ API Key 无效或未正确配置，请检查您的 API Key。"
        else:
            yield f"❌ 调用AI接口时出错: {error_msg}"


# 侧边栏配置
def sidebar_config():
    with st.sidebar:
        st.title("⚙️ 配置")
        st.subheader("🤖 API 设置")
        api_key = st.text_input("API Key", type="password", help="请输入您的 API Key")
        model = st.selectbox(
            "模型选择",
            options=["deepseek-chat", "deepseek-v4-pro"],
            # options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo-16k"],
            index=0
        )
        temperature = st.slider("温度 (Temperature)", 0.0, 1.0, 0.7, 0.05)

        st.divider()

        st.subheader("💬 系统提示")
        system_prompt = st.text_area(
            "系统提示词（修改后将重置对话）",
            value=st.session_state.messages[0]["content"] if st.session_state.messages and st.session_state.messages[0][
                "role"] == "system" else "",
            height=100
        )
        if system_prompt != (
                st.session_state.messages[0]["content"] if st.session_state.messages and st.session_state.messages[0][
                    "role"] == "system" else ""):
            if system_prompt.strip():
                if st.session_state.messages and st.session_state.messages[0]["role"] == "system":
                    st.session_state.messages[0]["content"] = system_prompt
                else:
                    st.session_state.messages.insert(0, {"role": "system", "content": system_prompt})
                st.session_state.messages = [st.session_state.messages[0]]
                st.session_state.messages.append(
                    {"role": "assistant", "content": "系统提示已更新，你好！有什么我可以帮你的吗？"})
                st.rerun()

        st.divider()

        st.subheader("📁 会话管理")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 新建会话", use_container_width=True):
                reset_conversation(keep_system=True)
        with col2:
            if st.button("🗑️ 清空对话", use_container_width=True):
                system_msg = None
                for msg in st.session_state.messages:
                    if msg.get("role") == "system":
                        system_msg = msg
                        break
                if system_msg:
                    st.session_state.messages = [system_msg]
                else:
                    st.session_state.messages = []
                st.session_state.messages.append({"role": "assistant", "content": "对话已清空，有什么我可以帮你的吗？"})
                st.rerun()

        if st.button("💾 保存当前会话", use_container_width=True):
            if len(st.session_state.messages) > 1:
                filepath = save_session_to_file(st.session_state.messages)
                st.success(f"会话已保存至: {os.path.basename(filepath)}")
                time.sleep(1)
                st.rerun()
            else:
                st.warning("没有对话内容可保存")

        st.divider()

        st.subheader("📜 历史会话")
        sessions = get_list_sessions()
        if not sessions:
            st.info("暂无保存的历史会话")
        else:
            for session in sessions:
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.caption(f"📄 {session['timestamp']}")
                        st.text(f"消息数: {session['msg_count']}")
                        if session['preview']:
                            st.caption(f"预览: {session['preview']}...")
                    with col2:
                        if st.button("加载", key=f"load_{session['filename']}"):
                            messages = load_session_from_file(session['filepath'])
                            if messages:
                                st.session_state.messages = messages
                                st.success(f"已加载会话: {session['timestamp']}")
                                st.rerun()
                    with col3:
                        if st.button("❌", key=f"del_{session['filename']}"):
                            if delete_session_file(session['filepath']):
                                st.rerun()
                    st.divider()

        return api_key, model, temperature


# 主聊天界面（流式渲染）
def main_chat(api_key, model, temperature):
    st.title("🤖 AI 对话助手")
    st.caption("支持 Markdown 和代码高亮 | 多轮对话 | 历史记录保存/加载 | **流式输出**")

    # 显示历史消息
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 聊天输入框
    if prompt := st.chat_input("请输入您的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 准备调用流式API
        with st.chat_message("assistant"):
            # 创建一个空的占位符
            message_placeholder = st.empty()
            full_response = ""

            # 准备完整的消息列表（包含系统消息）
            full_messages = st.session_state.messages.copy()

            # 流式获取回复并逐步更新
            for chunk in stream_ai_response(full_messages, api_key, model, temperature):
                if chunk:
                    full_response += chunk
                    # 实时更新显示，注意使用markdown渲染
                    message_placeholder.markdown(full_response + "▌")
            # 最终显示完整回复（去掉光标符号）
            message_placeholder.markdown(full_response)

        # 将完整的助手回复存入会话历史
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()


# def main():
init_session_state()
api_key, model, temperature = sidebar_config()
main_chat(api_key, model, temperature)


if __name__ == "__main__":
    #     main()
    gen = (x ** 2 for x in range(10))  # 圆括号，不是方括号
    for val in gen:
        print(val)
