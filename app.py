import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import chess
import chess.engine
import chess.svg
import requests
import streamlit as st
from streamlit import components

ENGINE_PATH = Path(__file__).parent / "sunfish.py"
DEFAULT_MODEL = "deepseek-chat"


@dataclass
class DeepSeekConfig:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = "https://api.deepseek.com"


class DeepSeekExplainer:
    def __init__(self, config: Optional[DeepSeekConfig]):
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config is not None and bool(self.config.api_key.strip())

    def explain(self, board: chess.Board, latest_event: str) -> str:
        """实时讲解接口。未配置 key 时返回本地占位内容。"""
        if not self.enabled:
            return "未配置 DeepSeek API Key，当前为本地占位讲解。可在侧边栏填写 Key 开启实时策略分析。"

        system_prompt = (
            "你是一名职业国际象棋教练。请根据局面给出简明、可执行的建议，"
            "并说明短期计划（1-3步）与中长期思路。"
        )
        user_prompt = (
            f"当前 FEN: {board.fen()}\n"
            f"最新事件: {latest_event}\n"
            f"轮到 {'白方' if board.turn == chess.WHITE else '黑方'} 行棋。"
            "请给出：\n"
            "1) 局面评估（优劣势）\n"
            "2) 关键战术或子力关系\n"
            "3) 推荐计划与候选走法"
        )

        try:
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.5,
                },
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            return payload["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            return f"DeepSeek 调用失败（已回退本地提示）：{exc}"


def ensure_session_state() -> None:
    if "board" not in st.session_state:
        st.session_state.board = chess.Board()
    if "events" not in st.session_state:
        st.session_state.events = ["新对局开始"]
    if "explanations" not in st.session_state:
        st.session_state.explanations = []
    if "selected_square" not in st.session_state:
        st.session_state.selected_square = None


def apply_move(move: chess.Move, board: chess.Board) -> str:
    if move not in board.legal_moves:
        raise ValueError("非法走法，请检查输入。")
    san = board.san(move)
    board.push(move)
    return san


def parse_and_push_move(raw_move: str, board: chess.Board) -> str:
    move_text = raw_move.strip()
    if not move_text:
        raise ValueError("请输入走法（支持 SAN 或 UCI，如 e4 / e2e4 / Nf3）。")

    try:
        move = board.parse_san(move_text)
    except ValueError:
        move = chess.Move.from_uci(move_text)
        if move not in board.legal_moves:
            raise ValueError("非法走法，请检查输入。")

    return apply_move(move, board)


def request_engine_move(board: chess.Board, think_time: float) -> str:
    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"未找到引擎文件: {ENGINE_PATH}")

    cmd = ["python", str(ENGINE_PATH)]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with chess.engine.SimpleEngine.popen_uci(cmd, creationflags=creationflags) as engine:
        result = engine.play(board, chess.engine.Limit(time=think_time))

    if result.move is None:
        raise RuntimeError("引擎未返回可用走法。")

    san = board.san(result.move)
    board.push(result.move)
    return san


def _clear_square_query_param() -> None:
    if "sq" in st.query_params:
        del st.query_params["sq"]


def handle_board_square_click(board: chess.Board) -> Tuple[Optional[str], Optional[str]]:
    """处理直接点击棋盘格事件，返回 (moved_san, message)。"""
    square_name = st.query_params.get("sq")
    if not square_name:
        return None, None

    try:
        square = chess.parse_square(square_name)
    except ValueError:
        _clear_square_query_param()
        return None, "无效棋盘坐标，已忽略该点击。"

    _clear_square_query_param()

    selected = st.session_state.selected_square
    if selected is None:
        piece = board.piece_at(square)
        if piece is None or piece.color != board.turn:
            return None, "请先点击当前行棋方的棋子作为起点。"
        st.session_state.selected_square = square
        return None, f"已选择起点：{chess.square_name(square)}"

    if square == selected:
        st.session_state.selected_square = None
        return None, "已取消当前起点选择。"

    move = chess.Move(selected, square)
    selected_piece = board.piece_at(selected)
    if selected_piece and selected_piece.piece_type == chess.PAWN and chess.square_rank(square) in (0, 7):
        move = chess.Move(selected, square, promotion=chess.QUEEN)

    st.session_state.selected_square = None
    if move not in board.legal_moves:
        return None, "该目标格不是合法走法，请重新点击起点和终点。"
    return apply_move(move, board), None


def render_interactive_board(board: chess.Board) -> None:
    links = {
        square: f"?sq={chess.square_name(square)}"
        for square in chess.SQUARES
    }
    selected = st.session_state.selected_square
    fill = {selected: "#f6f669"} if selected is not None else {}
    svg = chess.svg.board(board=board, size=560, fill=fill, links=links)

    st.markdown(
        """
        <style>
        .board-tip { margin: 0 0 0.4rem 0; color: #666; }
        .board-wrapper svg { max-width: 100%; height: auto; border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="board-tip">直接点击棋盘走子：起点 → 终点（支持鼠标/触屏）</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="board-wrapper">{svg}</div>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Sunfish Streamlit Chess", layout="wide")
    ensure_session_state()

    st.title("♟️ Sunfish + Streamlit 对弈与讲解")
    st.caption("基于现有 Sunfish 逻辑构建 UI，并预留 DeepSeek AI 实时讲解接口。")

    with st.sidebar:
        st.header("设置")
        think_time = st.slider("引擎思考时间（秒）", min_value=0.1, max_value=3.0, value=0.5, step=0.1)
        api_key = st.text_input("DeepSeek API Key（可选）", value=os.getenv("DEEPSEEK_API_KEY", ""), type="password")
        model_name = st.text_input("DeepSeek 模型", value=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL))
        if st.button("重置棋局", use_container_width=True):
            st.session_state.board = chess.Board()
            st.session_state.events = ["新对局开始"]
            st.session_state.explanations = []
            st.session_state.selected_square = None
            _clear_square_query_param()
            st.rerun()

    board: chess.Board = st.session_state.board
    explainer = DeepSeekExplainer(
        DeepSeekConfig(api_key=api_key, model=model_name) if api_key.strip() else None
    )

    left, right = st.columns([1.1, 1])

    with left:
        clicked_san, click_message = handle_board_square_click(board)
        if click_message:
            st.info(click_message)
        if clicked_san:
            event = f"玩家走子（棋盘点选）：{clicked_san}"
            st.session_state.events.append(event)
            st.session_state.explanations.append(explainer.explain(board, event))

        render_interactive_board(board)
        st.code(board.fen(), language="text")

        user_move = st.text_input("你的走法（备选）", placeholder="例如：e4 / Nf3 / e2e4")
        if st.button("提交走法", use_container_width=True):
            try:
                san = parse_and_push_move(user_move, board)
                event = f"玩家走子（文本）：{san}"
                st.session_state.events.append(event)
                st.session_state.explanations.append(explainer.explain(board, event))
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        if st.button("AI 走一步", use_container_width=True):
            try:
                san = request_engine_move(board, think_time=think_time)
                event = f"Sunfish 走子：{san}"
                st.session_state.events.append(event)
                st.session_state.explanations.append(explainer.explain(board, event))
                st.rerun()
            except Exception as exc:
                st.error(f"引擎错误：{exc}")

    with right:
        st.subheader("对局事件")
        for item in reversed(st.session_state.events[-12:]):
            st.write(f"- {item}")

        st.subheader("AI 讲解")
        if st.session_state.explanations:
            st.info(st.session_state.explanations[-1])
        else:
            st.write("暂未生成讲解。落子后可自动生成局面点评。")

        st.subheader("接口预留说明")
        st.markdown(
            "- 已封装 `DeepSeekExplainer`，可替换为任意兼容 OpenAI Chat Completions 的服务。\n"
            "- 当前请求携带局面 FEN + 最新动作，便于实时讲解与策略建议。\n"
            "- 若需要流式输出，可将 `requests.post` 改为 SSE/流式客户端。"
        )


if __name__ == "__main__":
    main()
