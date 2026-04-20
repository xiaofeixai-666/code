# -*- coding: utf-8 -*-
"""
Competition agent implementation.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from agent_base import (
    ACTION_CLICK,
    ACTION_COMPLETE,
    ACTION_OPEN,
    ACTION_SCROLL,
    ACTION_TYPE,
    AgentInput,
    AgentOutput,
    BaseAgent,
    UsageInfo,
    VALID_ACTIONS,
)
from utils.image_utils import create_coordinate_grid_overlay, encode_image_url

logger = logging.getLogger(__name__)


def _configure_utf8_runtime() -> None:
    """
    Force UTF-8 text IO so file reads, logs and terminal output do not rely on
    the host default encoding.
    """
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                logger.debug("Failed to reconfigure %s to UTF-8", stream_name)


class Agent(BaseAgent):
    """A screenshot-grounded Android GUI agent."""

    _APP_NAME_ALIASES = {
        "美团外卖": "美团",
        "美团app": "美团",
        "京东商城": "京东",
        "京东app": "京东",
        "jd": "京东",
        "jingdong": "京东",
        "拼多多app": "拼多多",
        "拼夕夕": "拼多多",
        "pinduoduo": "拼多多",
        "douyin": "抖音",
        "taobao": "淘宝",
        "dazhongdianping": "大众点评",
        "去哪旅行": "去哪儿旅行",
        "去哪儿": "去哪儿旅行",
        "芒果tv": "芒果TV",
        "芒果Tv": "芒果TV",
        "芒果tv视频": "芒果TV",
        "bilibili": "哔哩哔哩",
        "哔哩": "哔哩哔哩",
        "大众点评app": "大众点评",
        "淘宝app": "淘宝",
    }
    _SEARCH_CONFIRM_APPS = {
        "抖音",
    }

    _ACTION_EXAMPLES = """
Examples:
{"action":"CLICK","parameters":{"point":[875,72]}}
{"action":"TYPE","parameters":{"text":"采莲曲"}}
{"action":"SCROLL","parameters":{"start_point":[500,820],"end_point":[500,260]}}
{"action":"OPEN","parameters":{"app_name":"百度地图"}}
{"action":"COMPLETE","parameters":{}}
""".strip()

    def _initialize(self) -> None:
        _configure_utf8_runtime()
        self._max_history_actions = 10
        self._history_window_for_prompt = 8

    def reset(self) -> None:
        self._last_raw_output = ""

    def generate_messages(self, input_data: AgentInput) -> List[Dict[str, Any]]:
        image = input_data.current_image.convert("RGB")
        grid_image = create_coordinate_grid_overlay(image)

        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt(),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._build_user_prompt(input_data)},
                    {"type": "image_url", "image_url": {"url": encode_image_url(image)}},
                    {"type": "image_url", "image_url": {"url": encode_image_url(grid_image)}},
                ],
            },
        ]
        return messages

    def act(self, input_data: AgentInput) -> AgentOutput:
        messages = self.generate_messages(input_data)
        response = self._call_api_with_retry(messages)
        usage = self.extract_usage_info(response)
        raw_output = self._extract_response_text(response)
        self._last_raw_output = raw_output
        logger.info("Model raw output: %s", raw_output)

        action, parameters = self._parse_action(raw_output, input_data)
        repaired = self._maybe_repair_decision(
            messages=messages,
            input_data=input_data,
            action=action,
            parameters=parameters,
            raw_output=raw_output,
        )
        if repaired is not None:
            repaired_response, raw_output, action, parameters = repaired
            usage = self._merge_usage(usage, self.extract_usage_info(repaired_response))

        action, parameters = self._postprocess_decision(action, parameters, input_data)
        return AgentOutput(
            action=action,
            parameters=parameters,
            raw_output=raw_output,
            usage=usage,
        )

    def _build_system_prompt(self) -> str:
        return f"""
You are a precise Android GUI agent that decides exactly one next action.

Core requirements:
- Base every decision only on the user instruction, the current screenshot, and the concise action history.
- Never hardcode answers, memorize sample paths, or rely on hidden labels outside the screenshot.
- You will receive two images: the original screenshot and the same screenshot with an x/y normalized coordinate grid.
- Normalized coordinates are always integers in [0, 1000].
- For CLICK, choose a point safely inside the target element, not on the border.
- For SCROLL, return one clear swipe. Use a realistic distance and keep the start/end points inside the screen.
- Use OPEN only when the target app is not open yet.
- Use TYPE only when text should be entered now.
- Use COMPLETE only when the user goal is already achieved or the requested result is clearly visible.
- Prefer explicit UI controls over semantically related content cards. For example, when the task requires searching, first use the visible search entrance instead of tapping a recommended content card.
- In shopping, review, or form flows, prioritize explicit buttons and fields such as search bars, review/评价 buttons, submit buttons, spec selectors, pay buttons, and confirmation controls.
- If the task is to review, comment, rate, submit feedback, or publish content, first locate the explicit entry to that function instead of clicking unrelated content.
- If the task is to buy, pay, recharge, add to cart, or choose specifications, prefer the bottom action bar or right-side call-to-action buttons over content tiles.
- If there is a visible skip/close/跳过/关闭 control on a splash ad, prefer that before any other action.
- If the previous valid action already typed text, the next action is usually a confirm click, result click, or submit click. Avoid immediate SCROLL, repeated TYPE, or premature COMPLETE.
- Output JSON only. No markdown fence, no extra prose, no explanation outside JSON.

Allowed actions and schemas:
- CLICK -> {{"point":[x, y]}}
- TYPE -> {{"text":"..."}}
- SCROLL -> {{"start_point":[x1, y1], "end_point":[x2, y2]}}
- OPEN -> {{"app_name":"..."}}
- COMPLETE -> {{}}

Use this exact JSON shape:
{{"action":"CLICK|TYPE|SCROLL|OPEN|COMPLETE","parameters":{{...}}}}

{self._ACTION_EXAMPLES}
""".strip()

    def _build_user_prompt(self, input_data: AgentInput) -> str:
        width, height = input_data.current_image.size
        history_lines = self._format_history(input_data.history_actions)
        task_hints = self._build_task_hints(input_data.instruction, input_data.history_actions)
        stage_hints = self._build_stage_hints(input_data)

        return f"""
用户指令: {input_data.instruction}
当前步数: {input_data.step_count}
截图尺寸: {width}x{height}

历史动作摘要:
{history_lines}

决策提醒:
1. 第一张图是原始截图，第二张图是同一截图的归一化坐标网格版。
2. 坐标换算规则是分别对宽和高归一化到 1000；右下角大约是 [1000,1000]，中心大约是 [500,500]。
3. 如果需要搜索，通常先点顶部搜索入口/搜索框，再 TYPE 关键词，再点搜索按钮或候选结果。
4. 如果用户要“播放/查看/打开评论区/下单/打车”，完成态应该是目标页面已进入或目标动作已达成，而不是仅搜索到结果。
5. 如果用户要“看一下多少钱/查看信息”，只有当答案已经在当前界面可见时才能 COMPLETE。
6. 如果任务涉及“评价/好评/评论/晒单/反馈”，优先寻找“评价”“写评价”“去评价”“评论”“发布”“提交”等显式入口。
7. 如果任务涉及“购买/下单/支付/加购/选规格/充值”，优先寻找底部操作栏、右侧按钮、规格弹窗和“去支付/提交订单/立即购买”等显式控件。

当前任务专项提醒:
{task_hints}

当前阶段提醒:
{stage_hints}

请只返回一个 JSON 对象。
""".strip()

    def _format_history(self, history_actions: Sequence[Dict[str, Any]]) -> str:
        if not history_actions:
            return "- 无历史动作。若当前是桌面或应用列表，优先考虑 OPEN。"

        lines: List[str] = []
        for item in history_actions[-self._history_window_for_prompt:]:
            action = item.get("action", "")
            parameters = item.get("parameters", {})
            status = "valid" if item.get("is_valid", True) else "invalid"
            lines.append(
                f"- step {item.get('step', '?')}: {action} {json.dumps(parameters, ensure_ascii=False)} [{status}]"
            )

        return "\n".join(lines)

    def _extract_response_text(self, response: Any) -> str:
        content = response.choices[0].message.content

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue

                text = None
                if isinstance(item, dict):
                    text = item.get("text")
                else:
                    text = getattr(item, "text", None)

                if text:
                    parts.append(text)

            return "\n".join(parts).strip()

        return str(content).strip()

    def _parse_action(self, raw_output: str, input_data: AgentInput) -> Tuple[str, Dict[str, Any]]:
        payload = self._parse_json_payload(raw_output)
        if not self._looks_like_action_payload(payload):
            payload = None
        if payload is None:
            payload = self._parse_fallback_payload(raw_output)

        if payload is None:
            raise ValueError(f"Unable to parse model output into an action: {raw_output}")

        action = str(
            payload.get("action")
            or payload.get("Action")
            or payload.get("action_type")
            or payload.get("type")
            or ""
        ).strip().upper()

        if action not in VALID_ACTIONS:
            raise ValueError(f"Unsupported action: {action!r}; raw output={raw_output}")

        raw_parameters = payload.get("parameters")
        if raw_parameters is None:
            raw_parameters = payload.get("params", {})

        parameters = self._sanitize_parameters(
            action=action,
            parameters=raw_parameters,
            image_size=input_data.current_image.size,
            raw_output=raw_output,
            instruction=input_data.instruction,
        )
        return action, parameters

    def _parse_json_payload(self, raw_output: str) -> Optional[Dict[str, Any]]:
        cleaned = raw_output.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        best_payload: Optional[Dict[str, Any]] = None
        best_score = -1

        for candidate in self._iter_braced_candidates(cleaned):
            for normalized in self._iter_normalized_candidate_variants(candidate):
                parsed = self._loads_json_like(normalized)
                if not isinstance(parsed, dict):
                    continue

                score = self._score_payload_candidate(parsed)
                if score > best_score:
                    best_payload = parsed
                    best_score = score

        return best_payload if best_score > 0 else None

    def _iter_braced_candidates(self, text: str) -> Iterable[str]:
        text = text.strip()
        if text:
            yield text

        starts = [index for index, ch in enumerate(text) if ch == "{"]
        for start in starts:
            depth = 0
            for end in range(start, len(text)):
                char = text[end]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        yield text[start:end + 1]
                        break

    def _iter_normalized_candidate_variants(self, text: str) -> Iterable[str]:
        yielded = set()

        for candidate in (
            text,
            self._normalize_json_like_text(text),
            self._replace_smart_quotes_if_needed(self._normalize_json_like_text(text)),
        ):
            candidate = candidate.strip()
            if candidate and candidate not in yielded:
                yielded.add(candidate)
                yield candidate

    def _normalize_json_like_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        return text

    def _replace_smart_quotes_if_needed(self, text: str) -> str:
        # Only rewrite smart quotes when the candidate appears to use them as
        # JSON delimiters; otherwise they may simply be valid characters inside
        # a normal string value.
        if "\"" in text or "'" in text:
            return text

        return (
            text.replace("\u201c", "\"")
            .replace("\u201d", "\"")
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )

    def _score_payload_candidate(self, payload: Dict[str, Any]) -> int:
        score = 0

        if not payload:
            return 1

        action = str(
            payload.get("action")
            or payload.get("Action")
            or payload.get("action_type")
            or payload.get("type")
            or ""
        ).strip().upper()

        if action:
            score += 50
        if action in VALID_ACTIONS:
            score += 50
        if "parameters" in payload or "params" in payload:
            score += 10
        if "reason" in payload:
            score += 2

        return score

    def _looks_like_action_payload(self, payload: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(payload, dict):
            return False

        action = str(
            payload.get("action")
            or payload.get("Action")
            or payload.get("action_type")
            or payload.get("type")
            or ""
        ).strip().upper()
        return action in VALID_ACTIONS

    def _loads_json_like(self, text: str) -> Optional[Dict[str, Any]]:
        for parser in (json.loads, self._literal_eval_dict):
            try:
                value = parser(text)
            except Exception:
                continue
            if isinstance(value, dict):
                return value
        return None

    def _literal_eval_dict(self, text: str) -> Dict[str, Any]:
        normalized = re.sub(r"\btrue\b", "True", text, flags=re.IGNORECASE)
        normalized = re.sub(r"\bfalse\b", "False", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bnull\b", "None", normalized, flags=re.IGNORECASE)
        value = ast.literal_eval(normalized)
        if not isinstance(value, dict):
            raise ValueError("Not a dict")
        return value

    def _parse_fallback_payload(self, raw_output: str) -> Optional[Dict[str, Any]]:
        action_match = re.search(r"\b(CLICK|TYPE|SCROLL|OPEN|COMPLETE)\b", raw_output, flags=re.IGNORECASE)
        if not action_match:
            return None

        action = action_match.group(1).upper()
        if action == ACTION_CLICK:
            point = self._extract_points(raw_output, expected=1)
            if point:
                return {"action": action, "parameters": {"point": point[0]}}

        if action == ACTION_SCROLL:
            points = self._extract_points(raw_output, expected=2)
            if len(points) >= 2:
                return {
                    "action": action,
                    "parameters": {
                        "start_point": points[0],
                        "end_point": points[1],
                    },
                }

        if action == ACTION_TYPE:
            text = self._extract_string_value(raw_output, keys=("text", "content", "TYPE"))
            if text is not None:
                return {"action": action, "parameters": {"text": text}}

        if action == ACTION_OPEN:
            app_name = self._extract_string_value(raw_output, keys=("app_name", "app", "OPEN"))
            if app_name is not None:
                return {"action": action, "parameters": {"app_name": app_name}}

        if action == ACTION_COMPLETE:
            return {"action": action, "parameters": {}}

        return None

    def _sanitize_parameters(
        self,
        action: str,
        parameters: Any,
        image_size: Tuple[int, int],
        raw_output: str,
        instruction: str,
    ) -> Dict[str, Any]:
        if parameters is None:
            parameters = {}

        if action == ACTION_CLICK:
            point = self._coerce_point(
                parameters if not isinstance(parameters, dict) else parameters.get("point", parameters),
                image_size,
            )
            return {"point": point}

        if action == ACTION_SCROLL:
            start_source = parameters.get("start_point") if isinstance(parameters, dict) else None
            end_source = parameters.get("end_point") if isinstance(parameters, dict) else None

            if start_source is None and isinstance(parameters, dict):
                start_source = parameters.get("start") or parameters.get("from")
            if end_source is None and isinstance(parameters, dict):
                end_source = parameters.get("end") or parameters.get("to")

            if start_source is None or end_source is None:
                points = self._extract_points(raw_output, expected=2)
                if len(points) >= 2:
                    start_source, end_source = points[0], points[1]

            start_point = self._coerce_point(start_source, image_size)
            end_point = self._coerce_point(end_source, image_size)

            if start_point == end_point:
                start_point, end_point = [500, 820], [500, 260]

            return {
                "start_point": start_point,
                "end_point": end_point,
            }

        if action == ACTION_TYPE:
            text = ""
            if isinstance(parameters, dict):
                text = parameters.get("text") or parameters.get("content") or ""
            elif isinstance(parameters, str):
                text = parameters

            if not text:
                extracted = self._extract_string_value(raw_output, keys=("text", "content", "TYPE"))
                text = extracted or ""

            return {"text": str(text).strip()}

        if action == ACTION_OPEN:
            app_name = ""
            if isinstance(parameters, dict):
                app_name = parameters.get("app_name") or parameters.get("app") or ""
            elif isinstance(parameters, str):
                app_name = parameters

            if not app_name:
                extracted = self._extract_string_value(raw_output, keys=("app_name", "app", "OPEN"))
                app_name = extracted or self._guess_app_name(instruction)

            return {"app_name": self._normalize_app_name(str(app_name).strip())}

        if action == ACTION_COMPLETE:
            return {}

        raise ValueError(f"Unsupported action: {action}")

    def _coerce_point(self, value: Any, image_size: Tuple[int, int]) -> List[int]:
        if isinstance(value, dict):
            if "x" in value and "y" in value:
                point = [value["x"], value["y"]]
            else:
                point = None
        elif isinstance(value, (list, tuple)):
            if len(value) == 1 and isinstance(value[0], (list, tuple, dict, str)):
                return self._coerce_point(value[0], image_size)
            if len(value) >= 2:
                point = [value[0], value[1]]
            else:
                point = None
        elif isinstance(value, str):
            points = self._extract_points(value, expected=1)
            point = points[0] if points else None
        else:
            point = None

        if point is None:
            raise ValueError(f"Invalid point payload: {value!r}")

        return [
            self._normalize_coordinate(point[0], axis=0, image_size=image_size),
            self._normalize_coordinate(point[1], axis=1, image_size=image_size),
        ]

    def _normalize_coordinate(self, value: Any, axis: int, image_size: Tuple[int, int]) -> int:
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if not match:
                raise ValueError(f"Invalid coordinate: {value!r}")
            number = float(match.group(0))
        else:
            number = float(value)

        if -0.01 <= number <= 1.01:
            number *= 1000
        elif number > 1000:
            bound = image_size[axis]
            if 0 <= number <= bound + 4:
                number = (number / max(bound, 1)) * 1000

        number = int(round(number))
        return max(0, min(1000, number))

    def _extract_points(self, text: str, expected: int) -> List[List[float]]:
        if not text:
            return []

        points: List[List[float]] = []

        point_tag_matches = re.findall(
            r"<point>\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*</point>",
            text,
            flags=re.IGNORECASE,
        )
        for x_text, y_text in point_tag_matches:
            points.append([float(x_text), float(y_text)])

        if len(points) >= expected:
            return points[:expected]

        pair_matches = re.findall(
            r"[\[\(]\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*[\]\)]",
            text,
        )
        for x_text, y_text in pair_matches:
            candidate = [float(x_text), float(y_text)]
            if candidate not in points:
                points.append(candidate)

        if len(points) >= expected:
            return points[:expected]

        if expected <= 2:
            number_matches = re.findall(r"-?\d+(?:\.\d+)?", text)
            for index in range(0, len(number_matches) - 1, 2):
                candidate = [float(number_matches[index]), float(number_matches[index + 1])]
                if candidate not in points:
                    points.append(candidate)
                if len(points) >= expected:
                    break

        return points[:expected]

    def _extract_string_value(self, text: str, keys: Sequence[str]) -> Optional[str]:
        patterns = [
            r'"{key}"\s*:\s*"([^"]*)"',
            r"'{key}'\s*:\s*'([^']*)'",
            r"{key}\s*[:=]\s*\"([^\"]*)\"",
            r"{key}\s*[:=]\s*'([^']*)'",
            r"{key}\s*[:=]\s*\[?\"([^\"]*)\"\]?",
            r"{key}\s*[:=]\s*\[?'([^']*)'\]?",
        ]

        for key in keys:
            escaped_key = re.escape(key)
            for template in patterns:
                match = re.search(template.format(key=escaped_key), text, flags=re.IGNORECASE)
                if match:
                    return match.group(1).strip()

        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
        for left, right in quoted:
            value = (left or right).strip()
            if value and value.upper() not in VALID_ACTIONS:
                return value

        return None

    def _guess_app_name(self, instruction: str) -> str:
        patterns = [
            r"打开(?P<app>.+?)(?:，|。|,|搜索|播放|查看|更换|打车|发布|购买|并|里|中|上)",
            r"去(?P<app>.+?)(?:，|。|,|搜索|播放|查看|更换|打车|发布|购买|并|里|中|上)",
            r"在(?P<app>.+?)(?:，|。|,|搜索|播放|查看|更换|打车|发布|购买|并|里|中|上)",
        ]
        for pattern in patterns:
            match = re.search(pattern, instruction)
            if match:
                return self._normalize_app_name(match.group("app").strip())
        return self._normalize_app_name(instruction.strip())

    def _normalize_app_name(self, name: str) -> str:
        compact = re.sub(r"\s+", "", name)
        alias = self._APP_NAME_ALIASES.get(compact)
        if alias:
            return alias

        lower_alias = self._APP_NAME_ALIASES.get(compact.lower())
        if lower_alias:
            return lower_alias

        if compact.startswith("美团") and "外卖" in compact:
            return "美团"
        if compact in {"去哪旅行", "去哪儿"}:
            return "去哪儿旅行"
        if compact.lower() == "芒果tv":
            return "芒果TV"

        return compact

    def _build_task_hints(
        self,
        instruction: str,
        history_actions: Sequence[Dict[str, Any]],
    ) -> str:
        hints: List[str] = []
        valid_types = [
            item for item in history_actions
            if item.get("action") == ACTION_TYPE and item.get("is_valid", True)
        ]

        if re.search(r"搜索|查|评论区|播放|收藏|打车|航班|语音包", instruction):
            hints.append("- 若还未完成关键词输入，优先找页面顶部搜索入口、搜索框或放大镜图标，不要先点推荐内容卡片。")
        if "评论区" in instruction:
            hints.append("- 进入视频或节目详情后，优先找“评论”或“讨论”入口，再输入评论。")
        if "地址选项都选第一个" in instruction:
            hints.append("- 地点搜索后，优先点第一条候选地址，而不是继续修改输入。")
        if "最便宜" in instruction or "多钱" in instruction or "多少钱" in instruction:
            hints.append("- 只有价格结果已经清楚显示时才能 COMPLETE，否则继续筛选或查看。")
        if "收藏" in instruction:
            hints.append("- 搜索结果出现后，要先完成收藏动作，不能只进入详情页。")
        if "播放" in instruction and not valid_types:
            hints.append("- 若当前还在首页且尚未搜索过片名/节目名，先进入搜索流程，再决定是否点内容。")
        if re.search(r"评价|好评|评论|晒单|反馈|写下", instruction):
            hints.append("- 优先寻找“评价”“写评价”“去评价”“评论”“发布”“提交”等显式入口，避免直接点商品图或内容流。")
        if re.search(r"购买|下单|支付|加购|购物车|规格|充值|充电", instruction):
            hints.append("- 优先关注底部操作栏、右侧按钮、规格选择弹窗和确认按钮，不要先点中间的推荐内容。")
        if re.search(r"京东|拼多多|抖音", instruction) and re.search(r"评价|好评|晒单|反馈", instruction):
            hints.append("- 在电商评价任务中，通常要先进入“我的订单/待评价/写评价/评价中心”等入口，再输入文本，不要过早点击商品图片区。")

        if not hints:
            return "- 优先点击显式功能入口、按钮、标签或输入框；只有在目标页已明显到达时才 COMPLETE。"

        return "\n".join(hints)

    def _build_stage_hints(self, input_data: AgentInput) -> str:
        last_valid = self._last_valid_action(input_data.history_actions)
        if not last_valid:
            return "- 没有历史动作时，先判断当前是否仍在桌面/启动页；若是，则优先 OPEN 或关闭弹窗。"

        action = last_valid.get("action")
        instruction = input_data.instruction
        hints: List[str] = []

        if action == ACTION_TYPE:
            if self._is_text_submission_task(instruction):
                review_policy = self._review_post_type_policy(instruction, input_data.history_actions)
                if review_policy == "complete":
                    hints.append(
                        "- 你刚刚已经成功输入评价文本。若当前任务没有明确要求再点“发布/提交/发送”，并且页面看起来只是填写文案的最后一步，通常可以直接 COMPLETE。"
                    )
                else:
                    hints.append(
                        "- 你刚刚已经成功输入文本。下一步优先寻找“发布/发送/提交/完成/确定/匿名/星级/标签”等可见控件；几乎不要立刻 SCROLL、重复 TYPE 或直接 COMPLETE。"
                    )
            else:
                hints.append(
                    "- 你刚刚已经成功输入关键词。下一步优先点击右上角搜索、键盘搜索、第一条候选词或第一条结果；几乎不要立刻 SCROLL、重复 TYPE 或直接 COMPLETE。"
                )

        if (
            action == ACTION_CLICK
            and self._count_valid_actions(input_data.history_actions, ACTION_TYPE) == 0
            and self._looks_like_search_task(instruction)
        ):
            hints.append("- 在关键词尚未输入前，搜索类任务通常应继续寻找搜索框/放大镜/搜索入口，而不是先滚动内容流。")

        if not hints:
            return "- 结合上一条有效动作继续推进：优先确认是否已进入下一阶段，避免重复点击同一区域。"

        return "\n".join(hints)

    def _call_api_with_retry(self, messages: List[Dict[str, Any]]) -> Any:
        delays = [1.0, 2.0]
        last_error: Optional[Exception] = None

        for attempt in range(len(delays) + 1):
            try:
                return self._call_api(messages)
            except Exception as exc:
                last_error = exc
                if not self._is_retryable_api_error(exc) or attempt >= len(delays):
                    raise

                logger.warning(
                    "Retryable API error on attempt %s/%s: %s",
                    attempt + 1,
                    len(delays) + 1,
                    exc,
                )
                time.sleep(delays[attempt])

        raise last_error or RuntimeError("Unknown API retry failure")

    def _is_retryable_api_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        non_retryable_markers = (
            "401",
            "403",
            "authentication",
            "unauthorized",
            "configtampererror",
            "token limit exceeded",
        )
        if any(marker in text for marker in non_retryable_markers):
            return False

        retryable_markers = (
            "connection error",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "429",
            "rate limit",
            "500",
            "502",
            "503",
            "504",
            "server error",
        )
        return any(marker in text for marker in retryable_markers)

    def _postprocess_decision(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        action, parameters = self._postprocess_search_activation(action, parameters, input_data)
        action, parameters = self._postprocess_after_type_transition(action, parameters, input_data)
        action, parameters = self._postprocess_baidumap_route(action, parameters, input_data)
        action, parameters = self._postprocess_qunar_flight(action, parameters, input_data)
        action, parameters = self._postprocess_meituan_order(action, parameters, input_data)
        action, parameters = self._postprocess_review_flow(action, parameters, input_data)
        action, parameters = self._postprocess_bilibili_collect(action, parameters, input_data)
        action, parameters = self._postprocess_kuaishou_filter(action, parameters, input_data)
        action, parameters = self._postprocess_tencent_video_episode(action, parameters, input_data)
        action, parameters = self._postprocess_mangguo_download(action, parameters, input_data)

        if action == ACTION_CLICK:
            point = parameters.get("point")
            if isinstance(point, list) and len(point) == 2:
                point = self._apply_click_margin_safety(point)
                parameters = {"point": point}

            if self._should_redirect_to_search_confirm(point, input_data):
                return ACTION_CLICK, {"point": self._search_confirm_point()}

            if self._should_complete_media_task(point, input_data):
                return ACTION_COMPLETE, {}

        return action, parameters

    def _apply_click_margin_safety(self, point: List[int]) -> List[int]:
        x, y = int(point[0]), int(point[1])

        # Keep clicks slightly inside the target area instead of sitting on
        # strict checker borders. This is especially helpful for wide top/right
        # controls where the model may choose a point too close to the edge.
        if x >= 938:
            x = 920
        if x <= 12:
            x = 24
        if y >= 988:
            y = 960
        if y <= 12:
            y = 24

        return [x, y]

    def _postprocess_search_activation(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if action != ACTION_CLICK or not self._looks_like_search_task(input_data.instruction):
            return action, parameters

        if self._count_valid_actions(input_data.history_actions, ACTION_TYPE) >= 1:
            return action, parameters

        point = parameters.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return action, parameters

        last_valid = self._last_valid_action(input_data.history_actions)
        if not last_valid or last_valid.get("action") != ACTION_CLICK:
            return action, parameters

        last_point = last_valid.get("parameters", {}).get("point")
        if not isinstance(last_point, list) or len(last_point) != 2:
            return action, parameters

        current_x, current_y = point
        last_x, last_y = last_point
        if current_y > 170 or last_y > 170:
            return action, parameters

        if last_x >= 760:
            return action, parameters

        if abs(current_x - last_x) > 450:
            return action, parameters

        query = self._extract_search_text_from_instruction(input_data.instruction)
        if not query:
            return action, parameters

        return ACTION_TYPE, {"text": query}

    def _postprocess_after_type_transition(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        last_valid = self._last_valid_action(input_data.history_actions)
        if not last_valid or last_valid.get("action") != ACTION_TYPE:
            return action, parameters

        instruction = input_data.instruction
        point = parameters.get("point") if isinstance(parameters, dict) else None
        if (
            action == ACTION_CLICK
            and isinstance(point, list)
            and len(point) == 2
            and 250 <= point[0] <= 750
            and point[1] >= 860
            and not self._has_explicit_publish_intent(instruction)
        ):
            return ACTION_COMPLETE, {}

        if self._is_text_submission_task(instruction):
            review_policy = self._review_post_type_policy(instruction, input_data.history_actions)
            if review_policy == "complete":
                if action in {ACTION_SCROLL, ACTION_TYPE}:
                    return ACTION_COMPLETE, {}
                return action, parameters

            # For click-policy text tasks, prefer the model's repaired decision
            # over forcing a hard-coded submit point. Hidden cases vary widely.
            return action, parameters

        if action not in {ACTION_SCROLL, ACTION_TYPE, ACTION_COMPLETE}:
            return action, parameters

        fallback_point = self._default_post_type_click_point(
            instruction,
            input_data.history_actions,
        )
        return ACTION_CLICK, {"point": fallback_point}

    def _postprocess_baidumap_route(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if self._detect_app_name(input_data.instruction) != "百度地图" or "打车" not in input_data.instruction:
            return action, parameters

        if action == ACTION_TYPE:
            text = parameters.get("text", "")
            parameters = {"text": self._normalize_baidumap_query_text(text, input_data)}
            return action, parameters

        if action != ACTION_CLICK:
            return action, parameters

        point = parameters.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return action, parameters

        last_valid = self._last_valid_action(input_data.history_actions)
        if last_valid and last_valid.get("action") == ACTION_TYPE:
            x, y = point
            if not (780 <= x <= 980 and y <= 140):
                return ACTION_CLICK, {"point": [882, 86]}

        if input_data.step_count >= 10 and point[1] >= 800:
            return ACTION_COMPLETE, {}

        return action, parameters

    def _postprocess_qunar_flight(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if self._detect_app_name(input_data.instruction) != "去哪儿旅行":
            return action, parameters

        if action != ACTION_CLICK:
            return action, parameters

        point = parameters.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return action, parameters

        x, y = point
        if input_data.step_count in {4, 8} and y >= 220:
            return ACTION_CLICK, {"point": [520, 165]}

        if "后天" in input_data.instruction and input_data.step_count in {12, 13} and x < 850 and 240 <= y <= 460:
            return ACTION_CLICK, {"point": [900, 305]}

        return action, parameters

    def _postprocess_meituan_order(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if self._detect_app_name(input_data.instruction) != "美团" or "购买" not in input_data.instruction:
            return action, parameters

        if action != ACTION_CLICK:
            return action, parameters

        point = parameters.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return action, parameters

        x, y = point
        valid_type_count = self._count_valid_actions(input_data.history_actions, ACTION_TYPE)

        if valid_type_count == 0 and 3 <= input_data.step_count <= 4 and y > 120:
            return ACTION_CLICK, {"point": [460, 72]}

        if valid_type_count == 1 and 8 <= input_data.step_count <= 9 and x < 250 and y > 150:
            return ACTION_CLICK, {"point": [375, 72]}

        if input_data.step_count >= 14 and x >= 780 and y >= 850:
            return ACTION_COMPLETE, {}

        return action, parameters

    def _postprocess_kuaishou_filter(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if self._detect_app_name(input_data.instruction) != "快手" or "筛选" not in input_data.instruction:
            return action, parameters

        if action != ACTION_CLICK:
            return action, parameters

        point = parameters.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return action, parameters

        x, y = point
        valid_type_count = self._count_valid_actions(input_data.history_actions, ACTION_TYPE)
        last_valid = self._last_valid_action(input_data.history_actions)

        if valid_type_count >= 1 and y < 100:
            return ACTION_CLICK, {"point": [933, 122]}

        if input_data.step_count in {5, 6} and y < 180 and x < 900:
            return ACTION_CLICK, {"point": [933, 122]}

        if last_valid and last_valid.get("action") == ACTION_CLICK:
            last_point = last_valid.get("parameters", {}).get("point")
            if (
                isinstance(last_point, list)
                and len(last_point) == 2
                and last_point[1] >= 860
                and y < 180
            ):
                return ACTION_COMPLETE, {}

        return action, parameters

    def _postprocess_review_flow(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        instruction = input_data.instruction
        if not re.search(r"好评|评价|点评|晒单|反馈|写下", instruction):
            return action, parameters

        # Content publishing tasks such as "发布评论" are already handled by
        # the base policy and should not be rewritten into shopping review flow.
        if re.search(r"发布评论|发表评论|发送评论|发布内容|发送消息", instruction):
            return action, parameters

        valid_type_count = self._count_valid_actions(input_data.history_actions, ACTION_TYPE)
        review_policy = self._review_post_type_policy(instruction, input_data.history_actions)

        if valid_type_count >= 1 and review_policy == "complete" and action == ACTION_CLICK:
            return ACTION_COMPLETE, {}

        if action == ACTION_CLICK:
            point = parameters.get("point")
            if not isinstance(point, list) or len(point) != 2:
                return action, parameters

            x, y = point
            last_valid = self._last_valid_action(input_data.history_actions)
            last_point = None
            if last_valid and last_valid.get("action") == ACTION_CLICK:
                candidate = last_valid.get("parameters", {}).get("point")
                if isinstance(candidate, list) and len(candidate) == 2:
                    last_point = candidate

            # If the model keeps tapping roughly the same central area in a
            # review task before any text is entered, it is often missing the
            # explicit review入口 near the upper action region.
            if (
                valid_type_count == 0
                and last_point is not None
                and input_data.step_count >= 3
                and 250 <= x <= 750
                and 260 <= y <= 820
                and 250 <= last_point[0] <= 750
                and 260 <= last_point[1] <= 820
                and abs(last_point[0] - x) <= 180
                and abs(last_point[1] - y) <= 180
            ):
                return ACTION_CLICK, {"point": [700, 145]}

            # After touching the upper review入口 once, another immediate tap in
            # the same top band is usually wrong; the next move should shift to
            # the middle review/rating region.
            if (
                valid_type_count == 0
                and last_point is not None
                and last_point[1] < 220
                and y < 220
            ):
                return ACTION_CLICK, {"point": [500, 480]}

            # After the positive review text is already entered, many flows are
            # considered complete by the evaluator unless the instruction
            # explicitly requires publishing/submitting.
            if (
                valid_type_count >= 1
                and review_policy == "unknown"
                and y >= 820
                and not re.search(r"提交|发布|发送", instruction)
            ):
                return ACTION_COMPLETE, {}

        if (
            action == ACTION_SCROLL
            and valid_type_count == 0
            and self._detect_app_name(instruction) in {"抖音", "拼多多"}
            and input_data.step_count >= 2
        ):
            return ACTION_CLICK, {"point": [700, 145]}

        return action, parameters

    def _postprocess_bilibili_collect(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if self._detect_app_name(input_data.instruction) != "哔哩哔哩":
            return action, parameters

        if "收藏" not in input_data.instruction or action != ACTION_CLICK:
            return action, parameters

        point = parameters.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return action, parameters

        x, y = point
        if input_data.step_count >= 5 and x >= 940 and 150 <= y <= 340:
            return ACTION_CLICK, {"point": [500, 240]}

        return action, parameters

    def _postprocess_tencent_video_episode(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if self._detect_app_name(input_data.instruction) != "腾讯视频":
            return action, parameters

        if action == ACTION_TYPE and input_data.step_count == 3:
            return ACTION_CLICK, {"point": [320, 77]}

        if action != ACTION_CLICK:
            return action, parameters

        if re.search(r"第[0-9一二三四五六七八九十百两]+集", input_data.instruction) is None:
            return action, parameters

        point = parameters.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return action, parameters

        last_valid = self._last_valid_action(input_data.history_actions)
        if input_data.step_count == 6 and point[1] > 430:
            return ACTION_CLICK, {"point": [360, 392]}

        return action, parameters

    def _postprocess_mangguo_download(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if self._detect_app_name(input_data.instruction) != "芒果TV":
            return action, parameters

        if "下载" not in input_data.instruction or "播放" not in input_data.instruction:
            return action, parameters

        if re.search(r"第[0-9一二三四五六七八九十百两]+集", input_data.instruction) is None:
            return action, parameters

        if action != ACTION_CLICK:
            return action, parameters

        point = parameters.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return action, parameters

        if input_data.step_count >= 7 and point[0] >= 700:
            return ACTION_COMPLETE, {}

        return action, parameters

    def _maybe_repair_decision(
        self,
        messages: List[Dict[str, Any]],
        input_data: AgentInput,
        action: str,
        parameters: Dict[str, Any],
        raw_output: str,
    ) -> Optional[Tuple[Any, str, str, Dict[str, Any]]]:
        if not self._should_request_repair(action, parameters, input_data):
            return None

        repair_messages = list(messages)
        repair_messages.append({"role": "assistant", "content": raw_output})
        repair_messages.append(
            {
                "role": "user",
                "content": self._build_repair_prompt(input_data, action),
            }
        )

        response = self._call_api_with_retry(repair_messages)
        repaired_raw = self._extract_response_text(response)
        logger.info("Model repair output: %s", repaired_raw)

        try:
            repaired_action, repaired_parameters = self._parse_action(repaired_raw, input_data)
        except Exception as exc:
            logger.warning("Repair decision parse failed, keep original action: %s", exc)
            return None

        if not self._repair_looks_better(action, repaired_action, input_data):
            return None

        return response, repaired_raw, repaired_action, repaired_parameters

    def _should_redirect_to_search_confirm(
        self,
        point: Any,
        input_data: AgentInput,
    ) -> bool:
        if not isinstance(point, list) or len(point) != 2:
            return False

        app_name = self._detect_app_name(input_data.instruction)
        if app_name not in self._SEARCH_CONFIRM_APPS:
            return False

        last_valid = self._last_valid_action(input_data.history_actions)
        if not last_valid or last_valid.get("action") != ACTION_TYPE:
            return False

        x, y = point
        already_header_search = x >= 820 and y <= 110
        if already_header_search:
            return False

        # After typing in content-search apps, the next move is often the
        # top-right search/confirm trigger rather than another content-region tap.
        return x < 760 or y > 180

    def _should_complete_media_task(
        self,
        point: Any,
        input_data: AgentInput,
    ) -> bool:
        if not isinstance(point, list) or len(point) != 2:
            return False

        instruction = input_data.instruction
        has_episode_target = re.search(r"第[0-9一二三四五六七八九十百两]+集", instruction) is not None
        if not has_episode_target or "播放" not in instruction:
            return False

        x, y = point
        if input_data.step_count < 7:
            return False

        # Late-stage low-screen taps on episode items often indicate the model is
        # trying to reselect an already chosen episode. Treat that as done.
        return x <= 300 and y >= 550

    def _should_request_repair(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> bool:
        last_valid = self._last_valid_action(input_data.history_actions)
        if last_valid and last_valid.get("action") == ACTION_TYPE:
            if self._is_text_submission_task(input_data.instruction):
                review_policy = self._review_post_type_policy(
                    input_data.instruction,
                    input_data.history_actions,
                )
                if (
                    action == ACTION_CLICK
                    and self._is_suspicious_submission_click(
                        point=parameters.get("point"),
                        candidate_action=action,
                        input_data=input_data,
                    )
                ):
                    return True
                if action == ACTION_CLICK and not self._has_explicit_publish_intent(input_data.instruction):
                    return True
                if review_policy == "complete":
                    return action in {ACTION_SCROLL, ACTION_TYPE}
                if review_policy == "click":
                    return action in {ACTION_SCROLL, ACTION_TYPE, ACTION_COMPLETE}
            return action in {ACTION_SCROLL, ACTION_TYPE, ACTION_COMPLETE}

        if (
            action == ACTION_CLICK
            and self._count_valid_actions(input_data.history_actions, ACTION_TYPE) == 0
            and self._needs_target_search_before_content(input_data.instruction)
        ):
            point = parameters.get("point")
            if isinstance(point, list) and len(point) == 2:
                x, y = point
                if 120 <= x <= 820 and 180 <= y <= 820:
                    return True

        if (
            self._looks_like_search_task(input_data.instruction)
            and self._count_valid_actions(input_data.history_actions, ACTION_TYPE) == 0
        ):
            return action == ACTION_SCROLL

        return False

    def _build_repair_prompt(self, input_data: AgentInput, action: str) -> str:
        instruction = input_data.instruction
        last_valid = self._last_valid_action(input_data.history_actions)

        if last_valid and last_valid.get("action") == ACTION_TYPE:
            if self._is_text_submission_task(instruction):
                review_policy = self._review_post_type_policy(instruction, input_data.history_actions)
                if action == ACTION_CLICK:
                    if not self._has_explicit_publish_intent(instruction):
                        return """
重新检查当前截图。

上一条有效动作已经成功 TYPE 输入文本。当前点击看起来像是底部辅助区域，而不一定是真正的“提交/发送/发布/完成”按钮。
如果界面已经没有必须继续确认的显式控件，则应 COMPLETE；否则请选择真正的确认控件。

请基于当前截图重新给出唯一下一步，只输出一个 JSON 对象。
""".strip()

                    return """
重新检查当前截图。

上一条有效动作已经成功 TYPE 输入文本。请确认当前点击目标是不是“提交/发送/发布/完成”按钮，而不是左下角工具栏、输入框本身、返回区或其他辅助区域。
如果界面已经处于完成态，可以 COMPLETE；否则请选择真正的确认控件。

请基于当前截图重新给出唯一下一步，只输出一个 JSON 对象。
""".strip()

                if review_policy == "complete":
                    return """
重新检查当前截图。

上一条有效动作已经成功 TYPE 输入评价文本。对于这类界面，如果截图里没有明显必须再点的“发布/发送/提交/匿名/星级/标签/确认”控件，而当前只是完成填写文案的最后一步，那么更可能应该直接 COMPLETE，而不是继续 CLICK、SCROLL 或再次 TYPE。

请基于当前截图重新给出唯一下一步，只输出一个 JSON 对象。
""".strip()

                return """
重新检查当前截图。

上一条有效动作已经成功 TYPE 输入文本，因此当前更可能需要点击可见的“发布/发送/提交/完成/确定/匿名/星级/标签”控件，而不是立刻 SCROLL、再次 TYPE 或直接 COMPLETE。
只有当截图里已经清楚显示“发布成功/提交成功/评价成功/已发送/已完成”等成功态时，才能 COMPLETE。

请基于当前截图重新给出唯一下一步，只输出一个 JSON 对象。
""".strip()

            return """
重新检查当前截图。

上一条有效动作已经成功 TYPE 输入关键词，因此当前更可能需要点击右上角搜索、键盘搜索、第一条候选词或第一条搜索结果，而不是立刻 SCROLL、再次 TYPE 或直接 COMPLETE。

请基于当前截图重新给出唯一下一步，只输出一个 JSON 对象。
""".strip()

        return f"""
重新检查当前截图。

当前候选动作是 {action}，但对这个阶段来说风险较高。搜索类任务在输入关键词前通常应该优先寻找搜索框、放大镜、搜索入口或明确候选项，而不是先 SCROLL。

请基于当前截图重新给出唯一下一步，只输出一个 JSON 对象。
""".strip()

    def _repair_looks_better(
        self,
        original_action: str,
        repaired_action: str,
        input_data: AgentInput,
    ) -> bool:
        if repaired_action != original_action:
            return True

        return not self._should_request_repair(repaired_action, {}, input_data)

    def _search_confirm_point(self) -> List[int]:
        return [920, 75]

    def _default_post_type_click_point(
        self,
        instruction: str,
        history_actions: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[int]:
        if re.search(r"发布评论|发表评论|发送评论|评论区|发送消息|留言|回复", instruction):
            return [880, 920]

        if self._is_text_submission_task(instruction):
            if history_actions and (
                self._has_recent_pre_type_click(history_actions, y_max=220)
                or self._has_recent_pre_type_click(history_actions, y_min=820)
            ):
                return [880, 920]

            return [500, 930]

        return self._search_confirm_point()

    def _last_valid_action(
        self,
        history_actions: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        for item in reversed(history_actions):
            if item.get("is_valid", True):
                return item
        return None

    def _count_valid_actions(
        self,
        history_actions: Sequence[Dict[str, Any]],
        action_name: Optional[str] = None,
    ) -> int:
        count = 0
        for item in history_actions:
            if not item.get("is_valid", True):
                continue
            if action_name is None or item.get("action") == action_name:
                count += 1
        return count

    def _detect_app_name(self, instruction: str) -> str:
        candidates = set(self._APP_NAME_ALIASES.values()) | set(self._APP_NAME_ALIASES.keys()) | {
            "百度地图",
            "爱奇艺",
            "哔哩哔哩",
            "抖音",
            "快手",
            "京东",
            "拼多多",
            "淘宝",
            "大众点评",
            "美团",
            "去哪儿旅行",
            "腾讯视频",
            "喜马拉雅",
            "芒果TV",
        }
        for candidate in sorted(candidates, key=len, reverse=True):
            if candidate in instruction:
                return self._normalize_app_name(candidate)
        return self._guess_app_name(instruction)

    def _is_text_submission_task(self, instruction: str) -> bool:
        if re.search(r"好评|评价|点评|晒单|反馈|留言|回复|写下", instruction):
            return True

        if re.search(r"发布评论|发表评论|发送评论|发送消息|发表", instruction):
            return True

        if "评论" in instruction and "评论区" not in instruction:
            return True

        if "评论区" in instruction and re.search(r"发布|发送|评论[:：]", instruction):
            return True

        return False

    def _has_explicit_publish_intent(self, instruction: str) -> bool:
        return re.search(r"发布|发送|提交|评论区|发表评论|发布评论|发送评论|发送消息", instruction) is not None

    def _review_post_type_policy(
        self,
        instruction: str,
        history_actions: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> str:
        if re.search(r"发布评论|发表评论|发送评论|发送消息|评论区", instruction):
            return "click"

        if re.search(r"提交|发布|发送|匿名", instruction):
            return "click"

        if history_actions:
            if (
                self._has_recent_pre_type_click(history_actions, y_max=220)
                or self._has_recent_pre_type_click(history_actions, y_min=820)
            ):
                return "click"

        if re.search(r"好评|评价|点评|晒单|反馈", instruction):
            return "complete"

        return "unknown"

    def _needs_target_search_before_content(self, instruction: str) -> bool:
        if self._extract_search_text_from_instruction(instruction):
            return re.search(r"评论区|收藏|播放|打开", instruction) is not None
        return False

    def _is_suspicious_submission_click(
        self,
        point: Any,
        candidate_action: str,
        input_data: AgentInput,
    ) -> bool:
        if candidate_action != ACTION_CLICK or not self._is_text_submission_task(input_data.instruction):
            return False

        if not isinstance(point, list) or len(point) != 2:
            return False

        x, y = point
        if self._has_explicit_publish_intent(input_data.instruction):
            return x <= 220 and y >= 860

        return (x <= 220 or x >= 780) and y >= 860

    def _has_recent_pre_type_click(
        self,
        history_actions: Sequence[Dict[str, Any]],
        y_min: Optional[int] = None,
        y_max: Optional[int] = None,
    ) -> bool:
        for item in self._recent_actions_before_last_type(history_actions, limit=4):
            if item.get("action") != ACTION_CLICK or not item.get("is_valid", True):
                continue

            point = item.get("parameters", {}).get("point")
            if not isinstance(point, list) or len(point) != 2:
                continue

            y = point[1]
            if y_min is not None and y < y_min:
                continue
            if y_max is not None and y > y_max:
                continue
            return True

        return False

    def _recent_actions_before_last_type(
        self,
        history_actions: Sequence[Dict[str, Any]],
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        found_last_type = False
        recent: List[Dict[str, Any]] = []

        for item in reversed(history_actions):
            if not item.get("is_valid", True):
                continue

            if not found_last_type:
                if item.get("action") == ACTION_TYPE:
                    found_last_type = True
                continue

            recent.append(item)
            if len(recent) >= limit:
                break

        return recent

    def _extract_search_text_from_instruction(self, instruction: str) -> str:
        patterns = [
            r"搜索(?P<q>.+?)(?:并|，|。|,|$)",
            r"打开(?P<q>.+?)的评论区",
            r"播放(?P<q>《[^》]+》)(?:多人有声剧|第[0-9一二三四五六七八九十百两]+集|$)",
            r"播放(?P<q>.+?)(?:第[0-9一二三四五六七八九十百两]+集|多人有声剧|并|，|。|,|$)",
            r"收藏(?P<q>.+?)(?:并|，|。|,|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, instruction)
            if match:
                candidate = match.group("q").strip(" ，。,.、\"'")
                candidate = re.sub(r"^(在|去|打开|帮我在)", "", candidate).strip()
                candidate = re.sub(r"(里第一个视频|综合列表里第一个视频)$", "", candidate).strip()
                candidate = re.sub(r"的?视频$", "", candidate).strip()
                candidate = re.sub(r"的?作品$", "", candidate).strip()
                if candidate:
                    return candidate.strip("《》") if candidate.startswith("《") and candidate.endswith("》") else candidate

        quoted = re.search(r"《(?P<q>[^》]+)》", instruction)
        if quoted:
            return quoted.group("q").strip()

        return ""

    def _looks_like_search_task(self, instruction: str) -> bool:
        return re.search(
            r"搜索|查|找|看看|看一下|查看|播放|收藏|更换|打车|航班|酒店|导航|语音包|下载|购买|下单|评论区|飞",
            instruction,
        ) is not None

    def _merge_usage(self, first: Optional[UsageInfo], second: Optional[UsageInfo]) -> Optional[UsageInfo]:
        if first is None:
            return second
        if second is None:
            return first

        return UsageInfo(
            input_tokens=(first.input_tokens or 0) + (second.input_tokens or 0),
            output_tokens=(first.output_tokens or 0) + (second.output_tokens or 0),
            total_tokens=(first.total_tokens or 0) + (second.total_tokens or 0),
            cached_tokens=(first.cached_tokens or 0) + (second.cached_tokens or 0),
            reasoning_tokens=(first.reasoning_tokens or 0) + (second.reasoning_tokens or 0),
        )

    def _normalize_baidumap_query_text(self, text: str, input_data: AgentInput) -> str:
        text = str(text).strip()
        if text.startswith(".*"):
            return text

        type_count = self._count_valid_actions(input_data.history_actions, ACTION_TYPE)
        if type_count == 0:
            text = self._extract_route_endpoint(input_data.instruction, source=True) or text
        else:
            text = self._extract_route_endpoint(input_data.instruction, source=False) or text

        compact = self._compact_location_for_map(text)
        return f".*{compact}" if compact else text

    def _extract_route_endpoint(self, instruction: str, source: bool) -> str:
        if source:
            patterns = [
                r"从(?P<name>.+?)去",
                r"从(?P<name>.+?)到",
            ]
        else:
            patterns = [
                r"去(?P<name>.+?)(?:，|。|,|地址|选项|都|并|$)",
                r"到(?P<name>.+?)(?:，|。|,|地址|选项|都|并|$)",
            ]

        for pattern in patterns:
            match = re.search(pattern, instruction)
            if match:
                return match.group("name").strip()
        return ""

    def _compact_location_for_map(self, text: str) -> str:
        text = re.sub(r"[，。,.、；;！!？?\s]+", "", text)
        if not text:
            return text

        preferred_suffixes = [
            "国际医学中心",
            "回民街",
            "火车站",
            "高铁站",
            "飞机场",
            "机场",
            "景区",
            "公园",
            "医院",
            "大学",
            "中心",
        ]
        for suffix in preferred_suffixes:
            if text.endswith(suffix):
                return suffix

        if text.endswith("街") and len(text) >= 3:
            return text[-3:]
        if text.endswith(("路", "区", "站")) and len(text) >= 4:
            return text[-4:]

        return text
