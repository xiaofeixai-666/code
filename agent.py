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
        "抖音短视频": "抖音",
        "抖音app": "抖音",
        "快手app": "快手",
        "xiaohongshu": "小红书",
        "小红书app": "小红书",
        "taobao": "淘宝",
        "淘宝app": "淘宝",
        "手机淘宝": "淘宝",
        "tmall": "天猫",
        "天猫app": "天猫",
        "dazhongdianping": "大众点评",
        "大众点评app": "大众点评",
        "饿了么app": "饿了么",
        "eleme": "饿了么",
        "携程旅行": "携程",
        "携程app": "携程",
        "ctrip": "携程",
        "高德地图app": "高德地图",
        "amap": "高德地图",
        "去哪旅行": "去哪儿旅行",
        "去哪儿": "去哪儿旅行",
        "芒果tv": "芒果TV",
        "芒果Tv": "芒果TV",
        "芒果tv视频": "芒果TV",
        "bilibili": "哔哩哔哩",
        "哔哩": "哔哩哔哩",
        "b站": "哔哩哔哩",
        "qq音乐": "QQ音乐",
        "qqmusic": "QQ音乐",
        "网易云": "网易云音乐",
        "网易云音乐app": "网易云音乐",
    }
    _ACTION_EXAMPLES = """
Examples:
{"action":"CLICK","parameters":{"point":[875,72]}}
{"action":"TYPE","parameters":{"text":"关键词"}}
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
- For review/form tasks, TYPE only after the current screenshot clearly shows an active text field/keyboard, or after enough visible form controls have been selected. In shopping/order review forms, one or two clicks usually only enter the form and choose a rating; click tags/text field/anonymous options before TYPE.
- Use COMPLETE only when the user goal is already achieved or the requested result is clearly visible.
- Prefer explicit UI controls over semantically related content cards. For example, when the task requires searching, first use the visible search entrance instead of tapping a recommended content card.
- In shopping, review, or form flows, prioritize explicit buttons and fields such as search bars, review/评价 buttons, submit buttons, spec selectors, pay buttons, and confirmation controls.
- If the task is to review, comment, rate, submit feedback, or publish content, first locate the explicit entry to that function instead of clicking unrelated content.
- If the task is to buy, pay, recharge, add to cart, or choose specifications, prefer the bottom action bar or right-side call-to-action buttons over content tiles.
- If the task is to like/upvote a short video or post, first make sure the target content is actually opened. If the current screen is still a list, search result, profile, or recommendation grid, open the target content/card first. Once inside the content detail or full-screen feed, use the visible heart/thumb-up/like control; in vertical video UIs, the like button is usually above comment/share in the right-side icon stack.
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
6. 如果任务涉及“评价/好评/评论/晒单/反馈”，优先寻找“评价”“写评价”“去评价”“评论”“发布”“提交”等显式入口；在文本框真正激活前不要直接 TYPE。
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
            point = self._coerce_point_from_dict(value, image_size)
            if point is not None:
                return point
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

    def _coerce_point_from_dict(
        self,
        value: Dict[str, Any],
        image_size: Tuple[int, int],
    ) -> Optional[List[int]]:
        nested = value.get("parameters") if isinstance(value.get("parameters"), dict) else {}
        x_value = self._first_mapping_value(
            value,
            ("x", "normalized_x", "norm_x", "coordinate_x", "center_x"),
        )
        if x_value is None:
            x_value = self._first_mapping_value(
                nested,
                ("x", "normalized_x", "norm_x", "coordinate_x", "center_x"),
            )

        if x_value is not None:
            y_value = self._first_mapping_value(
                value,
                ("y", "normalized_y", "norm_y", "coordinate_y", "center_y"),
            )
            if y_value is None:
                y_value = self._first_mapping_value(
                    nested,
                    ("y", "normalized_y", "norm_y", "coordinate_y", "center_y"),
                )
            if y_value is None:
                y_value = self._find_numeric_dict_partner(
                    value,
                    excluded_keys={
                        "x",
                        "normalized_x",
                        "norm_x",
                        "coordinate_x",
                        "center_x",
                        "parameters",
                    },
                )
            if y_value is not None:
                return [
                    self._normalize_coordinate(x_value, axis=0, image_size=image_size),
                    self._normalize_coordinate(y_value, axis=1, image_size=image_size),
                ]

        if "point" in value:
            point_value = value.get("point")
            if "y" in value and not isinstance(point_value, (list, tuple, dict)):
                return [
                    self._normalize_coordinate(point_value, axis=0, image_size=image_size),
                    self._normalize_coordinate(value["y"], axis=1, image_size=image_size),
                ]
            return self._coerce_point(point_value, image_size)

        if isinstance(nested, dict):
            return self._coerce_point_from_dict(nested, image_size)

        return None

    def _first_mapping_value(
        self,
        value: Dict[str, Any],
        keys: Sequence[str],
    ) -> Optional[Any]:
        lowered = {str(key).lower(): item for key, item in value.items()}
        for key in keys:
            if key in lowered and lowered[key] is not None:
                return lowered[key]
        return None

    def _find_numeric_dict_partner(
        self,
        value: Dict[str, Any],
        excluded_keys: Optional[set] = None,
    ) -> Optional[Any]:
        excluded_keys = excluded_keys or set()
        for key, item in value.items():
            if key in excluded_keys:
                continue
            if self._is_numberish(item):
                return item
            if self._is_numberish(key):
                return key
        return None

    def _is_numberish(self, value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            return re.fullmatch(r"-?\d+(?:\.\d+)?", value.strip()) is not None
        return False

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
            r"打开(?P<app>.+?)(?:，|。|,|搜索|播放|查看|更换|打车|发布|购买|下单|评价|评论|点赞|点(?=赞|爱心|红心|小红心|喜欢)|给|并|里|中|上)",
            r"去(?P<app>.+?)(?:，|。|,|搜索|播放|查看|更换|打车|发布|购买|下单|评价|评论|点赞|点(?=赞|爱心|红心|小红心|喜欢)|给|并|里|中|上)",
            r"在(?P<app>.+?)(?:，|。|,|搜索|播放|查看|更换|打车|发布|购买|下单|评价|评论|点赞|点(?=赞|爱心|红心|小红心|喜欢)|给|并|里|中|上)",
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
        if self._looks_like_quick_like_task(instruction):
            hints.append("- 点赞/点爱心/点红心/喜欢类任务要先确认目标内容已经打开；若还在列表/搜索结果/推荐流入口，先点目标内容卡片，进入详情或全屏播放后再点心形、拇指或“赞”按钮。")
        if re.search(r"关注|关住|粉一下|作者|博主|主页|头像", instruction):
            hints.append("- 关注作者/进入作者主页/查看博主资料这类内容关系任务，也要先确认目标内容或目标作者卡片已经打开；若仍在列表或推荐入口，先点内容卡片，不要直接点右侧栏中下部图标。")
        if self._looks_like_initial_content_social_task(instruction):
            hints.append("- 评论区/收藏/分享/私信等内容关系任务，如果目标写成“第一个/列表里/搜索结果里”的内容，首步应先打开目标内容卡片；进入详情或全屏内容页后，再点右侧对应图标。")
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
        if re.search(r"评价|好评|评论|晒单|反馈|写下|追评|评语|差评|中评|五星|星级|晒图|买家秀", instruction):
            hints.append("- 优先寻找“评价”“写评价”“去评价”“评论”“发布”“提交”等显式入口，避免直接点商品图或内容流。")
        if self._is_review_form_task(instruction) and not valid_types:
            click_count = self._count_valid_actions(history_actions, ACTION_CLICK)
            if click_count < 3:
                hints.append("- 对填写评价/晒单/反馈这类表单任务，真正 TYPE 前通常还需要先点最高星级/好评标签/文本框/匿名开关等显式输入控件；如果正文框上方有推荐标签，先点一个合适的正向标签，再进入正文输入。")
            else:
                hints.append("- 已经完成多次评价表单点击后，若当前截图里文本框处于焦点、键盘弹出，或正重复点击正文输入区，就应 TYPE 指令里的评价文案。")
        if re.search(r"购买|下单|支付|加购|购物车|规格|充值|充电", instruction):
            hints.append("- 优先关注底部操作栏、右侧按钮、规格选择弹窗和确认按钮，不要先点中间的推荐内容。")
        if re.search(r"评价|好评|晒单|反馈|写下|追评|评语|差评|中评|五星|星级|晒图|买家秀", instruction) and not valid_types:
            hints.append("- 在文本评价类任务中，通常要先进入明确的“待评价/写评价/评价中心/评论入口”等功能页，再输入文本，不要过早点击内容图片区。")

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

        if (
            action == ACTION_CLICK
            and self._count_valid_actions(input_data.history_actions, ACTION_TYPE) == 0
            and self._is_review_form_task(instruction)
        ):
            click_count = self._count_valid_actions(input_data.history_actions, ACTION_CLICK)
            if click_count < 3:
                hints.append("- 当前还没完成文本输入，而且评价表单预处理点击还少；优先点最高星级、好评标签、文本框或匿名等控件，不要急着 TYPE。")
            else:
                hints.append("- 当前还没完成文本输入，但已经点过多个评价控件；若正文输入区/键盘已出现，下一步通常应直接 TYPE 文案。")

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
        action, parameters = self._postprocess_pretype_field_click(action, parameters, input_data)
        action, parameters = self._postprocess_search_keyboard_misclick_to_type(action, parameters, input_data)
        action, parameters = self._postprocess_search_field_entry(action, parameters, input_data)
        action, parameters = self._postprocess_review_type_after_tag_delay(action, parameters, input_data)
        action, parameters = self._postprocess_review_form_typing(action, parameters, input_data)
        action, parameters = self._postprocess_review_preinput_tag_alignment(action, parameters, input_data)
        action, parameters = self._postprocess_review_form_entry_click(action, parameters, input_data)
        action, parameters = self._postprocess_after_type_transition(action, parameters, input_data)
        action, parameters = self._postprocess_structured_field_entry(action, parameters, input_data)
        action, parameters = self._postprocess_followup_field_reentry(action, parameters, input_data)
        action, parameters = self._postprocess_route_query(action, parameters, input_data)
        action, parameters = self._postprocess_route_completion(action, parameters, input_data)
        action, parameters = self._postprocess_filter_completion(action, parameters, input_data)
        action, parameters = self._postprocess_filter_option_alignment(action, parameters, input_data)
        action, parameters = self._postprocess_review_flow(action, parameters, input_data)
        action, parameters = self._postprocess_initial_content_rail_misclick(action, parameters, input_data)
        action, parameters = self._postprocess_reaction_icon_alignment(action, parameters, input_data)
        action, parameters = self._postprocess_episode_row_alignment(action, parameters, input_data)
        action, parameters = self._postprocess_price_result_alignment(action, parameters, input_data)
        action, parameters = self._postprocess_price_answer_completion(action, parameters, input_data)
        action, parameters = self._postprocess_redundant_confirmation_completion(action, parameters, input_data)

        if action == ACTION_CLICK:
            point = parameters.get("point")
            if isinstance(point, list) and len(point) == 2:
                point = self._apply_click_margin_safety(point)
                parameters = {"point": point}

            if self._should_redirect_to_search_confirm(point, input_data):
                return ACTION_CLICK, {"point": self._search_confirm_point(input_data)}

            if self._should_complete_media_task(point, input_data):
                return ACTION_COMPLETE, {}

        return action, parameters

    def _postprocess_route_query(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if action != ACTION_TYPE or not self._looks_like_route_task(input_data.instruction):
            return action, parameters

        text = str(parameters.get("text", "")).strip()
        if text.startswith(".*"):
            return action, parameters

        type_count = self._count_valid_actions(input_data.history_actions, ACTION_TYPE)
        endpoint = self._extract_route_endpoint(input_data.instruction, source=(type_count == 0))
        normalized = self._compact_route_location(endpoint or text)
        if normalized:
            return ACTION_TYPE, {"text": f".*{normalized}"}

        return action, parameters

    def _postprocess_followup_field_reentry(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if (
            action == ACTION_SCROLL
            and self._needs_catalog_followup_input(input_data.instruction)
            and self._count_valid_actions(input_data.history_actions, ACTION_TYPE) == 1
            and self._has_valid_click_after_last_type(input_data.history_actions)
        ):
            return ACTION_CLICK, {"point": self._structured_followup_entry_point(input_data.instruction)}

        if action != ACTION_CLICK:
            return action, parameters

        if self._count_valid_actions(input_data.history_actions, ACTION_TYPE) != 1:
            return action, parameters

        is_catalog = self._needs_catalog_followup_input(input_data.instruction)
        is_travel = self._needs_travel_followup_input(input_data.instruction)
        if not is_catalog and not is_travel:
            return action, parameters

        if not self._has_valid_click_after_last_type(input_data.history_actions):
            return action, parameters

        point = parameters.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return action, parameters

        if is_catalog:
            if point[1] < 450:
                return action, parameters
            return ACTION_CLICK, {"point": self._structured_followup_entry_point(input_data.instruction)}

        if is_travel and self._just_selected_first_travel_candidate(input_data.history_actions):
            if point[1] >= 420:
                return ACTION_CLICK, {"point": self._structured_followup_entry_point(input_data.instruction)}

        return action, parameters

    def _postprocess_price_result_alignment(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if action != ACTION_CLICK:
            return action, parameters

        if re.search(r"最便宜|多少钱|多钱", input_data.instruction) is None:
            return action, parameters

        if self._count_valid_actions(input_data.history_actions, ACTION_TYPE) < 2:
            return action, parameters

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

        if (
            last_point
            and last_point[0] <= 400
            and 320 <= last_point[1] <= 400
            and 320 <= x <= 850
            and 250 <= y <= 420
        ):
            return ACTION_CLICK, {"point": [900, 300]}

        if 720 <= x <= 850 and 260 <= y <= 340:
            return ACTION_CLICK, {"point": [900, y]}

        return action, parameters

    def _postprocess_redundant_confirmation_completion(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if action != ACTION_CLICK or self._is_text_submission_task(input_data.instruction):
            return action, parameters

        if re.search(r"购买|下单|支付|购物车|加购|地址选择|默认地址|规格", input_data.instruction) is None:
            return action, parameters

        point = parameters.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return action, parameters

        if point[1] < 860:
            return action, parameters

        last_valid = self._last_valid_action(input_data.history_actions)
        if not last_valid or last_valid.get("action") != ACTION_CLICK:
            return action, parameters

        last_point = last_valid.get("parameters", {}).get("point")
        if not isinstance(last_point, list) or len(last_point) != 2:
            return action, parameters

        if last_point[1] >= 860 and input_data.step_count >= 12:
            return ACTION_COMPLETE, {}

        return action, parameters

    def _postprocess_episode_row_alignment(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if action != ACTION_CLICK:
            return action, parameters

        if re.search(r"第[0-9一二三四五六七八九十百两]+集", input_data.instruction) is None:
            return action, parameters

        point = parameters.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return action, parameters

        x, y = point
        if 320 <= x <= 700 and 430 <= y <= 520:
            return ACTION_CLICK, {"point": [x, 390]}

        return action, parameters

    def _postprocess_price_answer_completion(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if action not in {ACTION_COMPLETE, ACTION_SCROLL}:
            return action, parameters

        if re.search(r"最便宜|多少钱|多钱", input_data.instruction) is None:
            return action, parameters

        if self._count_valid_actions(input_data.history_actions, ACTION_TYPE) < 2:
            return action, parameters

        last_valid = self._last_valid_action(input_data.history_actions)
        if not last_valid or last_valid.get("action") != ACTION_CLICK:
            return action, parameters

        point = last_valid.get("parameters", {}).get("point")
        if not isinstance(point, list) or len(point) != 2:
            return action, parameters

        if action == ACTION_COMPLETE and 560 <= point[1] <= 680:
            return ACTION_CLICK, {"point": [500, 360]}

        if action == ACTION_SCROLL and 280 <= point[1] <= 450:
            return ACTION_COMPLETE, {}

        return action, parameters

    def _postprocess_structured_field_entry(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if action != ACTION_CLICK:
            return action, parameters

        type_count = self._count_valid_actions(input_data.history_actions, ACTION_TYPE)
        if type_count >= 1:
            return action, parameters

        if not self._needs_target_search_before_content(input_data.instruction):
            return action, parameters

        click_count = self._count_valid_actions(input_data.history_actions, ACTION_CLICK)
        if click_count < 2:
            return action, parameters

        point = parameters.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return action, parameters

        x, y = point
        if not (120 <= x <= 820 and 180 <= y <= 820):
            return action, parameters

        return ACTION_CLICK, {"point": self._structured_field_entry_point(input_data.instruction)}

    def _postprocess_filter_completion(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if action != ACTION_CLICK or "筛选" not in input_data.instruction:
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

        if last_point[1] >= 860 and point[1] < 180:
            return ACTION_COMPLETE, {}

        return action, parameters

    def _postprocess_filter_option_alignment(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if action != ACTION_CLICK or "筛选" not in input_data.instruction:
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

        if 100 <= last_point[1] <= 170 and 80 <= point[1] <= 180 and point[0] < 700:
            return ACTION_CLICK, {"point": [920, 122]}

        return action, parameters

    def _postprocess_route_completion(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if action != ACTION_CLICK or not self._looks_like_route_task(input_data.instruction):
            return action, parameters

        if self._count_valid_actions(input_data.history_actions, ACTION_TYPE) < 2:
            return action, parameters

        point = parameters.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return action, parameters

        if point[1] >= 820:
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

        if re.search(r"我的下载|下载里的|已下载|离线", input_data.instruction):
            return action, parameters

        if self._requires_explicit_field_entry_before_typing(input_data.instruction):
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
        if current_y > 170:
            if (
                last_y <= 130
                and last_x < 760
                and self._count_recent_top_clicks(input_data.history_actions) >= 2
            ):
                query = self._extract_search_text_from_instruction(input_data.instruction)
                if query:
                    return ACTION_TYPE, {"text": query}
            return action, parameters

        if last_y > 170:
            return action, parameters

        if last_x >= 760:
            return action, parameters

        if abs(current_x - last_x) > 450:
            return action, parameters

        query = self._extract_search_text_from_instruction(input_data.instruction)
        if not query:
            return action, parameters

        return ACTION_TYPE, {"text": query}

    def _count_recent_top_clicks(self, history_actions: Sequence[Dict[str, Any]], limit: int = 4) -> int:
        count = 0
        checked = 0
        for item in reversed(history_actions):
            if not item.get("is_valid", True):
                continue
            if item.get("action") != ACTION_CLICK:
                continue

            checked += 1
            point = item.get("parameters", {}).get("point")
            if isinstance(point, list) and len(point) == 2 and point[1] <= 130:
                count += 1

            if checked >= limit:
                break

        return count

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

        if (
            action == ACTION_CLICK
            and isinstance(point, list)
            and len(point) == 2
            and point[0] <= 300
            and point[1] >= 200
        ):
            fallback_point = self._default_post_type_click_point(
                instruction,
                input_data.history_actions,
            )
            return ACTION_CLICK, {"point": fallback_point}

        if (
            action == ACTION_CLICK
            and isinstance(point, list)
            and len(point) == 2
            and "筛选" in instruction
            and point[1] <= 90
        ):
            return ACTION_CLICK, {"point": [500, 120]}

        if action not in {ACTION_SCROLL, ACTION_TYPE, ACTION_COMPLETE}:
            return action, parameters

        fallback_point = self._default_post_type_click_point(
            instruction,
            input_data.history_actions,
        )
        return ACTION_CLICK, {"point": fallback_point}

    def _postprocess_search_field_entry(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if action != ACTION_CLICK or not self._looks_like_search_task(input_data.instruction):
            return action, parameters

        if self._count_valid_actions(input_data.history_actions, ACTION_TYPE) >= 1:
            return action, parameters

        if self._requires_explicit_field_entry_before_typing(input_data.instruction):
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

        if last_point[0] >= 720 and last_point[1] <= 120 and 120 <= point[0] <= 820 and point[1] >= 350:
            return ACTION_CLICK, {"point": [500, 72]}

        return action, parameters

    def _postprocess_search_keyboard_misclick_to_type(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if action != ACTION_CLICK or not self._looks_like_search_task(input_data.instruction):
            return action, parameters

        if re.search(r"我的下载|下载里的|已下载|离线", input_data.instruction):
            return action, parameters

        if self._count_valid_actions(input_data.history_actions, ACTION_TYPE) >= 1:
            return action, parameters

        point = parameters.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return action, parameters

        x, y = point
        if y < 760 or x < 650:
            return action, parameters

        last_valid = self._last_valid_action(input_data.history_actions)
        if not last_valid or last_valid.get("action") != ACTION_CLICK:
            return action, parameters

        last_point = last_valid.get("parameters", {}).get("point")
        if not isinstance(last_point, list) or len(last_point) != 2:
            return action, parameters

        last_x, last_y = last_point
        if not (120 <= last_x <= 850 and last_y <= 130):
            return action, parameters

        query = self._extract_search_text_from_instruction(input_data.instruction)
        if not query:
            return action, parameters

        return ACTION_TYPE, {"text": query}

    def _postprocess_pretype_field_click(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if action != ACTION_TYPE or not self._looks_like_search_task(input_data.instruction):
            return action, parameters

        if self._count_valid_actions(input_data.history_actions, ACTION_TYPE) >= 1:
            return action, parameters

        if self._looks_like_route_task(input_data.instruction):
            return action, parameters

        last_valid = self._last_valid_action(input_data.history_actions)
        if not last_valid or last_valid.get("action") != ACTION_CLICK:
            return action, parameters

        last_point = last_valid.get("parameters", {}).get("point")
        if not isinstance(last_point, list) or len(last_point) != 2:
            return action, parameters

        if last_point[0] >= 720 and last_point[1] <= 120:
            return ACTION_CLICK, {"point": [500, 72]}

        return action, parameters

    def _postprocess_review_type_after_tag_delay(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if action != ACTION_TYPE or not self._is_review_form_task(input_data.instruction):
            return action, parameters

        if self._count_valid_actions(input_data.history_actions, ACTION_TYPE) >= 1:
            return action, parameters

        last_valid = self._last_valid_action(input_data.history_actions)
        last_point = last_valid.get("parameters", {}).get("point") if last_valid else None
        if self._is_review_tag_row_click(last_point):
            return ACTION_CLICK, {"point": [450, 380]}

        return action, parameters

    def _postprocess_review_form_typing(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if action != ACTION_CLICK or not self._is_review_form_task(input_data.instruction):
            return action, parameters

        if self._count_valid_actions(input_data.history_actions, ACTION_TYPE) >= 1:
            return action, parameters

        inline_text = self._extract_inline_freeform_text(input_data.instruction)
        click_count = self._count_valid_actions(input_data.history_actions, ACTION_CLICK)
        if click_count < 3:
            return action, parameters

        point = parameters.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return action, parameters

        x, y = point
        if not (220 <= x <= 820 and 240 <= y <= 680):
            return action, parameters

        last_valid = self._last_valid_action(input_data.history_actions)
        if not last_valid or last_valid.get("action") != ACTION_CLICK:
            return action, parameters

        last_point = last_valid.get("parameters", {}).get("point")
        if not isinstance(last_point, list) or len(last_point) != 2:
            return action, parameters

        last_x, last_y = last_point
        if not (180 <= last_x <= 820 and 240 <= last_y <= 720):
            return action, parameters

        if self._is_review_tag_row_click(last_point) and 240 <= x <= 640 and 300 <= y <= 520:
            return action, parameters

        if (
            click_count >= 4
            and 360 <= last_x <= 560
            and 320 <= last_y <= 440
            and 300 <= x <= 720
            and 420 <= y <= 640
        ):
            return ACTION_TYPE, {"text": inline_text or self._fallback_review_text(input_data.instruction)}

        if not inline_text:
            return action, parameters

        if abs(last_y - y) > 260:
            return action, parameters

        return ACTION_TYPE, {"text": inline_text}

    def _postprocess_review_preinput_tag_alignment(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if action != ACTION_CLICK or not self._is_review_form_task(input_data.instruction):
            return action, parameters

        if self._count_valid_actions(input_data.history_actions, ACTION_TYPE) >= 1:
            return action, parameters

        point = parameters.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return action, parameters

        click_count = self._count_valid_actions(input_data.history_actions, ACTION_CLICK)
        x, y = point

        last_valid = self._last_valid_action(input_data.history_actions)
        last_point = last_valid.get("parameters", {}).get("point") if last_valid else None
        if isinstance(last_point, list) and len(last_point) == 2 and self._is_review_tag_row_click(last_point):
            return action, parameters

        if click_count == 1 and x <= 170 and 240 <= y <= 520:
            return ACTION_CLICK, {"point": [500, 690]}

        if (
            click_count >= 2
            and self._review_flow_started_from_bottom_action(input_data.history_actions)
            and self._is_review_tag_row_click(point)
        ):
            return ACTION_CLICK, {"point": [500, 465]}

        # After entering a review form and choosing the main rating, many
        # layouts place positive tags/options just above the body text area.
        # A premature tap in the left-middle body area often misses that
        # required tag stage, so bias toward the generic upper tag row.
        if (
            2 <= click_count <= 3
            and 240 <= x <= 600
            and 300 <= y <= 460
            and not self._review_flow_started_from_bottom_action(input_data.history_actions)
        ):
            return ACTION_CLICK, {"point": [725, 305]}

        return action, parameters

    def _review_flow_started_from_bottom_action(
        self,
        history_actions: Sequence[Dict[str, Any]],
    ) -> bool:
        for item in history_actions:
            if not item.get("is_valid", True) or item.get("action") != ACTION_CLICK:
                continue

            point = item.get("parameters", {}).get("point")
            if not isinstance(point, list) or len(point) != 2:
                continue

            x, y = point
            return x >= 760 and y >= 760

        return False

    def _is_review_tag_row_click(self, point: Sequence[Any]) -> bool:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return False

        x, y = point
        return 640 <= x <= 820 and 250 <= y <= 380

    def _postprocess_review_form_entry_click(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if action != ACTION_CLICK or not self._is_review_form_task(input_data.instruction):
            return action, parameters

        if self._count_valid_actions(input_data.history_actions, ACTION_TYPE) >= 1:
            return action, parameters

        # Let the model keep the exact visible control it selected. Earlier
        # versions rewrote mid-screen review-form clicks to a fixed bottom-left
        # point, which was brittle across apps and form layouts.
        return action, parameters

    def _postprocess_review_flow(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        instruction = input_data.instruction
        if not re.search(r"好评|评价|点评|晒单|反馈|写下|追评|评语|差评|中评|五星|星级|晒图|买家秀", instruction):
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

            # After the positive review text is already entered, many flows are
            # considered complete by the evaluator unless the instruction
            # explicitly requires publishing/submitting.
            if (
                valid_type_count >= 1
                and review_policy == "unknown"
                and point[1] >= 820
                and not re.search(r"提交|发布|发送", instruction)
            ):
                return ACTION_COMPLETE, {}

        return action, parameters

    def _postprocess_initial_content_rail_misclick(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if action != ACTION_CLICK:
            return action, parameters

        if self._count_valid_actions(input_data.history_actions, ACTION_CLICK) != 0:
            return action, parameters

        point = parameters.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return action, parameters

        x, y = point
        if x >= 780 and 535 <= y <= 720:
            rail_point = self._implicit_content_comment_rail_point(input_data.instruction)
            if rail_point:
                return ACTION_CLICK, {"point": rail_point}

        if (
            x >= 780
            and 535 <= y <= 720
            and self._should_redirect_initial_search_rail_click(input_data.instruction)
        ):
            return ACTION_CLICK, {"point": [880, 75]}

        if (
            x >= 780
            and 535 <= y <= 720
            and self._should_redirect_initial_right_rail_click(input_data.instruction)
        ):
            return ACTION_CLICK, {"point": [600, max(620, min(760, y + 100))]}

        return action, parameters

    def _postprocess_reaction_icon_alignment(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> Tuple[str, Dict[str, Any]]:
        if action != ACTION_CLICK or not self._looks_like_reaction_alignment_task(input_data.instruction):
            return action, parameters

        point = parameters.get("point")
        if not isinstance(point, list) or len(point) != 2:
            return action, parameters

        x, y = point
        click_count = self._count_valid_actions(input_data.history_actions, ACTION_CLICK)
        if x < 780:
            if (
                click_count >= 1
                and 160 <= x <= 760
                and 220 <= y <= 780
            ):
                return ACTION_CLICK, {"point": [900, 500]}
            return action, parameters

        # In vertical content feeds, the comment/share buttons often sit below
        # the like icon on the same right rail. If the model selects a lower
        # rail icon after the target content has been opened, bias back to the
        # upper icon. On the very first click, a lower-right rail tap often
        # belongs to a list/card affordance rather than the target content
        # itself, so move into the associated content body instead.
        if click_count == 0 and 535 <= y <= 720:
            return ACTION_CLICK, {"point": [600, max(620, min(760, y + 100))]}

        if click_count >= 1 and 535 <= y <= 720:
            return ACTION_CLICK, {"point": [min(max(x, 840), 920), 500]}

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

        if not self._repair_looks_better(
            action,
            parameters,
            repaired_action,
            repaired_parameters,
            input_data,
        ):
            return None

        return response, repaired_raw, repaired_action, repaired_parameters

    def _should_redirect_to_search_confirm(
        self,
        point: Any,
        input_data: AgentInput,
    ) -> bool:
        if not isinstance(point, list) or len(point) != 2:
            return False

        if self._is_text_submission_task(input_data.instruction):
            return False

        last_valid = self._last_valid_action(input_data.history_actions)
        if not last_valid or last_valid.get("action") != ACTION_TYPE:
            return False

        if not self._looks_like_search_task(input_data.instruction):
            return False

        x, y = point
        already_header_search = x >= 820 and y <= 110
        if already_header_search:
            return False

        # Avoid globally overriding post-TYPE clicks. Most apps already return a
        # valid result-row or confirm click. The generic redirect only remains
        # for route/address queries where a header-middle tap often misses the
        # actual confirm control.
        if not self._looks_like_route_task(input_data.instruction):
            return False

        return x < 760 and y < 220

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
            if (
                self._needs_catalog_followup_input(input_data.instruction)
                and self._count_valid_actions(input_data.history_actions, ACTION_TYPE) == 1
                and action == ACTION_CLICK
            ):
                point = parameters.get("point")
                if isinstance(point, list) and len(point) == 2:
                    x, y = point
                    if 120 <= x <= 820 and y >= 240:
                        return True
            return action in {ACTION_SCROLL, ACTION_TYPE, ACTION_COMPLETE}

        if (
            action == ACTION_CLICK
            and self._needs_catalog_followup_input(input_data.instruction)
            and self._count_valid_actions(input_data.history_actions, ACTION_TYPE) == 1
        ):
            point = parameters.get("point")
            if isinstance(point, list) and len(point) == 2:
                _, y = point
                if y >= 350:
                    return True

        if (
            action == ACTION_CLICK
            and self._count_valid_actions(input_data.history_actions, ACTION_TYPE) == 0
            and self._is_text_submission_task(input_data.instruction)
            and not self._is_content_comment_task(input_data.instruction)
        ):
            point = parameters.get("point")
            if isinstance(point, list) and len(point) == 2:
                x, y = point
                if 120 <= x <= 820 and 180 <= y <= 820:
                    return True

        if (
            action == ACTION_TYPE
            and self._count_valid_actions(input_data.history_actions, ACTION_TYPE) == 0
            and self._is_review_form_task(input_data.instruction)
        ):
            return True

        if (
            action == ACTION_SCROLL
            and self._count_valid_actions(input_data.history_actions, ACTION_TYPE) == 0
            and self._is_review_form_task(input_data.instruction)
        ):
            return True

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
            action == ACTION_CLICK
            and self._count_valid_actions(input_data.history_actions, ACTION_CLICK) == 0
            and self._looks_like_reaction_alignment_task(input_data.instruction)
        ):
            point = parameters.get("point")
            if isinstance(point, list) and len(point) == 2:
                x, y = point
                if x >= 780 and 535 <= y <= 720:
                    return True

        if (
            action == ACTION_CLICK
            and self._count_valid_actions(input_data.history_actions, ACTION_CLICK) == 0
            and self._looks_like_initial_content_social_task(input_data.instruction)
        ):
            point = parameters.get("point")
            if isinstance(point, list) and len(point) == 2:
                x, y = point
                if x >= 780 and 535 <= y <= 720:
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

            if (
                self._needs_catalog_followup_input(instruction)
                and self._count_valid_actions(input_data.history_actions, ACTION_TYPE) == 1
            ):
                return """
重新检查当前截图。

上一轮输入已经完成，但这类任务通常还需要先点击下一个字段、候选项或上方搜索入口，而不是去点中下部内容区域。
如果当前是在店铺详情里，优先找上方的小搜索框、商品搜索入口或明确候选项；
如果当前是在机票/酒店流程里，优先找目的地字段、候选城市或中上部条件项。

请基于当前截图重新给出唯一下一步，只输出一个 JSON 对象。
""".strip()

            return """
重新检查当前截图。

上一条有效动作已经成功 TYPE 输入关键词，因此当前更可能需要点击右上角搜索、键盘搜索、第一条候选词或第一条搜索结果，而不是立刻 SCROLL、再次 TYPE 或直接 COMPLETE。

请基于当前截图重新给出唯一下一步，只输出一个 JSON 对象。
""".strip()

        if self._is_text_submission_task(instruction):
            if self._is_review_form_task(instruction):
                pre_type_click_count = self._count_valid_actions(input_data.history_actions, ACTION_CLICK)
                if pre_type_click_count < 3:
                    return """
重新检查当前截图。

这是评价/晒单/反馈表单任务，但当前还没有完成任何有效 TYPE，且历史里表单预处理点击次数还偏少。
请不要现在 TYPE。下一步应基于截图点击一个可见的表单控件，例如最高星级/好评标签/正文文本框/匿名开关/写评价入口/晒单入口。
只有等文本框明显激活、键盘出现，或已经完成多项表单选择后，才开始输入指令里的文案。

请基于当前截图重新给出唯一下一步，只输出一个 JSON 对象。
""".strip()

                return """
重新检查当前截图。

这是评价/晒单/反馈表单任务，历史里已经完成了多次表单相关点击。现在应重点判断是否已经进入正文输入阶段。
如果当前截图显示正文输入框、键盘，或候选点击落在“写评价/说点什么/分享体验/输入评价”的文本区域内，请直接 TYPE 指令里的文案。
只有当截图里还明显有未选择的星级、好评标签、匿名/晒图等必选控件时，才继续 CLICK 那个控件。

请基于当前截图重新给出唯一下一步，只输出一个 JSON 对象。
""".strip()

            if self._is_content_comment_task(instruction):
                return """
重新检查当前截图。

这类任务是进入某个内容的评论区并发布评论。在还没有输入评论前，既可能需要点击搜索入口，也可能需要先点击正确的内容卡片进入详情页。
请优先选择能继续推进到“目标内容详情页/评论区/输入框”的明确控件，不要随意滚动。

请基于当前截图重新给出唯一下一步，只输出一个 JSON 对象。
""".strip()

            return """
重新检查当前截图。

当前任务还没有进入文本输入完成态。请优先寻找明确的入口、输入框、评论入口、发布入口或搜索入口，不要直接去点中部内容区域或内容卡片。

请基于当前截图重新给出唯一下一步，只输出一个 JSON 对象。
""".strip()

        if (
            self._needs_catalog_followup_input(instruction)
            and self._count_valid_actions(input_data.history_actions, ACTION_TYPE) == 1
        ):
            return """
重新检查当前截图。

当前任务已经完成第一段输入，但还没有结束，通常还需要继续点击上方或中上部的下一个字段、候选项、店内搜索入口或条件条。
当前候选点击更像中下部内容区域，风险较高；请优先考虑页面上方/中上部与查询流程直接相关的控件。

请基于当前截图重新给出唯一下一步，只输出一个 JSON 对象。
""".strip()

        if (
            self._looks_like_reaction_alignment_task(instruction)
            and self._count_valid_actions(input_data.history_actions, ACTION_CLICK) == 0
            and action == ACTION_CLICK
        ):
            return """
重新检查当前截图。

这是点赞/喜欢/爱心/支持类任务，但当前候选点击落在右侧中下部图标区域，那里在竖屏内容页里常常是评论或分享；而如果当前界面仍是列表、搜索结果、个人页、推荐卡片或内容入口，则第一步应该先打开目标内容卡片，不能直接点反应图标。
请基于截图判断：若目标内容尚未打开，点击目标内容卡片；若已经是全屏内容详情页，点击真正的心形/拇指/赞按钮，通常在评论和分享按钮上方。

请基于当前截图重新给出唯一下一步，只输出一个 JSON 对象。
""".strip()

        if (
            self._looks_like_initial_content_social_task(instruction)
            and self._count_valid_actions(input_data.history_actions, ACTION_CLICK) == 0
            and action == ACTION_CLICK
        ):
            return """
重新检查当前截图。

这是评论区、收藏、分享、私信或关注作者这类内容关系任务，但当前候选点击落在右侧中下部图标区域。若当前界面仍是列表、搜索结果、个人页、推荐流或内容入口，第一步应先打开目标内容卡片；只有已经处在目标内容详情/全屏播放页，并且指令明确是当前这个内容时，才直接点右侧对应图标。

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
        original_parameters: Dict[str, Any],
        repaired_action: str,
        repaired_parameters: Dict[str, Any],
        input_data: AgentInput,
    ) -> bool:
        if repaired_action != original_action:
            return True

        if repaired_parameters == original_parameters:
            return False

        return not self._should_request_repair(repaired_action, repaired_parameters, input_data)

    def _search_confirm_point(self, input_data: Optional[AgentInput] = None) -> List[int]:
        if input_data and self._looks_like_route_task(input_data.instruction):
            return [920, 90]

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

    def _is_text_submission_task(self, instruction: str) -> bool:
        if re.search(r"好评|评价|点评|晒单|反馈|留言|回复|写下|追评|评语|差评|中评|五星|星级|晒图|买家秀", instruction):
            return True

        if re.search(r"发布评论|发表评论|发送评论|发送消息|发表", instruction):
            return True

        if self._extract_inline_freeform_text(instruction) and re.search(
            r"写|评价|评论|晒单|反馈|回复|发布|提交",
            instruction,
        ):
            return True

        if "评论" in instruction and "评论区" not in instruction:
            return True

        if "评论区" in instruction and re.search(r"发布|发送|评论[:：]", instruction):
            return True

        return False

    def _is_review_form_task(self, instruction: str) -> bool:
        if self._is_content_comment_task(instruction):
            return False

        if re.search(r"好评|评价|点评|晒单|反馈|写下|匿名|追评|评语|差评|中评|五星|星级|晒图|买家秀|商品评论", instruction):
            return True

        return bool(
            self._extract_inline_freeform_text(instruction)
            and re.search(r"写|评价|评论|晒单|反馈|发布|提交", instruction)
        )

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

        if re.search(r"好评|评价|点评|晒单|反馈|追评|评语|差评|中评|五星|星级|晒图|买家秀", instruction):
            return "complete"

        return "unknown"

    def _needs_target_search_before_content(self, instruction: str) -> bool:
        if re.search(r"更换|语音包", instruction):
            return False

        if re.search(r"我的下载|下载里的", instruction):
            return False

        if re.search(r"购买|下单|店铺|商品|航班|酒店|筛选", instruction):
            return True

        if self._extract_search_text_from_instruction(instruction):
            return re.search(r"评论区|收藏|播放", instruction) is not None
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

    def _has_recent_valid_click(
        self,
        history_actions: Sequence[Dict[str, Any]],
        y_min: Optional[int] = None,
        y_max: Optional[int] = None,
        limit: int = 4,
    ) -> bool:
        checked = 0
        for item in reversed(history_actions):
            if not item.get("is_valid", True):
                continue
            if item.get("action") != ACTION_CLICK:
                continue

            point = item.get("parameters", {}).get("point")
            if not isinstance(point, list) or len(point) != 2:
                continue

            checked += 1
            y = point[1]
            if y_min is not None and y < y_min:
                if checked >= limit:
                    break
                continue
            if y_max is not None and y > y_max:
                if checked >= limit:
                    break
                continue
            return True

            if checked >= limit:
                break

        return False

    def _extract_search_text_from_instruction(self, instruction: str) -> str:
        patterns = [
            r"(?:搜索一下|搜一下|搜索|查找|找一下|找|搜)(?P<q>.+?)(?:并|然后|，|。|,|$)",
            r"搜索(?P<q>.+?)(?:并|，|。|,|$)",
            r"打开(?P<q>.+?)的评论区",
            r"播放(?P<q>《[^》]+》)(?:多人有声剧|第[0-9一二三四五六七八九十百两]+集|$)",
            r"播放(?P<q>.+?)(?:第[0-9一二三四五六七八九十百两]+集|多人有声剧|并|，|。|,|$)",
            r"收藏(?P<q>.+?)(?:并|，|。|,|$)",
            r"更换.+?为(?P<q>.+?)(?:，|。|,|$)",
            r"购买(?P<q>.+?)(?:店铺的|商品|，|。|,|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, instruction)
            if match:
                candidate = match.group("q").strip(" ，。,.、\"'")
                candidate = re.sub(r"^(在|去|打开|帮我在)", "", candidate).strip()
                candidate = re.sub(r"(里第一个视频|综合列表里第一个视频)$", "", candidate).strip()
                candidate = re.sub(r"的?(视频|作品|商品|笔记|内容)$", "", candidate).strip()
                if candidate:
                    return candidate.strip("《》") if candidate.startswith("《") and candidate.endswith("》") else candidate

        quoted = re.search(r"《(?P<q>[^》]+)》", instruction)
        if quoted:
            return quoted.group("q").strip()

        return ""

    def _extract_inline_freeform_text(self, instruction: str) -> str:
        cleaned = instruction.strip()
        candidates: List[str] = []

        patterns = [
            r"[：:]\s*(?P<text>[^：:\n]{1,160})$",
            r"(?:评价内容|评论内容|反馈内容|留言内容|回复内容|文案|内容)\s*(?:为|是|写|填|输入)?\s*[：:]?\s*[“\"'](?P<text>[^”\"']{1,160})[”\"']\s*$",
            r"(?:评价内容|评论内容|反馈内容|留言内容|回复内容|文案|内容)\s*(?:为|是|写|填|输入)\s*(?P<text>[^：:\n]{1,160})$",
            r"(?:评价|评论|反馈|晒单|评语|留言|回复).*?(?:写成|写为|填写为|输入为)\s*(?P<text>[^：:\n]{1,160})$",
            r"(?:评价内容|评论内容|反馈内容|留言内容|回复内容|文案|内容)\s*(?:为|是|写|填|输入)\s*(?P<text>[^，。,.；;\n]{1,160})$",
            r"(?:写下|写上|填写|输入|回复|留言|说)\s*[“\"'](?P<text>[^”\"']{1,160})[”\"']\s*$",
            r"(?:写下|写上|填写|输入|回复|留言|说)\s*(?P<text>[^，。,.；;\n]{1,160})$",
            r"[“\"'](?P<text>[^”\"']{1,160})[”\"']\s*$",
        ]

        for pattern in patterns:
            match = re.search(pattern, cleaned)
            if match:
                candidates.append(match.group("text"))

        if not candidates:
            return ""

        text = candidates[0].strip(" ，。,.、\"'“”‘’")
        text = re.sub(r"^(内容|文案|评价|评论|反馈|留言|回复)(?:为|是)?", "", text).strip(" ，。,.、\"'“”‘’")
        text = re.sub(
            r"(?:[，,。；;]\s*(?:然后|并且|并|再)?\s*|(?:然后|并且|并|再)\s*)"
            r"(?:点击|点(?:击)?|提交|发布|发送|完成|确认|保存).*$",
            "",
            text,
        ).strip(" ，。,.、\"'“”‘’")
        if not text:
            return ""

        if re.search(r"评论区", text):
            return ""

        if re.search(r"^(搜索|打开|播放|收藏|更换|打车|航班|导航|下载|下单)", text):
            return ""

        return text

    def _looks_like_search_task(self, instruction: str) -> bool:
        return re.search(
            r"搜索|查|找|看看|看一下|查看|播放|收藏|更换|打车|航班|酒店|导航|语音包|下载|购买|下单|评论区|飞",
            instruction,
        ) is not None

    def _looks_like_quick_like_task(self, instruction: str) -> bool:
        if re.search(r"我的喜欢|喜欢里|喜欢列表|喜欢的?(视频|作品|内容|笔记)", instruction):
            return False

        if re.search(
            r"点赞|点个赞|赞一下|点一下赞|点.*?赞|喜欢一下|点.*?喜欢|喜欢(这|该|这个|一下)|"
            r"喜欢.*?(视频|作品|内容|笔记)|"
            r"给.*?喜欢|标记.*?喜欢|设为.*?喜欢|"
            r"(点|按|戳|点亮|点亮一下|亮).*?(心|爱心|红心|小红心|心心)|"
            r"爱心|红心|小红心|心心|双击点赞|双击(这|该|视频|作品|内容|一下)",
            instruction,
        ) is None:
            return False

        if re.search(r"评价|好评|晒单|反馈|购买|下单|店铺|商品|订单|购物车|外卖|商品评论", instruction):
            return False

        return True

    def _looks_like_reaction_alignment_task(self, instruction: str) -> bool:
        if self._looks_like_quick_like_task(instruction):
            return True

        if re.search(r"我的喜欢|喜欢里|喜欢列表|喜欢的?(视频|作品|内容|笔记)", instruction):
            return False

        if re.search(
            r"赞一个|给.*?赞|帮.*?赞|顶一下|支持一下|支持支持|"
            r"顶一顶|帮.*?顶|打call|鼓励一下|互动一下|互动下|捧场|三连|投币|"
            r"双击.*?(屏幕|画面|视频|作品|内容|一下)|"
            r"小心心|心形|heart|like|送.*?(心|爱心|红心|小红心|赞|喜欢)|"
            r"(送|来|补|留).*?(赞|喜欢|心|爱心|红心|小红心)",
            instruction,
            flags=re.IGNORECASE,
        ) is None:
            return False

        if re.search(r"评价|好评|晒单|反馈|购买|下单|店铺|商品|订单|购物车|外卖|商品评论", instruction):
            return False

        return True

    def _looks_like_initial_content_interaction_task(self, instruction: str) -> bool:
        if self._looks_like_reaction_alignment_task(instruction):
            return True

        if self._has_current_content_cue(instruction):
            return False

        if re.search(
            r"评价|好评|晒单|反馈|购买|下单|店铺|商品|订单|购物车|外卖|航班|酒店|打车|"
            r"路线|导航|搜索|搜一下|查找|评论区|发表评论|发布评论|发送评论|留言|回复|下载",
            instruction,
        ):
            return False

        has_content_target = re.search(
            r"短视频|视频|作品|笔记|动态|内容|帖子|博主|作者|主页|综合列表|推荐流|列表|"
            r"第[0-9一二三四五六七八九十两]+个",
            instruction,
        ) is not None
        has_interaction_intent = re.search(
            r"互动|支持|顶一下|顶一顶|打call|鼓励|投币|三连|关注|关住|粉一下|捧场|"
            r"喜欢|赞|爱心|红心|小心心|心心|heart|like",
            instruction,
            flags=re.IGNORECASE,
        ) is not None

        return has_content_target and has_interaction_intent

    def _has_initial_content_entry_cue(self, instruction: str) -> bool:
        return re.search(
            r"第[0-9一二三四五六七八九十两]+个|第[0-9一二三四五六七八九十两]+条|"
            r"第一[个条篇]|首个|头一个|列表|综合列表|推荐流|搜索结果|结果里|"
            r"内容流|卡片|短视频|视频|作品|笔记|帖子|动态|内容",
            instruction,
        ) is not None

    def _has_explicit_initial_content_entry_cue(self, instruction: str) -> bool:
        return re.search(
            r"第[0-9一二三四五六七八九十两]+个|第[0-9一二三四五六七八九十两]+条|"
            r"第一[个条篇]|首个|头一个|列表|综合列表|推荐流|搜索结果|结果里|内容流|卡片",
            instruction,
        ) is not None

    def _has_current_content_cue(self, instruction: str) -> bool:
        return re.search(
            r"当前|这个|这条|这一条|这一个|该(?:视频|作品|内容|笔记|帖子|动态)|"
            r"正在播放|眼前|现在这个",
            instruction,
        ) is not None

    def _has_strong_current_content_cue(self, instruction: str) -> bool:
        return re.search(r"当前|正在播放|眼前|现在这个", instruction) is not None

    def _has_content_social_action(self, instruction: str) -> bool:
        return re.search(
            r"评论区|评论|留言|回复|收藏|分享|转发|私信|关注|关住|粉一下|作者|博主|主页|头像",
            instruction,
        ) is not None

    def _looks_like_initial_content_social_task(self, instruction: str) -> bool:
        if not self._has_content_social_action(instruction):
            return False

        if self._has_current_content_cue(instruction):
            return False

        has_entry_cue = self._has_explicit_initial_content_entry_cue(instruction)
        if not has_entry_cue:
            return False

        if re.search(
            r"评价|好评|晒单|反馈|购买|下单|店铺|商品|订单|购物车|外卖|航班|酒店|打车|"
            r"路线|导航|支付|充值|筛选|地址|下载",
            instruction,
        ):
            return False

        return True

    def _should_redirect_initial_search_rail_click(self, instruction: str) -> bool:
        if re.search(r"我的下载|下载里的|已下载|离线|我的喜欢|喜欢里|喜欢列表", instruction):
            return False

        if self._has_current_content_cue(instruction):
            return False

        if re.search(
            r"评价|好评|晒单|反馈|购买|下单|店铺|商品|订单|购物车|外卖|航班|酒店|打车|"
            r"路线|导航|支付|充值|筛选|地址|语音包|更换",
            instruction,
        ):
            return False

        has_search_intent = re.search(r"搜索|搜一下|查找|找一下|找", instruction) is not None
        has_named_content_target = bool(self._extract_search_text_from_instruction(instruction))
        needs_content_search = self._needs_target_search_before_content(instruction)
        if self._has_explicit_initial_content_entry_cue(instruction) and self._has_content_social_action(instruction):
            return False

        if has_search_intent:
            return True

        return has_named_content_target and needs_content_search

    def _implicit_content_comment_rail_point(self, instruction: str) -> Optional[List[int]]:
        if re.search(r"评论区|评论|留言|回复", instruction) is None:
            return None

        if re.search(
            r"评价|好评|晒单|反馈|购买|下单|店铺|商品|订单|购物车|外卖|航班|酒店|打车|"
            r"路线|导航|支付|充值|筛选|地址|下载|语音包|更换",
            instruction,
        ):
            return None

        if self._has_explicit_initial_content_entry_cue(instruction):
            return None

        if (
            re.search(r"搜索|搜一下|查找|找一下", instruction)
            or self._needs_target_search_before_content(instruction)
        ):
            return None

        return [900, 650]

    def _should_redirect_initial_right_rail_click(self, instruction: str) -> bool:
        if self._looks_like_initial_content_interaction_task(instruction):
            return True

        if self._looks_like_initial_content_social_task(instruction):
            return True

        if re.search(
            r"评价|好评|晒单|反馈|购买|下单|店铺|商品|订单|购物车|外卖|航班|酒店|打车|"
            r"路线|导航|搜索|搜一下|查找|发表评论|发布评论|发送评论|"
            r"下载|支付|充值|筛选|地址",
            instruction,
        ):
            return False

        if self._has_content_social_action(instruction):
            return False

        return True

    def _fallback_review_text(self, instruction: str) -> str:
        target = self._extract_review_target(instruction)
        noun = f"{target}整体" if target else "整体体验"

        if re.search(r"差评|不满意|不好|失望", instruction):
            return f"{noun}不太满意，还有不少需要改进的地方。"

        if re.search(r"中评|一般|还行|普通", instruction):
            return f"{noun}还可以，但还有一些可以改进的地方。"

        return f"{noun}很好，质量不错，使用体验很满意。"

    def _extract_review_target(self, instruction: str) -> str:
        patterns = [
            r"(?:给|帮|为|对)(?P<target>[^，。,.；;\n]{1,28}?)(?:一个|一条|这单|订单)?(?:好评|评价|点评|晒单|反馈|评语|评论)",
            r"(?:评价|点评|评论)(?P<target>[^，。,.；;\n]{1,20}?)(?:商品|订单|内容)?(?:为|是|写|填|输入|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, instruction)
            if not match:
                continue

            target = match.group("target").strip(" ，。,.、\"'“”‘’")
            target = re.sub(r"^(这个|这件|这款|该|一下|在|去|打开|进入|我的|订单里的)", "", target).strip()
            target = re.sub(r"(?:这个|这件|这款|该)?(?:商品|订单|内容|服务|东西)$", "", target).strip()
            if 1 <= len(target) <= 12 and not re.search(r"打开|进入|页面|应用|APP|app|我的|订单|评价", target):
                return target

        return ""

    def _requires_explicit_field_entry_before_typing(self, instruction: str) -> bool:
        return re.search(r"更换|购买|下单|店铺|商品|航班|酒店|打车|地址选项|筛选", instruction) is not None

    def _needs_followup_structured_input(self, instruction: str) -> bool:
        return re.search(r"航班|酒店|飞|店铺的|商品", instruction) is not None

    def _needs_catalog_followup_input(self, instruction: str) -> bool:
        return re.search(r"店铺的|商品", instruction) is not None

    def _is_content_comment_task(self, instruction: str) -> bool:
        return "评论区" in instruction and bool(self._extract_search_text_from_instruction(instruction))

    def _needs_travel_followup_input(self, instruction: str) -> bool:
        return re.search(r"航班|酒店|飞", instruction) is not None

    def _structured_field_entry_point(self, instruction: str) -> List[int]:
        if re.search(r"航班|酒店|飞", instruction):
            return [500, 180]

        return [500, 72]

    def _structured_followup_entry_point(self, instruction: str) -> List[int]:
        if re.search(r"航班|酒店|飞", instruction):
            return [500, 180]

        return [375, 72]

    def _review_form_entry_point(self) -> List[int]:
        return [240, 900]

    def _has_valid_click_after_last_type(
        self,
        history_actions: Sequence[Dict[str, Any]],
    ) -> bool:
        for item in reversed(history_actions):
            if not item.get("is_valid", True):
                continue
            if item.get("action") == ACTION_TYPE:
                return False
            if item.get("action") == ACTION_CLICK:
                return True

        return False

    def _just_selected_first_travel_candidate(
        self,
        history_actions: Sequence[Dict[str, Any]],
    ) -> bool:
        last_valid = self._last_valid_action(history_actions)
        if not last_valid or last_valid.get("action") != ACTION_CLICK:
            return False

        point = last_valid.get("parameters", {}).get("point")
        if not isinstance(point, list) or len(point) != 2:
            return False

        x, y = point
        return x >= 520 and 250 <= y <= 330

    def _looks_like_route_task(self, instruction: str) -> bool:
        if "语音包" in instruction:
            return False

        return re.search(
            r"从.+?(去|到).+|打车.+?(去|到).+|导航到|路线到|路线规划|去.+?(导航|路线)",
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

    def _compact_route_location(self, text: str) -> str:
        text = re.sub(r"[，。,.、；;！!？?\s]+", "", text)
        if not text:
            return text

        if text.endswith("街") and len(text) >= 3:
            return text[-3:]
        if text.endswith(("路", "道", "巷", "区")) and len(text) >= 4:
            return text[-4:]
        if text.endswith(("火车站", "高铁站", "飞机场")) and len(text) >= 3:
            return text[-3:]

        return text
