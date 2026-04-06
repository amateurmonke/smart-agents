from __future__ import annotations
from dataclasses import dataclass, field
from openai import OpenAI
from tools import TOOLS, dispatch_tool

PERSONAS: dict[str, str] = {
    "Analytical": (
        "You reason methodically and prioritize logical structure. "
        "You challenge arguments that lack empirical grounding and always explain your reasoning step by step."
    ),
    "Philosophical": (
        "You engage with the conceptual and ethical dimensions of problems. "
        "You draw on theoretical frameworks and examine underlying assumptions before addressing surface claims."
    ),
    "Empiricist": (
        "You ground every claim in observable evidence. "
        "You distrust speculation, demand concrete examples, and are skeptical of theoretical arguments unsupported by data."
    ),
    "Devil's Advocate": (
        "You challenge the prevailing consensus and stress-test every argument. "
        "You surface hidden assumptions and argue the opposing view even when it is uncomfortable."
    ),
}

STANCE_VALUES = ("FOR", "AGAINST", "NEUTRAL")

STANCE_CLASSIFIER_PROMPT = (
    "You are a stance classifier. Given a debate topic and an argument, "
    "output exactly one word representing the stance expressed: FOR, AGAINST, or NEUTRAL."
)


@dataclass
class DebateAgent:
    name: str
    persona: str
    initial_stance: str
    client: OpenAI
    model: str = "gpt-4o-mini"
    use_tools: bool = True

    current_stance: str = field(init=False)
    position_history: list[str] = field(default_factory=list, init=False)
    last_tool_calls: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.current_stance = self.initial_stance

    def _system_prompt(self) -> str:
        persona_desc = PERSONAS.get(self.persona, "")
        return (
            f"You are {self.name}, a participant in a structured debate. "
            f"Your reasoning style is {self.persona}: {persona_desc} "
            f"Your assigned position on the debate topic is: {self.initial_stance}. "
            "Be concise (3 to 5 sentences per turn). Engage directly with what others have said. "
            "If you use web search, briefly cite your source inline. "
            "Do not use em dashes. Do not use filler phrases like 'Certainly' or 'Great point'."
        )

    def respond(self, topic: str, history: list[dict], round_type: str = "rebuttal") -> str:
        self.last_tool_calls = []

        messages = [{"role": "system", "content": self._system_prompt()}]

        if history:
            recent = history[-8:]
            context_lines = "\n\n".join(
                f"{h['agent']} ({h['round_type']}): {h['content']}" for h in recent
            )
            user_msg = (
                f"Debate topic: {topic}\n\n"
                f"Conversation so far:\n{context_lines}\n\n"
                f"Now give your {round_type}."
            )
        else:
            user_msg = f"Debate topic: {topic}\n\nGive your opening statement."

        messages.append({"role": "user", "content": user_msg})

        response_text = self._call_with_tools(messages)
        stance = self._classify_stance(topic, response_text)
        self.position_history.append(stance)
        self.current_stance = stance

        return response_text

    def _call_with_tools(self, messages: list[dict]) -> str:
        kwargs: dict = {"model": self.model, "messages": messages}
        if self.use_tools:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        while message.tool_calls:
            tool_results = []
            for tc in message.tool_calls:
                result = dispatch_tool(tc.function.name, tc.function.arguments)
                self.last_tool_calls.append({
                    "name": tc.function.name,
                    "args": tc.function.arguments,
                    "result_preview": result[:200]
                })
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            messages = messages + [message] + tool_results
            response = self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            message = response.choices[0].message

        return message.content or ""

    def _classify_stance(self, topic: str, response_text: str) -> str:
        result = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": STANCE_CLASSIFIER_PROMPT},
                {
                    "role": "user",
                    "content": f"Topic: {topic}\nArgument: {response_text}\nOutput one word only.",
                },
            ],
            max_tokens=5,
            temperature=0,
        )
        raw = (result.choices[0].message.content or "").strip().upper()
        return raw if raw in STANCE_VALUES else "NEUTRAL"
