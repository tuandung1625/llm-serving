from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from benchmark.metrics import request_score
from benchmark.schemas import ChatMessage, RequestPlan, RequestResult, WorkloadTrace
from benchmark.streaming_client import OpenAIStreamingClient
from benchmark.tokenizer_utils import TokenCounter

LOGGER = logging.getLogger(__name__)


class WorkloadRunner:
    def __init__(
        self,
        *,
        experiment_id: str,
        trace: WorkloadTrace,
        plans: list[RequestPlan],
        client: OpenAIStreamingClient,
        token_counter: TokenCounter,
        server_model_name: str,
    ):
        self.experiment_id = experiment_id
        self.trace = trace
        self.plans = sorted(plans, key=lambda item: item.request_id)
        self.client = client
        self.token_counter = token_counter
        self.server_model_name = server_model_name
        self._validate_plan_count()

    async def run(self) -> list[RequestResult]:
        run_start_s = time.perf_counter()
        grouped = self._plans_by_conversation()
        tasks = [
            asyncio.create_task(self._run_conversation(conversation_id, conversation_plans, run_start_s))
            for conversation_id, conversation_plans in grouped.items()
        ]
        nested = await asyncio.gather(*tasks)
        results = [item for conversation_results in nested for item in conversation_results]
        return sorted(results, key=lambda row: row.request_id)

    async def _run_conversation(
        self,
        conversation_id: int,
        plans: list[RequestPlan],
        run_start_s: float,
    ) -> list[RequestResult]:
        history = self._initial_history(conversation_id)
        results: list[RequestResult] = []
        for plan in plans:
            target_start_s = run_start_s + plan.scheduled_arrival_s
            await asyncio.sleep(max(0.0, target_start_s - time.perf_counter()))
            user_content = self._user_content(conversation_id, plan)
            messages = [*history, {"role": "user", "content": user_content}]
            input_tokens = self.token_counter.count_messages(messages)
            actual_start_offset_s = time.perf_counter() - run_start_s
            measurement = await self.client.stream_chat_completion(
                messages,
                max_tokens=plan.requested_output_tokens,
                request_id=plan.request_id,
            )
            output_tokens = self.token_counter.count_text(measurement.generated_text)
            success = (
                measurement.http_status is not None
                and 200 <= measurement.http_status < 300
                and not measurement.timeout
                and measurement.error_type is None
                and output_tokens > 0
            )
            error_type = measurement.error_type
            error_message = measurement.error_message
            if output_tokens == 0 and error_type is None:
                error_type = "zero_token_response"
                error_message = "Response completed without generated text tokens"
            score = request_score(
                measurement.ttft_ms,
                measurement.tpot_ms,
                output_tokens,
                error=error_type is not None and error_type != "zero_token_response",
                timeout=measurement.timeout,
            )
            result = RequestResult(
                experiment_id=self.experiment_id,
                request_id=plan.request_id,
                conversation_id=conversation_id,
                turn_index=plan.turn_index,
                scheduled_arrival_s=plan.scheduled_arrival_s,
                actual_start_offset_s=actual_start_offset_s,
                http_status=measurement.http_status,
                success=success,
                error_type=error_type,
                error_message=error_message,
                timeout=measurement.timeout,
                ttft_ms=measurement.ttft_ms,
                tpot_ms=measurement.tpot_ms,
                latency_ms=measurement.latency_ms,
                input_token_count=input_tokens,
                output_token_count=output_tokens,
                expected_output_token_count=plan.expected_output_tokens,
                requested_output_token_count=plan.requested_output_tokens,
                stream_chunk_count=measurement.stream_chunk_count,
                non_empty_delta_count=measurement.non_empty_delta_count,
                request_score=score,
                server_model_name=self.server_model_name,
                benchmark_timestamp=datetime.now(timezone.utc).isoformat(),
                generated_text=measurement.generated_text,
            )
            results.append(result)
            history.append({"role": "user", "content": user_content})
            if measurement.generated_text:
                history.append({"role": "assistant", "content": measurement.generated_text})
            LOGGER.info(
                "request_finished",
                extra={
                    "request_id": plan.request_id,
                    "conversation_id": conversation_id,
                    "turn_index": plan.turn_index,
                    "success": success,
                    "ttft_ms": measurement.ttft_ms,
                    "tpot_ms": measurement.tpot_ms,
                    "output_token_count": output_tokens,
                },
            )
        return results

    def _initial_history(self, conversation_id: int) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        shared_prefix = self.trace.shared_system_prefix_text
        if shared_prefix is None:
            shared_prefix = self.token_counter.synthetic_text("shared_system_prefix", self.trace.shared_system_prefix_tokens)
        if shared_prefix:
            messages.append({"role": "system", "content": shared_prefix})
        if not self.trace.add_per_conversation_prefix_to_first_turn:
            conv_prefix = self._conversation_prefix_text(conversation_id)
            if conv_prefix:
                messages.append({"role": "system", "content": conv_prefix})
        return messages

    def _user_content(self, conversation_id: int, plan: RequestPlan) -> str:
        if self.trace.user_turn_texts is not None:
            user_text = self.trace.user_turn_texts[plan.turn_index]
        else:
            user_text = self.token_counter.synthetic_text(
                f"conversation_{conversation_id}_turn_{plan.turn_index}_user",
                plan.new_user_tokens,
            )
        if plan.turn_index == 0 and self.trace.add_per_conversation_prefix_to_first_turn:
            conv_prefix = self._conversation_prefix_text(conversation_id)
            if conv_prefix:
                return f"{conv_prefix}\n\n{user_text}" if user_text else conv_prefix
        return user_text

    def _conversation_prefix_text(self, conversation_id: int) -> str:
        if self.trace.per_conversation_prefix_texts is not None:
            return self.trace.per_conversation_prefix_texts[conversation_id]
        tokens = self.trace.conversation_prefix_tokens(conversation_id)
        return self.token_counter.synthetic_text(f"conversation_{conversation_id}_prefix", tokens)

    def _plans_by_conversation(self) -> dict[int, list[RequestPlan]]:
        grouped: dict[int, list[RequestPlan]] = {}
        for plan in self.plans:
            grouped.setdefault(plan.conversation_id, []).append(plan)
        for conversation_plans in grouped.values():
            conversation_plans.sort(key=lambda item: item.turn_index)
        return grouped

    def _validate_plan_count(self) -> None:
        if len(self.plans) != self.trace.total_request:
            raise ValueError(f"Received {len(self.plans)} plans, expected {self.trace.total_request}")

