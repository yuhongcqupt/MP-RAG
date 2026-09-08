from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from tqdm import tqdm

from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_generator, get_retriever


@dataclass
class ContextTreeNode:
    """Node used by the PruneRAG query tree."""

    query: str
    parent: Optional["ContextTreeNode"] = None
    query_answer: str = ""
    answer_again: str = ""
    subqueries: List[str] = field(default_factory=list)
    context: List[Tuple[str, str]] = field(default_factory=list)
    node_type: str = "node"  # node / answer / entity / error
    children: List["ContextTreeNode"] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.depth = self.parent.depth + 1 if self.parent else 0


class FlashRAGPruneRAG(BasicPipeline):
    """
    FlashRAG migration of the official PruneRAG pipeline for Llama-3/3.1-8B-Instruct.

    Reproduction choices in this version:
        - topk defaults to 3, as requested for the user's unified experiments.
        - max_depth=3, all_decom_depth=0, threshold=0.95 by default.
        - Llama prompt templates are copied from official PruneRAG scripts/prompts.py.
        - Chat prompt rendering uses tokenizer.apply_chat_template(..., add_generation_prompt=True),
          matching official tree_pipeline.py.
        - Answer confidence is computed from generation-time token logprobs exactly in the
          official style: match the generated answer string in the full JSON output, average
          the matched generated-token logprobs, then exp(avg_logprob).
        - No non-token fallback, no self-reported confidence, no cal_gen_probs rescoring,
          and no whole-output probability fallback are used.

    Config keys commonly used:
        topk / retrieval_topk: per-query retrieval document number, default 3
        max_depth: maximum query tree depth, default 3
        all_decom_depth: official run_prunerag.sh uses 0, default 0
        threshold: confidence pruning threshold, default 0.95
        strict_token_confidence: default True; should remain True for paper reproduction
        dataset_name: used only to select official multi-choice final prompt when needed
        log_dir, progress_path, pred_path: resume/logging paths
    """

    def __init__(
        self,
        config: Any,
        prompt_template: Any = None,
        generator: Any = None,
        retriever: Any = None,
        topk: Optional[int] = None,
        max_depth: Optional[int] = None,
        all_decom_depth: Optional[int] = None,
        threshold: Optional[float] = None,
        strict_token_confidence: Optional[bool] = None,
    ):
        super().__init__(config, prompt_template)
        self.config = config
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever

        # Keep default top-k = 3 for all experiments, as requested.
        self.topk = int(topk if topk is not None else self._cfg("topk", self._cfg("retrieval_topk", 3)))
        self.max_depth = int(max_depth if max_depth is not None else self._cfg("max_depth", 3))
        self.all_decom_depth = int(
            all_decom_depth if all_decom_depth is not None else self._cfg("all_decom_depth", 0)
        )
        self.threshold = float(threshold if threshold is not None else self._cfg("threshold", 0.95))
        self.strict_token_confidence = bool(
            strict_token_confidence
            if strict_token_confidence is not None
            else self._cfg("strict_token_confidence", True)
        )

        # Official tree_pipeline.py sets max_tokens=4096 for llama-3/3.1-8b-instruct.
        self.max_generation_tokens = int(self._cfg("max_tokens", self._cfg("generation_max_tokens", 4096)))
        self.temperature = 0.0  # official SamplingParams uses temperature=0 for PruneRAG
        self.logprobs_size = int(self._cfg("logprobs_size", 100))  # official max_logprobs/logprobs = 100

        self.dataset_name = str(self._cfg("dataset_name", self._cfg("dataset", ""))).lower()
        self.multi_choice_datasets = {"gpqa", "math500", "aime", "amc", "livecode"}

        # For prompt strictness, do not silently fall back to a different FlashRAG template.
        self.require_official_chat_template = bool(self._cfg("require_official_chat_template", True))

        # Logs + resume, consistent with the uploaded FlashRAG-style pipeline.
        self.log_dir = str(self._cfg("log_dir", "qa_logs_prunerag_HotpotQA"))
        self.qa_prefix = str(self._cfg("qa_prefix", "question_result"))
        self.qa_file = None
        self.progress_path = str(self._cfg("progress_path", "progress_prunerag_HotpotQA.txt"))
        self.pred_path = str(self._cfg("pred_path", "pred_results_prunerag_HotpotQA.json"))

        self.retrieval_num = 0

    # ---------------------------------------------------------------------
    # Main FlashRAG pipeline entry
    # ---------------------------------------------------------------------

    def run(self, dataset: Iterable[Any], do_eval: bool = True):
        os.makedirs(self.log_dir, exist_ok=True)

        existing_preds: Dict[str, str] = {}
        if os.path.exists(self.pred_path):
            with open(self.pred_path, "r", encoding="utf-8") as f:
                existing_preds = json.load(f)
        print(f"Loaded {len(existing_preds)} existing predictions")

        # Backfill existing predictions for FlashRAG evaluation.
        for idx, item in enumerate(dataset):
            item_id = self._get_item_id(item, idx)
            if item_id in existing_preds:
                self._update_item_pred(item, existing_preds[item_id])

        processed_ids = set()
        if os.path.exists(self.progress_path):
            with open(self.progress_path, "r", encoding="utf-8") as f:
                processed_ids = {line.strip() for line in f if line.strip()}
        print(f"Loaded from {self.progress_path} loaded {len(processed_ids)} completed questions")

        global_processed = len(processed_ids)

        for idx, item in tqdm(enumerate(dataset), desc="Inference: "):
            item_id = self._get_item_id(item, idx)
            if item_id in processed_ids:
                print(f"Skipping processed question ID: {item_id}")
                continue

            global_processed += 1
            self._maybe_open_log_file(global_processed)

            question = self._get_question(item)
            current_log: List[str] = []
            self._append_log_section(current_log, "Question ID", item_id)
            self._append_log_section(current_log, "Original Question", question)

            try:
                # Official PruneRAG has an artificial ROOT node, so actual questions start at depth=1.
                # This makes all_decom_depth=0 behave exactly like official run_prunerag.sh:
                # the first processed question node uses get_subqueries_llama3_8b(), not the first-only prompt.
                virtual_root = ContextTreeNode("ROOT")
                root = ContextTreeNode(question, parent=virtual_root)
                root.context = self._retrieve_context(question)
                self._append_log_section(current_log, "Root Node Retrieval Results", self._format_context_for_log(root.context))

                self._build_query_tree(root, current_log)

                inference_tree = self._merge_tree(root)
                self._append_log_section(current_log, "PruneRAG Inference Tree", inference_tree)

                final_raw = self._generate_final_answer(question, inference_tree)
                final_answer = self._extract_final_answer(final_raw)
                self._append_log_section(current_log, "Final Generated Text", final_raw)
                self._append_log_section(current_log, "Final Answer", final_answer)

            except Exception as exc:
                final_answer = "unknown"
                self._append_log_section(current_log, "Runtime Exception", repr(exc))
                if self.strict_token_confidence and "token" in str(exc).lower():
                    if self.qa_file:
                        self.qa_file.write("".join(current_log))
                        self.qa_file.flush()
                    raise

            self._update_item_pred(item, final_answer)

            if self.qa_file:
                self.qa_file.write("".join(current_log))
                self.qa_file.flush()

            existing_preds[item_id] = final_answer
            with open(self.pred_path, "w", encoding="utf-8") as f:
                json.dump(existing_preds, f, ensure_ascii=False, indent=2)

            with open(self.progress_path, "a", encoding="utf-8") as f:
                f.write(f"{item_id}\n")
            processed_ids.add(item_id)
            print(f"Question {item_id} processing completed，final answer: {final_answer}")

        if self.qa_file:
            self.qa_file.write("\n========== Log End ==========\n")
            self.qa_file.close()
            self.qa_file = None

        if hasattr(self, "evaluate"):
            dataset = self.evaluate(dataset, do_eval=do_eval)
        return dataset

    # ---------------------------------------------------------------------
    # Core PruneRAG tree construction
    # ---------------------------------------------------------------------

    def _build_query_tree(self, root: ContextTreeNode, log_list: List[str]) -> None:
        """Breadth-first query tree construction with confidence-guided pruning."""
        node_queue: List[ContextTreeNode] = [root]
        current_depth = root.depth

        # Official loop: current_depth starts from 1 and runs while current_depth < max_depth.
        while current_depth < self.max_depth and node_queue:
            current_level_nodes = node_queue
            node_queue = []

            for node in current_level_nodes:
                action = self._generate_node_action(node, log_list)
                action_type = action.get("type", "error")

                if action_type == "decomposition":
                    node.subqueries = [q for q in action.get("subqueries", []) if q]
                    for subq in node.subqueries:
                        child = ContextTreeNode(subq, parent=node)
                        child.context = self._retrieve_context(subq)
                        node.children.append(child)
                        node_queue.append(child)
                    self._append_log_section(
                        log_list,
                        f"Depth {node.depth} Decomposition Node",
                        f"Query: {node.query}\nSubqueries: {node.subqueries}\n",
                    )

                elif action_type == "answer":
                    answer = str(action.get("answer", "")).strip()
                    confidence = float(action.get("confidence", 0.0))
                    node.query_answer = answer
                    node.node_type = "answer"

                    # Official tree_pipeline.py uses confidence > threshold, not >=.
                    if confidence > self.threshold:
                        node.subqueries = []
                        self._append_log_section(
                            log_list,
                            f"Depth {node.depth} Accepted Answer",
                            f"Query: {node.query}\nAnswer: {answer}\nConfidence: {confidence:.6f}\nThreshold: {self.threshold}\n",
                        )
                    else:
                        node.answer_again = (
                            "In the last round, the action you chose was to answer directly, "
                            "but the confidence of your answer is very low, so please rethink your action."
                        )
                        node_queue.append(node)
                        self._append_log_section(
                            log_list,
                            f"Depth {node.depth} Low-confidence Answer，Reprocess",
                            f"Query: {node.query}\nAnswer: {answer}\nConfidence: {confidence:.6f}\nThreshold: {self.threshold}\n",
                        )

                elif action_type == "entity":
                    node.subqueries = [e for e in action.get("entities", []) if e]
                    # Official behavior: create entity children, do not enqueue them for further generation.
                    for ent in node.subqueries:
                        child = ContextTreeNode(ent, parent=node)
                        child.node_type = "entity"
                        child.context = self._retrieve_context(ent)
                        node.children.append(child)
                    self._append_log_section(
                        log_list,
                        f"Depth {node.depth} Entity Retrieval Node",
                        f"Query: {node.query}\nEntities: {node.subqueries}\n",
                    )

                else:
                    node.node_type = "error"
                    self._append_log_section(
                        log_list,
                        f"Depth {node.depth} Parsing Failed",
                        f"Query: {node.query}\nRaw action: {json.dumps(action, ensure_ascii=False)}\n",
                    )

            current_depth += 1

    def _generate_node_action(self, node: ContextTreeNode, log_list: List[str]) -> Dict[str, Any]:
        """Generate and parse the official Llama PruneRAG node action JSON."""
        context = self._format_context_for_prompt(node.context)
        parent_query = node.parent.query if node.parent else node.query

        # Official condition: if nodes[0].depth > all_decom_depth use normal action prompt,
        # else use get_subqueries_llama3_8b_first().
        if node.depth > self.all_decom_depth:
            prompt_body = self._llama_action_prompt().format(
                query=node.query,
                parent_query=parent_query,
                context=context,
            ) + node.answer_again
        else:
            prompt_body = self._llama_first_decomposition_prompt().format(
                query=node.query,
                context=context,
            )

        prompt = self._render_official_user_prompt(prompt_body)
        generated_text, token_logprobs, token_texts = self._generate_with_official_token_logprobs(prompt)
        action = self._parse_action_json_official(generated_text)

        if action.get("type") == "answer":
            answer = str(action.get("answer", "")).strip()
            confidence = self._answer_confidence_official(
                generated_text=generated_text,
                answer=answer,
                token_logprobs=token_logprobs,
                token_texts=token_texts,
            )
            action["confidence"] = confidence

        self._append_log_section(
            log_list,
            f"Node Action Generation Depth {node.depth}",
            f"Query: {node.query}\nPrompt:\n{prompt}\n\nGenerated:\n{generated_text}\n\nParsed:\n{json.dumps(action, ensure_ascii=False, indent=2)}\n",
        )
        return action

    # ---------------------------------------------------------------------
    # Official-style token logprob confidence
    # ---------------------------------------------------------------------

    def _generate_with_official_token_logprobs(self, prompt: str) -> Tuple[str, List[float], List[str]]:
        """
        Generate text and require generation-time token logprobs.

        Official PruneRAG uses vLLM SamplingParams(logprobs=100), then extracts the rank=1
        generated token's decoded_token and logprob for each generated position. This method
        first tries to call an exposed vLLM llm object directly; otherwise it attempts common
        FlashRAG generator return formats. If no exact generated-token logprobs are available,
        strict mode raises an error instead of using an approximation.
        """
        direct = self._try_vllm_direct_generate(prompt, with_logprobs=True)
        if direct is not None:
            return direct

        generate_kwargs_candidates = [
            {
                "return_dict": True,
                "logprobs": self.logprobs_size,
                "max_tokens": self.max_generation_tokens,
                "temperature": 0,
            },
            {
                "return_dict": True,
                "return_scores": True,
                "logprobs": self.logprobs_size,
                "max_tokens": self.max_generation_tokens,
                "temperature": 0,
            },
            {
                "return_scores": True,
                "logprobs": self.logprobs_size,
                "max_tokens": self.max_generation_tokens,
                "temperature": 0,
            },
            {
                "logprobs": self.logprobs_size,
                "max_tokens": self.max_generation_tokens,
                "temperature": 0,
            },
            {"return_dict": True, "return_scores": True},
            {"return_dict": True},
            {"return_scores": True},
        ]

        last_text = ""
        for kwargs in generate_kwargs_candidates:
            try:
                result = self.generator.generate(prompt, **kwargs)
            except TypeError:
                continue
            except Exception:
                continue

            text, logprobs, token_texts = self._parse_generate_result_for_logprobs(result)
            last_text = text or last_text
            if text and logprobs and token_texts:
                return text.strip(), logprobs, token_texts

        message = (
            "Token logprobs are required for strict PruneRAG confidence pruning, but the current "
            "FlashRAG generator did not expose generation-time token logprobs. Use a vLLM-backed "
            "generator that returns output.outputs[0].logprobs, or modify the FlashRAG generator "
            "to return generated token texts plus generated token logprobs."
        )
        if self.strict_token_confidence:
            raise RuntimeError(message)
        raise RuntimeError(message + f" Last generated text: {last_text[:200]}")

    def _try_vllm_direct_generate(self, prompt: str, with_logprobs: bool) -> Optional[Tuple[str, List[float], List[str]]]:
        llm = getattr(self.generator, "llm", None) or getattr(self.generator, "model", None)
        if llm is None or not hasattr(llm, "generate"):
            return None

        try:
            from vllm import SamplingParams
        except Exception:
            return None

        try:
            if with_logprobs:
                params = SamplingParams(
                    max_tokens=self.max_generation_tokens,
                    temperature=0,
                    logprobs=self.logprobs_size,
                )
            else:
                params = SamplingParams(
                    max_tokens=self.max_generation_tokens,
                    temperature=0,
                )
            outputs = llm.generate([prompt], params)
            out = outputs[0].outputs[0]
            text = str(out.text).strip()
            if not with_logprobs:
                return text, [], []
            token_texts, token_logprobs = self._extract_actual_tokens_from_vllm_logprobs(out.logprobs)
            if text and token_texts and token_logprobs:
                return text, token_logprobs, token_texts
        except Exception:
            return None
        return None

    def _parse_generate_result_for_logprobs(self, result: Any) -> Tuple[str, List[float], List[str]]:
        text = ""
        raw_logprobs: Any = None
        token_logprobs: Any = None
        token_probs: Any = None
        token_ids: Any = None
        token_texts: Any = None

        if isinstance(result, dict):
            text = self._first_text(
                self._pick_first_present(result, ["responses", "response", "text", "outputs", "output"])
            )
            raw_logprobs = self._pick_first_present(
                result, ["logprobs", "generated_logprobs", "output_logprobs", "vllm_logprobs"]
            )
            token_logprobs = self._pick_first_present(
                result, ["token_logprobs", "generated_token_logprobs", "generated_log_probs", "log_probs"]
            )
            token_probs = self._pick_first_present(
                result, ["token_probs", "generated_token_probs", "generated_probs", "probs"]
            )
            token_texts = self._pick_first_present(result, ["generated_tokens", "tokens", "token_texts"])
            token_ids = self._pick_first_present(result, ["generated_token_ids", "token_ids"])

        elif isinstance(result, tuple):
            if len(result) >= 1:
                text = self._first_text(result[0])
            if len(result) >= 2:
                # Only accept this if it can be interpreted as selected-token logprobs/probs.
                token_logprobs = result[1]
            if len(result) >= 3:
                token_ids = result[2]
            if len(result) >= 4:
                token_texts = result[3]

        elif isinstance(result, list):
            text = self._first_text(result)
        else:
            text = str(result)

        if raw_logprobs is not None:
            try:
                extracted_texts, extracted_logprobs = self._extract_actual_tokens_from_vllm_logprobs(raw_logprobs)
                if extracted_texts and extracted_logprobs:
                    return text, extracted_logprobs, extracted_texts
            except Exception:
                pass

        if token_texts is not None and token_texts and isinstance(token_texts, list) and isinstance(token_texts[0], list):
            token_texts = token_texts[0]
        if token_texts is None and token_ids is not None:
            token_texts = self._decode_token_ids(token_ids)

        logprob_list = self._normalize_logprob_or_prob_list(token_logprobs)
        if not logprob_list and token_probs is not None:
            prob_list = self._to_float_list(token_probs)
            logprob_list = [math.log(max(float(p), 1e-12)) for p in prob_list]

        if token_texts is not None:
            token_texts = [str(x) for x in token_texts]
        else:
            token_texts = []

        return text, logprob_list, token_texts

    def _extract_actual_tokens_from_vllm_logprobs(self, vllm_logprobs: Any) -> Tuple[List[str], List[float]]:
        """Replicate official get_logprobs_for_matched_string preprocessing for vLLM logprobs."""
        if vllm_logprobs is None:
            return [], []

        # Single-item batch wrappers sometimes add one extra list level.
        if isinstance(vllm_logprobs, list) and len(vllm_logprobs) == 1 and isinstance(vllm_logprobs[0], list):
            vllm_logprobs = vllm_logprobs[0]

        token_texts: List[str] = []
        token_logprobs: List[float] = []

        for token_pos_logprobs_dict in vllm_logprobs:
            actual_token_logprob_obj = None

            if isinstance(token_pos_logprobs_dict, dict):
                # Official behavior: find the Logprob object whose rank == 1.
                for logprob_val in token_pos_logprobs_dict.values():
                    if self._get_attr_or_key(logprob_val, "rank") == 1:
                        actual_token_logprob_obj = logprob_val
                        break

                # Some wrappers return the selected token directly as a dict.
                if actual_token_logprob_obj is None and (
                    "decoded_token" in token_pos_logprobs_dict or "token" in token_pos_logprobs_dict
                ):
                    actual_token_logprob_obj = token_pos_logprobs_dict
            else:
                # Some wrappers return a Logprob-like object directly.
                if self._get_attr_or_key(token_pos_logprobs_dict, "rank") == 1:
                    actual_token_logprob_obj = token_pos_logprobs_dict

            if actual_token_logprob_obj is None:
                continue

            decoded_token = (
                self._get_attr_or_key(actual_token_logprob_obj, "decoded_token")
                or self._get_attr_or_key(actual_token_logprob_obj, "token")
                or self._get_attr_or_key(actual_token_logprob_obj, "text")
            )
            logprob = self._get_attr_or_key(actual_token_logprob_obj, "logprob")
            if logprob is None:
                logprob = self._get_attr_or_key(actual_token_logprob_obj, "log_prob")

            if decoded_token is not None and logprob is not None:
                token_texts.append(str(decoded_token))
                token_logprobs.append(float(logprob))

        return token_texts, token_logprobs

    @staticmethod
    def _get_attr_or_key(obj: Any, name: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    def _answer_confidence_official(
        self,
        generated_text: str,
        answer: str,
        token_logprobs: Sequence[float],
        token_texts: Sequence[str],
    ) -> float:
        """
        Official confidence computation:
            1. Find target answer string in the generated JSON text.
            2. Collect generated tokens whose character span overlaps the answer span.
            3. Average their logprobs.
            4. confidence = e ** average_logprob.

        If the answer string is absent in the generated text, official code sets confidence to 0.
        If token logprobs are unavailable, this strict implementation raises an error.
        """
        if not answer:
            return 0.0
        if not token_logprobs or not token_texts:
            raise RuntimeError("Token logprobs are required for strict PruneRAG confidence pruning.")

        matches = list(re.finditer(re.escape(answer), generated_text))
        if not matches:
            return 0.0

        match = matches[0]
        start_char_idx = match.start()
        end_char_idx = match.end()

        matched_logprobs: List[float] = []
        current_char_offset = 0
        start_token_idx = -1
        end_token_idx = -1

        for i, (token_text, logprob) in enumerate(zip(token_texts, token_logprobs)):
            token_length = len(str(token_text))
            if token_length == 0:
                continue

            if max(current_char_offset, start_char_idx) < min(current_char_offset + token_length, end_char_idx):
                if start_token_idx == -1:
                    start_token_idx = i
                end_token_idx = i
            elif start_token_idx != -1:
                break

            current_char_offset += token_length

        if start_token_idx != -1 and end_token_idx != -1:
            matched_logprobs = [float(x) for x in token_logprobs[start_token_idx : end_token_idx + 1]]

        if not matched_logprobs:
            return 0.0

        average_logprob = sum(matched_logprobs) / len(matched_logprobs)
        return float(math.e ** average_logprob)

    @staticmethod
    def _normalize_logprob_or_prob_list(values: Any) -> List[float]:
        vals = FlashRAGPruneRAG._to_float_list_static(values)
        if not vals:
            return []
        # vLLM/HF logprobs are <= 0. If a wrapper returns selected token probabilities in [0, 1],
        # convert them to logprobs so the official exp(avg_logprob) computation is equivalent.
        if all(0.0 <= v <= 1.0 for v in vals):
            return [math.log(max(v, 1e-12)) for v in vals]
        return vals

    def _to_float_list(self, obj: Any) -> List[float]:
        return self._to_float_list_static(obj)

    @staticmethod
    def _to_float_list_static(obj: Any) -> List[float]:
        if obj is None:
            return []
        if hasattr(obj, "detach"):
            obj = obj.detach().cpu().tolist()
        elif hasattr(obj, "cpu") and hasattr(obj, "tolist"):
            obj = obj.cpu().tolist()
        elif hasattr(obj, "tolist"):
            obj = obj.tolist()

        if isinstance(obj, (int, float)):
            return [float(obj)]
        if isinstance(obj, tuple):
            obj = list(obj)
        if isinstance(obj, list):
            if len(obj) == 1 and isinstance(obj[0], list):
                obj = obj[0]
            flat: List[float] = []
            for x in obj:
                if isinstance(x, (list, tuple)) or hasattr(x, "tolist"):
                    flat.extend(FlashRAGPruneRAG._to_float_list_static(x))
                else:
                    try:
                        flat.append(float(x))
                    except Exception:
                        continue
            return flat
        return []

    def _decode_token_ids(self, token_ids: Any) -> Optional[List[str]]:
        ids = token_ids
        if hasattr(ids, "detach"):
            ids = ids.detach().cpu().tolist()
        elif hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        if not isinstance(ids, list):
            return None

        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return None

        tokens: List[str] = []
        for tid in ids:
            try:
                if hasattr(tokenizer, "decode"):
                    tokens.append(tokenizer.decode([int(tid)], skip_special_tokens=False))
                elif hasattr(tokenizer, "convert_ids_to_tokens"):
                    tokens.append(str(tokenizer.convert_ids_to_tokens(int(tid))))
            except Exception:
                continue
        return tokens or None

    @staticmethod
    def _pick_first_present(mapping: Dict[str, Any], keys: Sequence[str]) -> Any:
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
        return None

    @staticmethod
    def _first_text(obj: Any) -> str:
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, (list, tuple)):
            if not obj:
                return ""
            first = obj[0]
            if isinstance(first, dict):
                return str(first.get("text") or first.get("response") or first.get("output") or first)
            return str(first)
        return str(obj)

    # ---------------------------------------------------------------------
    # Retrieval and final generation
    # ---------------------------------------------------------------------

    def _retrieve_context(self, query: str) -> List[Tuple[str, str]]:
        docs = self.retriever.search(query, self.topk)
        self.retrieval_num += 1
        context: List[Tuple[str, str]] = []
        for i, doc in enumerate(docs or []):
            if isinstance(doc, dict):
                doc_id = str(doc.get("id", f"doc_{i}"))
                contents = str(doc.get("contents", doc.get("text", "")))
            else:
                doc_id = str(getattr(doc, "id", f"doc_{i}"))
                contents = str(getattr(doc, "contents", getattr(doc, "text", "")))
            context.append((doc_id, contents))
        return context

    @staticmethod
    def _format_context_for_prompt(context: List[Tuple[str, str]]) -> str:
        # Official _generate_subqueries uses: "\n".join(content for _, content in node.context)
        return "\n".join(content for _, content in context)

    @staticmethod
    def _format_context_for_tree(context: List[Tuple[str, str]]) -> str:
        # Official _merge_tree uses [Doc i] labels only in the final inference tree.
        return "\n".join(f"[Doc {i + 1}] {content}" for i, (_doc_id, content) in enumerate(context))

    @staticmethod
    def _format_context_for_log(context: List[Tuple[str, str]]) -> str:
        return "\n".join(f"Doc {i + 1}: {content}" for i, (_doc_id, content) in enumerate(context))

    def _merge_tree(self, root: ContextTreeNode) -> str:
        """Replicate official _merge_tree structure for a single FlashRAG item."""

        def build_tree(node: ContextTreeNode) -> Dict[str, Any]:
            include_context = node.node_type in {"answer", "entity"} or node.depth == self.max_depth
            return {
                "query": node.query,
                "answer": node.query_answer,
                "context": self._format_context_for_tree(node.context) if include_context else "",
                "children": [build_tree(child) for child in node.children],
            }

        return json.dumps(build_tree(root), ensure_ascii=False)

    def _generate_final_answer(self, question: str, inference_tree: str) -> str:
        prompt_body = self._llama_final_answer_prompt(multi_choice=self.dataset_name in self.multi_choice_datasets).format(
            question=question,
            context=inference_tree,
        )
        prompt = self._render_official_user_prompt(prompt_body)

        direct = self._try_vllm_direct_generate(prompt, with_logprobs=False)
        if direct is not None:
            return str(direct[0]).strip()

        for kwargs in [
            {"max_tokens": self.max_generation_tokens, "temperature": 0},
            {"max_new_tokens": self.max_generation_tokens, "temperature": 0},
            {},
        ]:
            try:
                output = self.generator.generate(prompt, **kwargs)
                return self._first_text(output).strip()
            except TypeError:
                continue
        output = self.generator.generate(prompt)
        return self._first_text(output).strip()

    # ---------------------------------------------------------------------
    # Official-style JSON parsing and answer extraction
    # ---------------------------------------------------------------------

    def _parse_action_json_official(self, text: str) -> Dict[str, Any]:
        """Parse action JSON with the same pattern family as official tree_pipeline.py."""
        processed_item: Dict[str, Any] = {"type": "error", "message": "No valid pattern found"}

        decomposition_re = r'\{\s*\"type\"\s*:\s*\"decomposition\".*?\}'
        answer_re = r'\{\s*\"type\"\s*:\s*\"answer\".*?\}'
        entity_re = r'\{\s*\"type\"\s*:\s*\"entity\".*?\}'
        combined_json_pattern = re.compile(
            r'((' + decomposition_re + r')|(' + answer_re + r')|(' + entity_re + r'))',
            re.DOTALL,
        )
        all_matches = combined_json_pattern.findall(text)
        if not all_matches:
            return processed_item

        json_str = all_matches[-1][0]
        if not json_str:
            return processed_item

        try:
            parsed_json = json.loads(json_str)
            json_type = parsed_json.get("type")

            if json_type == "decomposition" and "subquery1" in parsed_json and "subquery2" in parsed_json:
                return {
                    "type": "decomposition",
                    "subqueries": [
                        (parsed_json["subquery1"] or "").strip(),
                        (parsed_json["subquery2"] or "").strip(),
                    ],
                }

            if json_type == "answer" and "answer" in parsed_json:
                return {
                    "type": "answer",
                    "answer": (str(parsed_json["answer"]) or "").strip(),
                }

            if json_type == "entity" and "entity1" in parsed_json and "entity2" in parsed_json:
                return {
                    "type": "entity",
                    "entities": [
                        (str(parsed_json["entity1"]) or "").strip(),
                        (str(parsed_json["entity2"]) or "").strip(),
                    ],
                }

            return {"type": "error", "message": "Last JSON format is invalid or incomplete"}

        except json.JSONDecodeError as exc:
            return {"type": "error", "message": f"Last JSON parsing error: {exc}"}
        except KeyError as exc:
            return {"type": "error", "message": f"Last JSON key missing: {exc}"}
        except Exception as exc:
            return {"type": "error", "message": f"Unknown error processing the last JSON: {exc}"}

    @staticmethod
    def _extract_final_answer(raw_output: str) -> str:
        text = str(raw_output).strip()
        if not text:
            return "unknown"

        boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
        if boxed:
            return boxed[-1].strip()

        patterns = [
            r"\[Final Answer\]\s*(.*?)(?:\n\n|$)",
            r"\*\*Final Answer\*\*\s*(.*?)(?:\n\n|$)",
            r"Final Answer\s*:\s*(.*?)(?:\n\n|$)",
            r"Answer\s*:\s*(.*?)(?:\n\n|$)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
            if matches:
                ans = matches[-1].strip()
                ans = re.sub(r"^```|```$", "", ans).strip()
                if ans:
                    return ans

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else "unknown"

    # ---------------------------------------------------------------------
    # Official Llama 3 8B prompt templates copied from PruneRAG scripts/prompts.py
    # ---------------------------------------------------------------------

    @staticmethod
    def _llama_first_decomposition_prompt() -> str:
        return '''
You are given a query along with its parent question and optional context. Your task is to decompose the query into two logically related sub-queries and judge whether the contexts can help you solve the query using a relevant_score.

```
### Rules:

for relevant_score:
1. Only consider information relevant to the query in the Query_context.
2. **If a context provides direct or indirect evidence to answer the query, assign a higher relevant_score.**
3. The output must be a single json object containing float values.
4. The relevant_score should be a float value between 0 and 1, where 0 means completely irrelevant and 1 means highly relevant.
5. Use Doc_i to represent the i-th context,which means the output json's key should be{{"Doc_i": ..}} for the i-th context but not other specific format.
6. **The number of keys in the output json must be equal to the number of contexts provided.for example, if there are 5 contexts given, the output json must contain 5 key-value pairs.**

for decomposition:
    Ignore information not relevant to the query in the Query_context.
    Ensure subquery1, subquery2 are all string values.
    The output must be a single JSON object inside a markdown code block.
    Do not provide any explanation or commentary outside the JSON.



If the Query cannot be directly answered but can be split into two logically related subqueries that together answer the parent query, respond as:
```json
{{"relevant_score": {{"Doc_1": .., "Doc_2": ..,..."Doc_n":..}},"type": "decomposition", "subquery1": "...", "subquery2": "..."}}

### Example:
Query_context: ...
Query: ...
Output:
```json
{{"relevant_score": {{"Doc_1": .., "Doc_2": ..,..."Doc_n":..}}, "type": "decomposition", "subquery1": "...", "subquery2": "..."}}
```

### Your Task:
Query_context: {context}
Query: {query}
Output:
'''

    @staticmethod
    def _llama_action_prompt() -> str:
        return '''
You are given a query along with its parent question and optional context. Your task is to select the correct action type based on the following rules:
### Available Action Types:
1. **Direct Answer**  
If the `Query_context` contains a clear and verifiable answer to the `Query`, respond as:  
```json
{{"type": "answer", "answer": "..."}}
```
2. **Decomposition**
If the Query cannot be directly answered but can be split into two logically related subqueries that together answer the parent query, respond as:
```json
{{"type": "decomposition", "subquery1": "...", "subquery2": "..."}}
```
3. **Entity Extraction**
If the Query lacks sufficient context to be answered or decomposed, but contains key identifiable entities, extract them as:
```json
{{"type": "entity", "entity1": "...", "entity2": "..."}}
```
### Rules:
Only use "answer" when you are 100% sure the answer is directly supported by the context.
Ignore information not relevant to the query in the Query_context.
Ensure subquery1, subquery2, entity1, entity2, and answer are all string values.
The output must be a single JSON object inside a markdown code block.
Do not provide any explanation or commentary outside the JSON.
### Examples:
**Example 1 (Direct Answer):**
Query_context: Doc 1: Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.
Parent_query of the Query: Which magazine was started first Arthur's Magazine or First for Women?
Query: When was the Arthur's Magazine started?
Output: 
``` json
{{"type": "answer", "answer": "1844"}}
```
**Example 2 (Decomposition):**
Query_context:  Doc 1: A Flame in My Heart is a 1987 French- Swiss drama film directed by Alain Tanner.
Parent_query of the Query: Which film has the director born later, A Flame In My Heart or Butcher, Baker, Nightmare Maker?
Query: What is the birhday of the director of A Flame In My Heart?
Output: 
``` json
{{"type": "decomposition", "subquery1": "Who is Alain Tanner?", "subquery2": "What is the birhday of Alain Tanner?"}}
```
**Example 3 (Entity Extraction):**
Query_context: ...
Parent_query of the Query: Which film has the director born later, A Flame In My Heart or Butcher, Baker, Nightmare Maker?
Query: Who is the director of Butcher, Baker, Nightmare Maker?
Output: 
``` json 
{{"type": "entity", "entity1": "Butcher, Baker, Nightmare Maker", "entity2": "Director of Butcher, Baker, Nightmare Maker"}}
```
### Your Task:
Query_context: {context}
Parent_query of the Query: {parent_query}
Query: {query}
Output:

'''

    @staticmethod
    def _llama_final_answer_prompt(multi_choice: bool = False) -> str:
        if multi_choice:
            return '''
    ### Inference Tree:
    {context}
    Please answer the following multiple-choice question.
    You can use the helpful information in the above Inference Tree.
    Your final answer must be formatted as \\boxed{{YOUR_ANSWER}}.And YOUR_ANSWER should be one of the letters A, B, C, or D, DO NOT include any answer content.
    For example, Question: What is the capital of France?\n(A) Paris \n(B) London \n(C) Berlin \n(D) Dubai \n Answer: \\boxed{{A}}.

    ### Question:
    {question}
    ### Answer:
    '''
        return '''
    ### Inference Tree:
    {context}

    Please answer the following question.
    You can use the helpful information in the above Inference Tree.
    Your final answer must be formatted as \\boxed{{YOUR_ANSWER}}.And YOUR_ANSWER should be a NON-SENTENTIAL answer.
    For example, Question: What is the capital of France? Answer: \\boxed{{Paris}}.

    ### Question:
    {question}

    ### Answer: 
    '''

    # ---------------------------------------------------------------------
    # Utility methods compatible with FlashRAG dataset items
    # ---------------------------------------------------------------------

    def _render_official_user_prompt(self, prompt_body: str) -> str:
        tokenizer = self._get_tokenizer()
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_body}],
                tokenize=False,
                add_generation_prompt=True,
            )

        if self.require_official_chat_template:
            raise RuntimeError(
                "Official Llama prompt rendering requires tokenizer.apply_chat_template(..., "
                "add_generation_prompt=True). The current FlashRAG generator does not expose such tokenizer."
            )

        # Non-strict fallback is kept only for debugging, not for reproduction.
        return prompt_body

    def _get_tokenizer(self) -> Any:
        tokenizer = getattr(self.generator, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        model = getattr(self.generator, "model", None)
        return getattr(model, "tokenizer", None)

    def _cfg(self, key: str, default: Any = None) -> Any:
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    @staticmethod
    def _get_question(item: Any) -> str:
        if hasattr(item, "question"):
            return str(item.question)
        if isinstance(item, dict):
            return str(item.get("question", item.get("Question", item.get("query", ""))))
        return str(getattr(item, "query", ""))

    @staticmethod
    def _get_item_id(item: Any, idx: int) -> str:
        if hasattr(item, "id"):
            return str(item.id)
        if isinstance(item, dict):
            return str(item.get("id", item.get("_id", idx)))
        return str(idx)

    @staticmethod
    def _update_item_pred(item: Any, pred: str) -> None:
        if hasattr(item, "update_output"):
            item.update_output("pred", pred)
        elif isinstance(item, dict):
            item["pred"] = pred
        else:
            setattr(item, "pred", pred)

    def _maybe_open_log_file(self, global_processed: int) -> None:
        if self.qa_file is None or (global_processed - 1) % 10 == 0:
            if self.qa_file:
                self.qa_file.write("\n========== Log End ==========\n")
                self.qa_file.close()
                self.qa_file = None
            batch_start = ((global_processed - 1) // 10) * 10 + 1
            batch_end = batch_start + 9
            file_path = os.path.join(self.log_dir, f"{self.qa_prefix}_{batch_start}-{batch_end}.txt")
            self.qa_file = open(file_path, "a", encoding="utf-8")
            if os.path.getsize(file_path) == 0:
                self.qa_file.write("========== Log Start ==========\n")
                self.qa_file.flush()

    @staticmethod
    def _append_log_section(log_list: List[str], title: str, content: str) -> None:
        log_list.append(f"\n{'=' * 80}\n")
        log_list.append(f"【{title}】\n")
        log_list.append(f"{'=' * 80}\n")
        log_list.append((content if content else "No content") + "\n")

    def __del__(self):
        if hasattr(self, "qa_file") and self.qa_file is not None:
            try:
                self.qa_file.write("\n========== Log End ==========\n")
                self.qa_file.close()
            except Exception:
                pass


# Short aliases for convenient imports.
PruneRAG = FlashRAGPruneRAG
PruneRAGPipeline = FlashRAGPruneRAG

