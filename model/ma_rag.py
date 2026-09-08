import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_generator, get_retriever


# ============================================================
# Prompts adapted from the original MA-RAG project
# ============================================================

PLANNING_SYSTEM_MESSAGE = """You are tasked with assisting users in generating structured plans for answering questions. Your goal is to deconstruct a query into manageable, simpler components. For each question, perform the following tasks:

*Analysis: Identify the core components of the question, emphasizing the key elements and context needed for a comprehensive understanding. Determine whether the question is straightforward or requires multiple steps to provide an accurate answer.

*Plan Creation:
- Break down the question into smaller, simpler questions by reasoning that lead to the final answer. Ensure those steps are non overlap. Stop at the step where its answer can be the final answer.
- Ensure each step is clear and logically sequenced.
- Consider any past attempts or experiences provided as context, and use them to refine or adjust the plan to avoid past pitfalls.
- Each step is a question to search, or to aggregate output from previous steps. Do not verify previous step.
- Your task is planning, not answering. Do not put any answer from your knowledge into the plan.

# Notes:
- Your task is to provide clarity and guidance on the approach to answering, rather than providing the final answer directly.
- Put your output in a list of strings, each string describes a sub-task.

# Example plan:
Question: What country of origin does House of Cosbys and Bill Cosby have in common?
Steps: ["Determine the country of origin for House of Cosbys.", "Determine the country of origin for Bill Cosby.", "From previous answers, which is the common country"]
Question: Which film has the director who died later, The House Of Tears or College Ranga?
Steps: ["Identify the director of The House Of Tears", "Identify the director of College Ranga", "When did the director of The House Of Tears die", "When did the director of College Ranga die", "Compare the death dates of the two directors to determine which one died later."]
Question: Peter Griffith's granddaughter had her screen debut in what 1999 film?
Steps: ["Who is Peter Griffith's granddaughter", "What 1999 film did she have screen debut"]
Question: how many episodes are in chicago fire season 4?
Steps: ["how many episodes are in chicago fire season 4"]
Question: Are both directors of films The Stoneman Murders and Chandralekha (2014 Film) from the same country?
Steps: ["Who is the director of film The Stoneman Murders", "Who is the director of film Chandralekha (2014 Film)", "Determine the country of origin for the director of The Stoneman Murders", "Determine the country of origin for the director of Chandralekha (2014 Film)", "Compare the two countries to determine if they are the same"]

Return ONLY valid JSON in this format:
{"analysis": "brief analysis", "step": ["step 1", "step 2"]}
"""

PLANNING_HUMAN_MESSAGE = """Question: {question}?
Past experience:
{memory}
"""

STEP_SYSTEM_MESSAGE = """Given a plan, the current step, and the results from finished steps, decide the task for this step.
Output the type of task and the query.
The query needs to be in detail. Do not put vague phrases like "based on the previous results" in the query.
Include all information from previous step results in the query if it is useful, especially for aggregate tasks.
Be concise.

The task type must be one of:
- question-answering: the step needs retrieval and evidence-based QA.
- aggregate: the step only combines previous step outputs.

Return ONLY valid JSON in this format:
{"type": "question-answering|aggregate", "task": "the executable query or aggregation task"}
"""

STEP_HUMAN_MESSAGE = """Plan: {plan}
Current step: {cur_step}
Results of finished steps:
{memory}
"""

EXTRACT_SYSTEM_MESSAGE = """Summarize and extract all relevant information from the provided passages based on the given question. Remove all irrelevant information. Think step-by-step.

# Steps

1. **Identify Key Elements**: Read the question carefully to determine what specific information is being requested.
2. **Analyze Passages**: Review the passages thoroughly to find any segments that contain information relevant to the question.
3. **Extract Relevant Information**: Highlight or note down sentences, phrases, or words from the passages that relate to the question.
4. **Remove Irrelevant Details**: Ensure that all extracted information is relevant to the question, eliminating any unnecessary or unrelated content.

# Output Format
- Output a list of notes. Each note contains related information from the passage as well as precise evidences and why.
- Each note is clear, standalone.

# Notes
- Avoid any irrelevant details.
- If a piece of information is mentioned in multiple places, include it only once.
- If there is no related information, output exactly: No related information from this document.
"""

EXTRACT_HUMAN_MESSAGE = """Passage:
###
{passage}
###

Query: {question}?
"""

QA_SYSTEM_MESSAGE = """You are an assistant for question-answering tasks. Use the following process to deliver concise and precise answers based on the retrieved context. If all of retrieved context are not relevant, answer based on general knowledge.

1. **Analyze Carefully**: Begin by thoroughly analyzing both the question and the provided context.

2. **Identify Core Details**: Focus on identifying the essential names, terms, or details that directly answer the question. Disregard any irrelevant information.

3. **Provide a Concise Answer**:
   - Remove redundant words and extraneous details.
   - Present the answer by listing only the necessary names, terms, or very brief facts that are crucial for answering the question.

4. **Clarity and Accuracy**: Ensure that your answer is clear and maintains the original meaning of the information provided.

5. **Consensus**: If the contexts are not consensus, pick one which is the most logical, consensus, or confident.

6. **IMPORTANT**: If the provided context could not bring any related information, answer by yourself.

Return ONLY valid JSON in this format:
{"analysis": "brief reasoning", "answer": "concise answer", "success": "Yes|No", "rating": 0}
"""

QA_HUMAN_MESSAGE = """Retrieved documents:
{context}
Question: {question}
"""

AGGREGATE_SYSTEM_MESSAGE = """Answer the question from human.
Provide a Concise Answer:
- Remove redundant words and extraneous details.
- Present the answer by listing only the necessary names, terms, or very brief facts that are crucial for answering the question.
- If you have multiple answers, only output one answer which is most confident.
Think step-by-step.

Return ONLY valid JSON in this format:
{"analysis": "brief reasoning", "answer": "concise answer", "success": "Yes|No", "rating": 0}
"""

AGGREGATE_HUMAN_MESSAGE = """{question}"""

SUMMARY_SYSTEM_MESSAGE = """Your task is writing a summary about a plan to solve a question.

** Input
- The question
- The plan: a sequence of sub-task. Ideally if we can solve all of them, we can solve the question.
- Output of each step in the plan.

**Output
- If all of steps are solved, output the final answer for the original question by combining step's output and a confident score calculated as the mean of scores from steps.
- If one or many of steps are unsolved, but you can still find the answer based on step's output, output the final answer.
- If you could not find the final answer for the question, output Unsuccessful and why, with score 0.

Return ONLY valid JSON in this format:
{"output": "Successful or Unsuccessful with brief reason", "answer": "Output ONLY the final answer here - NO reasoning words, NO explanations, NO markdown, NO extra text.", "score": 0}
"""

SUMMARY_HUMAN_MESSAGE = """Original Question: {question}
Plan: {plan}
Output of steps:
{memory}

Original Question: {question}
"""


class MARAGPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, generator=None, retriever=None):
        super().__init__(config, prompt_template)
        self.config = config
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever

    
        self.retrieval_topk = 3
        raw_max_steps = None

        if raw_max_steps in [None, "", 0, "0", "none", "None"]:
            self.max_plan_steps = None
        else:
            self.max_plan_steps = int(raw_max_steps)

        # Optional: if set to False, QA will output unknown when context is empty.
        # Default True follows the original MA-RAG prompt: use general knowledge
        # when all retrieved context is irrelevant.
        self.allow_general_knowledge = True

        # Save intermediate MA-RAG traces for later analysis.
        # save_dir = 'output'
        # os.makedirs(save_dir, exist_ok=True)
        # self.trace_path = os.path.join(save_dir, "ma_rag_traces.jsonl")

    # ============================================================
    # Public entry
    # ============================================================
    def run(self, dataset, do_eval=True):
        """Run MA-RAG on a FlashRAG dataset."""
        total = len(dataset) if hasattr(dataset, "__len__") else None

        for idx, item in tqdm(enumerate(dataset), total=total, desc="MA-RAG Inference"):
            question = self._get_question(item)
            # item_id = self._get_item_id(item, idx)

            trace = self.answer_question(question)
            final_answer = trace.get("final_answer", "unknown")
            print(final_answer)

            # FlashRAG dataset items usually support update_output.
            if hasattr(item, "update_output"):
                item.update_output("pred", final_answer)
                item.update_output("raw_pred", trace.get("raw_summary", final_answer))
            elif isinstance(item, dict):
                item["pred"] = final_answer
                item["raw_pred"] = trace.get("raw_summary", final_answer)

            # with open(self.trace_path, "a", encoding="utf-8") as f:
            #     record = {
            #         "idx": idx,
            #         "id": item_id,
            #         "question": question,
            #         **trace,
            #     }
            #     f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return self.evaluate(dataset, do_eval=do_eval)

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer one question using the MA-RAG workflow."""
        plan = self._plan(question)
        if not plan:
            plan = [question]

        step_tasks: List[Dict[str, str]] = []
        step_outputs: List[Dict[str, Any]] = []

        # This sequential loop is the non-LangGraph equivalent of the original
        # plan_executor graph: task_definer -> single_task_execute -> task_definer.
        for step_idx, cur_step in enumerate(plan[: self.max_plan_steps]):
            step_task = self._define_step(plan, cur_step, step_outputs)
            step_tasks.append(step_task)

            task_type = step_task.get("type", "question-answering").lower().strip()
            task_query = step_task.get("task", cur_step).strip() or cur_step

            if "aggregate" in task_type or "summary" in task_type:
                output = self._aggregate(task_query, step_outputs)
            else:
                output = self._rag_answer(task_query)

            output["step_index"] = step_idx
            output["original_step"] = cur_step
            step_outputs.append(output)

            # Original LangGraph stops before the next step if the latest step fails.
            success_flag = str(output.get("success", "Yes")).lower().strip()
            if success_flag.startswith("no"):
                break

        summary = self._summarize_plan(question, plan, step_tasks, step_outputs)
        final_answer = self._clean_final_answer(
            summary.get("answer") or summary.get("output") or "unknown"
        )

        return {
            "plan": plan,
            "step_tasks": step_tasks,
            "step_outputs": step_outputs,
            "plan_summary": summary,
            "raw_summary": summary.get("raw", summary.get("output", "")),
            "final_answer": final_answer,
        }

    # ============================================================
    # MA-RAG stage 1: Planner
    # ============================================================
    def _plan(self, question: str) -> List[str]:
        messages = [
            {"role": "system", "content": PLANNING_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": PLANNING_HUMAN_MESSAGE.format(
                    question=question,
                    memory="empty",
                ),
            },
        ]

        fallback = {"analysis": "fallback", "step": [question]}
        data, _ = self._json_generate(messages, fallback)

        steps = data.get("step") or data.get("steps") or data.get("plan") or [question]
        if isinstance(steps, str):
            steps = self._parse_lines(steps)
        elif isinstance(steps, list):
            steps = [str(s).strip() for s in steps if str(s).strip()]
        else:
            steps = [question]

        return steps[: self.max_plan_steps] if steps else [question]

    # ============================================================
    # MA-RAG stage 2: Step Definer
    # ============================================================
    def _define_step(
        self,
        plan: List[str],
        current_step: str,
        previous_outputs: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        memory = self._format_step_definer_memory(plan, previous_outputs)

        messages = [
            {"role": "system", "content": STEP_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": STEP_HUMAN_MESSAGE.format(
                    plan=f"[{', '.join(plan)}]",
                    cur_step=current_step,
                    memory=memory if memory else "None",
                ),
            },
        ]

        fallback = {"type": "question-answering", "task": current_step}
        data, _ = self._json_generate(messages, fallback)

        task_type = str(data.get("type", "question-answering")).strip().lower()
        task = str(data.get("task", current_step)).strip()

        if not task:
            task = current_step
        if task_type not in ["question-answering", "aggregate"]:
            if "aggregate" in task_type or "summar" in task_type or "combine" in task_type:
                task_type = "aggregate"
            else:
                task_type = "question-answering"

        return {"type": task_type, "task": task}

    # ============================================================
    # MA-RAG stage 3: Retrieval + Evidence Extractor + QA Agent
    # ============================================================
    def _rag_answer(self, query: str) -> Dict[str, Any]:
        docs = self._search_documents(query)

        notes = []
        for doc_idx, doc in enumerate(docs):
            content = self._get_doc_content(doc)
            doc_id = self._get_doc_id(doc, doc_idx)
            note = self._extract_note(query, content)

            notes.append(
                {
                    "doc_id": doc_id,
                    "note": note,
                    "document": content,
                }
            )

        # Follow original formatting: doc_id: [note]
        context_items = []
        for n in notes:
            note_text = n.get("note", "")
            context_items.append(f"doc_{n['doc_id']}: [{note_text}]")

        if context_items:
            context = "\n\n".join(context_items)
        else:
            context = "No related information from retrieved documents."

        qa_system = QA_SYSTEM_MESSAGE
        if not self.allow_general_knowledge:
            qa_system = qa_system.replace(
                "If all of retrieved context are not relevant, answer based on general knowledge.",
                "If the retrieved context is not relevant, answer unknown.",
            ).replace(
                "If the provided context could not bring any related information, answer by yourself.",
                "If the provided context could not bring any related information, answer unknown.",
            )

        messages = [
            {"role": "system", "content": qa_system},
            {
                "role": "user",
                "content": QA_HUMAN_MESSAGE.format(context=context, question=query),
            },
        ]

        fallback = {
            "analysis": "Fallback because the model did not return valid JSON.",
            "answer": "unknown",
            "success": "No",
            "rating": 0,
        }
        data, raw = self._json_generate(messages, fallback)

        answer = self._clean_final_answer(str(data.get("answer", "unknown")))
        success = str(data.get("success", "Yes")).strip()
        if answer.lower() in ["", "unknown", "no answer", "none"]:
            success = "No"

        return {
            "type": "question-answering",
            "query": query,
            "answer": answer if answer else "unknown",
            "analysis": str(data.get("analysis", "")),
            "success": success,
            "rating": self._safe_int(data.get("rating", 5), default=5),
            "notes": notes,
            "raw": raw,
        }

    def _extract_note(self, question: str, passage: str) -> str:
        if not passage.strip():
            return "No related information from this document."

        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": EXTRACT_HUMAN_MESSAGE.format(
                    passage=passage,
                    question=question,
                ),
            },
        ]
        return self._generate(messages)

    # ============================================================
    # MA-RAG stage 4: Aggregation step
    # ============================================================
    def _aggregate(self, task: str, previous_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        memory = self._format_memory(previous_outputs)
        aggregate_question = task
        if memory:
            aggregate_question = f"{task}\n\nPrevious step results:\n{memory}"

        messages = [
            {"role": "system", "content": AGGREGATE_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": AGGREGATE_HUMAN_MESSAGE.format(question=aggregate_question),
            },
        ]

        fallback = {
            "analysis": "No previous results to aggregate.",
            "answer": "unknown",
            "success": "No",
            "rating": 0,
        }
        data, raw = self._json_generate(messages, fallback)

        answer = self._clean_final_answer(str(data.get("answer", "unknown")))
        success = str(data.get("success", "Yes")).strip()
        if answer.lower() in ["", "unknown", "no answer", "none"]:
            success = "No"

        return {
            "type": "aggregate",
            "query": task,
            "answer": answer if answer else "unknown",
            "analysis": str(data.get("analysis", "")),
            "success": success,
            "rating": self._safe_int(data.get("rating", 5), default=5),
            "notes": [],
            "raw": raw,
        }

    # ============================================================
    # MA-RAG stage 5: Final Summary Agent
    # ============================================================
    def _summarize_plan(
        self,
        original_question: str,
        plan: List[str],
        step_tasks: List[Dict[str, str]],
        step_outputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        memory_lines = []
        for i, output in enumerate(step_outputs):
            task = step_tasks[i].get("task", plan[i] if i < len(plan) else "")
            memory_lines.append(
                f"Task: {plan[i] if i < len(plan) else task}\n"
                f"Question: {task}\n"
                f"Answer: {output.get('answer', '')}\n"
                f"Confident score: {output.get('rating', '')}\n"
                f"Success: {output.get('success', '')}\n"
                f"Analysis: {output.get('analysis', '')}\n"
            )
        memory = "\n".join(memory_lines)

        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": SUMMARY_HUMAN_MESSAGE.format(
                    question=original_question,
                    plan=f"[{', '.join(plan)}]",
                    memory=memory if memory else "None",
                ),
            },
        ]

        fallback_answer = step_outputs[-1].get("answer", "unknown") if step_outputs else "unknown"
        fallback = {
            "output": "fallback summary",
            "answer": fallback_answer,
            "score": 0,
        }
        data, raw = self._json_generate(messages, fallback)

        return {
            "output": str(data.get("output", "")),
            "answer": self._clean_final_answer(str(data.get("answer", fallback_answer))),
            "score": self._safe_int(data.get("score", 0), default=0),
            "raw": raw,
        }

    # ============================================================
    # Retrieval helper
    # ============================================================
    def _search_documents(self, query: str) -> List[Any]:
        """Search documents with several common FlashRAG retriever APIs."""
        docs = self.retriever.search(query, self.retrieval_topk)
        # Some retrievers return a dict, e.g. {"docs": [...]}.
        if isinstance(docs, dict):
            for key in ["docs", "documents", "retrieval_result", "results"]:
                if key in docs and isinstance(docs[key], list):
                    return docs[key]
            return []

        # Some retrievers return a tuple: (docs, scores) or (docs, doc_ids).
        if isinstance(docs, tuple):
            docs = docs[0]

        if docs is None:
            return []
        if isinstance(docs, list):
            return docs[: self.retrieval_topk]
        return [docs]

    # ============================================================
    # LLM generation helpers
    # ============================================================
    def _generate(self, messages: List[Dict[str, str]], max_retry: int = 1) -> str:
        prompt = self._render_prompt(messages)
        last_output = ""
        for _ in range(max_retry + 1):
            output = self.generator.generate(prompt)[0]
            last_output = str(output).strip()
            if last_output:
                return last_output
        return last_output

    def _json_generate(
        self,
        messages: List[Dict[str, str]],
        fallback: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        raw = self._generate(messages)
        data = self._extract_json(raw)
        if data is None:
            merged = dict(fallback)
            return merged, raw

        merged = dict(fallback)
        merged.update(data)
        return merged, raw

    def _render_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Render chat messages through FlashRAG prompt_template when available."""
        template = getattr(self, "prompt_template", None)
        if template is not None and hasattr(template, "get_string"):
            try:
                return template.get_string(messages=messages)
            except Exception:
                pass

        # Fallback renderer for environments without a chat prompt template.
        rendered = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            rendered.append(f"{role}: {content}")
        rendered.append("ASSISTANT:")
        return "\n\n".join(rendered)

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            obj = json.loads(cleaned)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        # Extract first JSON-like object.
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                obj = json.loads(candidate)
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass

        # Soft fallback for common non-JSON structured outputs.
        parsed = MARAGPipeline._parse_structured_text(cleaned)
        return parsed if parsed else None

    @staticmethod
    def _parse_structured_text(text: str) -> Optional[Dict[str, Any]]:
        """Best-effort parser for model outputs that ignore JSON instructions."""
        if not text:
            return None

        result: Dict[str, Any] = {}

        # Steps: ["...", "..."]
        step_match = re.search(r"(?:steps?|plan)\s*:\s*\[(.*?)\]", text, flags=re.IGNORECASE | re.DOTALL)
        if step_match:
            inside = step_match.group(1)
            quoted = re.findall(r"['\"]([^'\"]+)['\"]", inside)
            if quoted:
                result["step"] = [s.strip() for s in quoted if s.strip()]
            else:
                result["step"] = MARAGPipeline._parse_lines(inside)

        # Type: aggregate / question-answering
        type_match = re.search(r"type\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
        if type_match:
            result["type"] = type_match.group(1).strip().strip('"\'')

        # Task: ...
        task_match = re.search(r"task\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
        if task_match:
            result["task"] = task_match.group(1).strip().strip('"\'')

        # Answer / final answer.
        ans_match = re.search(r"(?:final\s+answer|answer)\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
        if ans_match:
            result["answer"] = ans_match.group(1).strip().strip('"\'')

        # Success.
        success_match = re.search(r"success\s*:\s*(yes|no)", text, flags=re.IGNORECASE)
        if success_match:
            result["success"] = success_match.group(1).strip()

        # Rating / score.
        rating_match = re.search(r"(?:rating|score)\s*:\s*(\d+)", text, flags=re.IGNORECASE)
        if rating_match:
            value = int(rating_match.group(1))
            result["rating"] = value
            result["score"] = value

        # Output.
        output_match = re.search(r"output\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
        if output_match:
            result["output"] = output_match.group(1).strip().strip('"\'')

        return result if result else None

    # ============================================================
    # Formatting / parsing helpers
    # ============================================================
    @staticmethod
    def _get_question(item) -> str:
        if hasattr(item, "question"):
            return str(item.question)
        if isinstance(item, dict):
            return str(item.get("question", item.get("input", item.get("query", ""))))
        return str(item)

    @staticmethod
    def _get_item_id(item, idx: int) -> str:
        if hasattr(item, "id"):
            return str(item.id)
        if isinstance(item, dict):
            return str(item.get("id", item.get("qid", idx)))
        return str(idx)

    @staticmethod
    def _get_doc_content(doc) -> str:
        if isinstance(doc, dict):
            for key in ["contents", "content", "text", "passage", "document"]:
                if key in doc and doc[key] is not None:
                    return str(doc[key])
            # Some FlashRAG docs keep title and text separately.
            title = str(doc.get("title", "")).strip()
            text = str(doc.get("body", "")).strip()
            return (title + "\n" + text).strip()
        if hasattr(doc, "contents"):
            return str(doc.contents)
        if hasattr(doc, "content"):
            return str(doc.content)
        if hasattr(doc, "text"):
            return str(doc.text)
        return str(doc)

    @staticmethod
    def _get_doc_id(doc, idx: int) -> str:
        if isinstance(doc, dict):
            for key in ["id", "doc_id", "pid", "title"]:
                if key in doc and doc[key] is not None:
                    return str(doc[key])
        if hasattr(doc, "id"):
            return str(doc.id)
        if hasattr(doc, "doc_id"):
            return str(doc.doc_id)
        return str(idx)

    @staticmethod
    def _parse_lines(text: str) -> List[str]:
        lines = []
        for line in str(text).split("\n"):
            line = line.strip()
            line = re.sub(r"^[-*\d\.\)\s]+", "", line).strip()
            if line:
                lines.append(line)
        return lines

    @staticmethod
    def _format_step_definer_memory(
        plan: List[str],
        step_outputs: List[Dict[str, Any]],
    ) -> str:
        if not step_outputs:
            return ""

        lines = []
        for i, output in enumerate(step_outputs):
            task = plan[i] if i < len(plan) else output.get("query", "")
            lines.append(
                f"Task: {task}\n"
                f"Answer: {output.get('answer', '')}\n"
                f"Success: {output.get('success', '')}\n"
                f"Confidence: {output.get('rating', '')}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_memory(step_outputs: List[Dict[str, Any]]) -> str:
        if not step_outputs:
            return ""

        lines = []
        for i, output in enumerate(step_outputs, 1):
            lines.append(
                f"Step {i}:\n"
                f"Query/Task: {output.get('query', '')}\n"
                f"Answer: {output.get('answer', '')}\n"
                f"Success: {output.get('success', '')}\n"
                f"Confidence: {output.get('rating', '')}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _is_valid_note(note: str) -> bool:
        if not note:
            return False
        lower = note.strip().lower()
        invalid_patterns = [
            "no related information",
            "not related",
            "irrelevant",
            "cannot answer",
            "unknown",
        ]
        return not any(p in lower for p in invalid_patterns)

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            match = re.search(r"\d+", str(value))
            return int(match.group(0)) if match else default

    @staticmethod
    def _clean_final_answer(text: str) -> str:
        text = (text or "").strip()
        prefixes = [
            "The answer is:",
            "The answer is",
            "Answer:",
            "Final Answer:",
            "[Final Answer]",
            "Final answer:",
            "Final answer",
        ]
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        return text.strip().strip('"').strip("'").strip()


# This wrapper follows the style of functions in model/baseline.py.
def ma_rag(cfg, test_data):
    pipeline = MARAGPipeline(cfg)
    return pipeline.run(test_data)
