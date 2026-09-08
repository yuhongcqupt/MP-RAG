from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_generator, get_retriever
from tqdm import tqdm
import os
import json
from collections import defaultdict


class MP_RAG(BasicPipeline):
    def __init__(self, config, prompt_template=None,
                 max_fusion_rounds=6, generator=None, retriever=None):

        super().__init__(config, prompt_template)
        self.config = config
        self.max_fusion_rounds = max_fusion_rounds
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever

        self.log_dir = "qa_logs_non_adversarial_musique_full_dataset"
        self.qa_prefix = "question_result"
        self.qa_file = None

        self.progress_path = "progress_non_adversarial_musique_full_dataset.txt"
        self.pred_path = "pred_results_non_adversarial_musique_full_dataset.json"

    def run(self, dataset, do_eval=True):

        os.makedirs(self.log_dir, exist_ok=True)


        existing_preds = {}
        if os.path.exists(self.pred_path):
            with open(self.pred_path, "r", encoding="utf-8") as f:
                existing_preds = json.load(f)
        print(f"Loaded {len(existing_preds)} existing predictions")


        for item in dataset:
            item_id = str(getattr(item, 'id', ''))
            if item_id in existing_preds:
                item.update_output("pred", existing_preds[item_id])


        processed_ids = set()
        if os.path.exists(self.progress_path):
            with open(self.progress_path, "r", encoding="utf-8") as f:
                for line in f:
                    pid = line.strip()
                    if pid:
                        processed_ids.add(pid)
        print(f"Loaded from {self.progress_path}: {len(processed_ids)} completed questions")


        total_done = len(processed_ids)
        global_processed = total_done

        for idx, item in tqdm(enumerate(dataset), desc="Inference: "):

            item_id = str(getattr(item, 'id', idx))


            if item_id in processed_ids:
                print(f"Skipping processed question ID: {item_id}")
                continue


            global_processed += 1


            if self.qa_file is None or (global_processed - 1) % 10 == 0:

                if self.qa_file:
                    self.qa_file.write("\n========== Log End ==========\n")
                    self.qa_file.close()
                    self.qa_file = None

                batch_start = ((global_processed - 1) // 10) * 10 + 1
                batch_end = batch_start + 9
                file_name = f"{self.qa_prefix}_{batch_start}-{batch_end}.txt"
                file_path = os.path.join(self.log_dir, file_name)

                self.qa_file = open(file_path, "a", encoding="utf-8")
                if os.path.getsize(file_path) == 0:
                    self.qa_file.write("========== Log Start ==========\n")
                    self.qa_file.flush()
                    print(f"New log file created: {file_path}")
                else:
                    print(f"Appending logs to existing file: {file_path}")


            current_log = []
            current_log.append(f"\n{'=' * 80}\n")
            current_log.append(f"[Question ID]: {item_id}\n")
            current_log.append(f"{'=' * 80}\n")

            item.question, sub_queries, decomposition_type = self.question_decomposition(item, idx)

            self._append_log_section(current_log, "Original Question", item.question)
            self._append_log_section(current_log, "Question Decomposition Type", decomposition_type)
            sub_q_log = ""
            if sub_queries:
                for i, sub_q in enumerate(sub_queries, 1):
                    sub_q_log += f"Sub-question {i}: {sub_q}\n"
            else:
                sub_q_log = "No decomposition\n"
            self._append_log_section(current_log, "Question Decomposition Result", sub_q_log)

            original_filtered_results = self.original_query_retriever_filtered(item.question)
            self._append_log_section(current_log, "Original-Question Retrieval and Filtering Results",
                                     self._format_filtered_results(original_filtered_results))

            sub_id = 0
            sub_question_to_answer = {}
            global_flat_answer_pool = []
            for q in sub_queries:
                if (sub_id != 0 and decomposition_type == "Sequential"):

                    previous_context = ""
                    if sub_question_to_answer:
                        last_q, last_ans = list(sub_question_to_answer.items())[-1]
                        previous_context = f"- {last_q}: {last_ans}\n"


                    replace_round_message = [
                        self._sequential_sub_query_replace_system_message(),
                        {
                            "role": "user",
                            "content": (
                                f"Previous Sub-questions & Answers:\n{previous_context}\n\n"
                                f"Current Sub-question: {q}"
                            )
                        }
                    ]
                    replace_prompt = self.prompt_template.get_string(messages=replace_round_message)
                    replaced_q = self.generator.generate(replace_prompt)[0].strip()


                    replaced_q = replaced_q.strip("\"'.,!?\n").strip()
                    print(f"Sequential replacement: original question[{sub_id}] → {replaced_q} → new question → {replaced_q}")


                    replace_log = (
                        f"Previous Sub-question Answers:\n{previous_context}\n"
                        f"Current Original Sub-question: {q}\n"
                        f"Rewritten Sub-question: {replaced_q}\n"
                    )
                    self._append_log_section(current_log, f"Sequential Sub-question Rewriting (Sub-ID: {sub_id})", replace_log)

                    replaced_q = replaced_q.lstrip("0123456789. ").strip()
                    docs = self.retriever.search(replaced_q, 3)
                    kept_docs = []
                    for doc in docs:
                        round_message = [
                            self._document_filter_system_message(),
                            {
                                "role": "user",
                                "content": (
                                    f"Question: {replaced_q}\n\n"
                                    f"Document:\n{doc['contents']}"
                                )
                            }
                        ]

                        input_prompt = self.prompt_template.get_string(messages=round_message)
                        output = self.generator.generate(input_prompt)[0]

                        clean_output = output.strip().lower().rstrip('.').rstrip(',')
                        if "yes" in clean_output and "no" not in clean_output:
                            kept_docs.append(doc)
                    final_answer_pool = self.first_stage_fusion(q, kept_docs, current_log, sub_id)

                    if not final_answer_pool:
                        current_sub_answer = "unknown"
                    else:
                        current_sub_answer = final_answer_pool[0]["evidence"]
                    sub_question_to_answer[q] = current_sub_answer
                    global_flat_answer_pool.extend(final_answer_pool)
                else:
                    docs = self.retriever.search(q, 3)
                    kept_docs = []
                    for doc in docs:
                        round_message = [
                            self._document_filter_system_message(),
                            {
                                "role": "user",
                                "content": (
                                    f"Question: {q}\n\n"
                                    f"Document:\n{doc['contents']}"
                                )
                            }
                        ]

                        input_prompt = self.prompt_template.get_string(messages=round_message)
                        output = self.generator.generate(input_prompt)[0]

                        clean_output = output.strip().lower().rstrip('.').rstrip(',')
                        if "yes" in clean_output and "no" not in clean_output:
                            kept_docs.append(doc)
                    final_answer_pool = self.first_stage_fusion(q, kept_docs, current_log, sub_id)

                    if not final_answer_pool:
                        current_sub_answer = "unknown"
                    else:
                        current_sub_answer = final_answer_pool[0]["evidence"]
                    sub_question_to_answer[q] = current_sub_answer
                    global_flat_answer_pool.extend(final_answer_pool)
                sub_id = sub_id + 1

            middle_answer, structured_evidence_text = self.generate_final_answer_with_llm(item.question,
                                                                                          global_flat_answer_pool)
            self._append_log_section(current_log, "First-stage Debate Final Answer", middle_answer)
            final_answer = self.second_stage_fusion(item, original_filtered_results, middle_answer,
                                                    structured_evidence_text, current_log)
            item.update_output("pred", final_answer)
            self._append_log_section(current_log, "Final Answer", final_answer)


            self.qa_file.write("".join(current_log))
            self.qa_file.flush()


            existing_preds[item_id] = final_answer
            with open(self.pred_path, "w", encoding="utf-8") as f:
                json.dump(existing_preds, f, ensure_ascii=False, indent=2)


            with open(self.progress_path, "a", encoding="utf-8") as f:
                f.write(f"{item_id}\n")
            processed_ids.add(item_id)

            print(f"Question {item_id} completed, final answer:{final_answer}")

        dataset = self.evaluate(dataset, do_eval=do_eval)

    def question_decomposition(self, item, idx):
        input_query = item.question


        question_type = self.judge_question_type(input_query)


        if question_type == "parallel":
            sub_queries = self._decompose_parallel(input_query, idx)
        elif question_type == "serial":
            sub_queries = self._decompose_serial(input_query, idx)
        else:
            sub_queries = [input_query]


        if question_type == "serial":
            decomposition_type = "Sequential"
        elif question_type == "parallel":
            decomposition_type = "Parallel"
        else:
            decomposition_type = "Parallel"


        return item.question, sub_queries, decomposition_type

    def original_query_retriever_filtered(self, input_query):
        retrieval_results = {
            "original_query": {
                "query": input_query,
                "docs": self.retriever.search(input_query, 3)
            }
        }

        filtered_results = {
            "original_query": {
                "query": retrieval_results["original_query"]["query"],
                "docs": []
            }
        }
        original_query = retrieval_results["original_query"]["query"]

        for doc in retrieval_results["original_query"]["docs"]:
            round_message = [
                self._document_filter_system_message(),
                {
                    "role": "user",
                    "content": (
                        f"Question: {original_query}\n\n"
                        f"Document:\n{doc['contents']}"
                    )
                }
            ]

            input_prompt = self.prompt_template.get_string(messages=round_message)
            output = self.generator.generate(input_prompt)[0]

            clean_output = output.strip().lower().rstrip('.').rstrip(',')
            if "yes" in clean_output and "no" not in clean_output:
                filtered_results["original_query"]["docs"].append(doc)
        return filtered_results

    def first_stage_fusion(self, sub_query, kept_docs, log_list=None, sub_id=-1):
        final_answer_pool = []
        if log_list is not None:
            title = f"First-stage Debate - Sub-question {sub_id}" if sub_id >= 0 else "First-stage Debate"
            log_list.append(f"\n{'=' * 60}\n")
            log_list.append(f"【{title}】\n")
            log_list.append(f"Target Sub-question: {sub_query}\n")
            log_list.append(f"{'=' * 60}\n")


        candidate_answers = []
        answer_id_counter = 0

        for doc_idx, doc in enumerate(kept_docs):
            doc_content = doc["contents"]
            doc_id = doc.get("id", f"doc_{doc_idx}")

            gen_messages = [
                self._answer_generation_system_message(),
                {
                    "role": "user",
                    "content": (
                        f"Sub-question: {sub_query}\n\n"
                        f"Document content:\n{doc_content}"
                    )
                }
            ]
            gen_prompt = self.prompt_template.get_string(messages=gen_messages)
            answer_text = self.generator.generate(gen_prompt)[0]
            is_unknown = self._process_llm_answer_output(answer_text)

            if is_unknown:
                continue

            candidate_answers.append({
                "position_id": f"Position {answer_id_counter + 1}",
                "answer_text": answer_text,
                "doc_id": doc_id,
                "doc_content": doc_content,
                "prompt": gen_prompt
            })
            answer_id_counter += 1


        if len(kept_docs) > 1:

            all_docs_content = "\n\n".join([
                f"[Document {i + 1}]: {doc['contents']}"
                for i, doc in enumerate(kept_docs)
            ])
            meta_doc_id = "meta_all_docs"


            meta_gen_messages = [
                self._answer_generation_system_message(),
                {
                    "role": "user",
                    "content": (
                        f"Sub-question: {sub_query}\n\n"
                        f"Document content (ALL relevant documents):\n{all_docs_content}"
                    )
                }
            ]
            meta_gen_prompt = self.prompt_template.get_string(messages=meta_gen_messages)
            meta_answer_text = self.generator.generate(meta_gen_prompt)[0]


            meta_is_unknown = self._process_llm_answer_output(meta_answer_text)
            if not meta_is_unknown:

                candidate_answers.append({
                    "position_id": "Position Meta",
                    "answer_text": meta_answer_text,
                    "doc_id": meta_doc_id,
                    "doc_content": all_docs_content,
                    "prompt": meta_gen_prompt
                })


        if log_list is not None:
            gen_log = f"Valid documents: {len(kept_docs)}, generated valid positions: {len(candidate_answers)}\n"
            for ans in candidate_answers:
                gen_log += f"\n[{ans['position_id']} (Doc: {ans['doc_id']})]:\n{ans['answer_text']}\n"
            log_list.append(gen_log)


        if not candidate_answers:

            internal_messages = [
                self._internal_knowledge_system_message(),
                {"role": "user", "content": f"Sub-question: {sub_query}"}
            ]
            internal_prompt = self.prompt_template.get_string(messages=internal_messages)
            internal_output = self.generator.generate(internal_prompt)[0]
            is_unknown = self._process_llm_answer_output(internal_output)

            final_answer = "unknown" if is_unknown else internal_output
            final_answer_pool.append({
                "sub_query": sub_query,
                "source": "internal" if not is_unknown else "ultimate_fallback",
                "evidence_id": "internal",
                "evidence": final_answer,
                "document": None,
                "doc_id": None,
                "reason": "No valid document-based answer"
            })
            return final_answer_pool

        if len(candidate_answers) == 1:
            ans = candidate_answers[0]
            final_answer_pool.append({
                "sub_query": sub_query,
                "source": "external",
                "evidence_id": ans["position_id"],
                "evidence": ans["answer_text"],
                "document": ans["doc_content"],
                "doc_id": ans["doc_id"],
                "reason": "Only one valid position"
            })
            return final_answer_pool


        position_history = {ans["position_id"]: [] for ans in candidate_answers}
        prev_judge_answer = None

        for round_id in range(3):
            print(f"\n===== First Stage Round {round_id + 1} =====")
            if log_list is not None:
                log_list.append(f"\n--- First Stage Round {round_id + 1} ---\n")


            if round_id == 0:
                for ans in candidate_answers:
                    pos_id = ans["position_id"]

                    pos_system_msg = {
                        "role": "system",
                        "content": (
                            "First, clearly state the answer to the question, "
                            "then explain why this answer is correct based on the provided Relevant Documents. "
                            "Do not add extra content, focus on the answer and evidence-based explanation."
                        )
                    }

                    pos_user_msg = {
                        "role": "user",
                        "content": (
                            f"Original Question:\n{sub_query}\n\n"
                            f"Relevant Documents:\n{ans['doc_content']}\n\n"
                        )
                    }

                    current_pos_messages = [pos_system_msg, pos_user_msg]
                    pos_prompt = self.prompt_template.get_string(messages=current_pos_messages)
                    pos_answer = self.generator.generate(pos_prompt)[0]

                    position_history[pos_id].extend(current_pos_messages)
                    position_history[pos_id].append({"role": "assistant", "content": pos_answer})

                    if log_list is not None:
                        log_list.append(f"\n[{pos_id} Prompt]:\n{pos_prompt}\n")
                        log_list.append(f"[{pos_id} Answer]:\n{pos_answer}\n")


            else:
                last_round_answers = {}
                for ans in candidate_answers:
                    pos_id = ans["position_id"]
                    last_round_answers[pos_id] = position_history[pos_id][-1]["content"]

                for ans in candidate_answers:
                    pos_id = ans["position_id"]
                    my_last_answer = last_round_answers[pos_id]


                    other_answers_text = ""
                    for other_pos_id, other_ans in last_round_answers.items():
                        if other_pos_id != pos_id:
                            other_answers_text += f"{other_pos_id}'s Last Answer:\n{other_ans}\n\n"


                    refute_user_msg = {
                        "role": "user",
                        "content": (
                            f"You are a participant. Based on your Relevant Documents, and also considering the other participants' last answers and your own last answer, "
                            "provide your final answer to the question and explain the reasoning. "
                            "Output in the format: Answer: <your answer> Reason: <your reasoning>. "
                            "Do not add extra irrelevant content.\n\n"
                            f"Question:\n{sub_query}\n\n"
                            f"Your Last Answer:\n{my_last_answer}\n\n"
                            f"Other Participants' Last Answers:\n{other_answers_text}\n\n"
                        )
                    }

                    position_history[pos_id].append(refute_user_msg)
                    refute_prompt = self.prompt_template.get_string(messages=position_history[pos_id])
                    refute_answer = self.generator.generate(refute_prompt)[0]
                    position_history[pos_id].append({"role": "assistant", "content": refute_answer})

                    if log_list is not None:
                        log_list.append(f"\n[{pos_id} Refutation Prompt]:\n{refute_prompt}\n")
                        log_list.append(f"[{pos_id} Refutation Answer]:\n{refute_answer}\n")

            all_positions_text = ""
            for ans in candidate_answers:
                pos_id = ans["position_id"]
                final_pos_answer = position_history[pos_id][-1]["content"]
                all_positions_text += f"{pos_id}:\n{final_pos_answer}\n\n"

            judge_prompt = self.prompt_template.get_string(messages=[
                {
                    "role": "system",
                    "content": (
                        "Your task is to output the EXACT, MINIMAL final answer based on the views. "
                        "Follow these rules strictly:\n"
                        "1. Output ONLY the core answer - NO extra words, NO explanations, NO markdown, NO context from the question.\n"
                        "2. Do not add any punctuation unless it is part of the answer itself.\n"
                        "If the correct answer to the original question cannot be determined, output 'unknown'."
                    )
                },
                {
                    "role": "user",
                    "content":
                        f"Question:\n{sub_query}\n\n"
                        f"{all_positions_text}"
                }
            ])

            judge_answer = self.generator.generate(judge_prompt)[0].strip()
            if log_list is not None:
                log_list.append(f"\n[Judge Prompt]:\n{judge_prompt}\n")
                log_list.append(f"[Judge Decision]: {judge_answer}\n")


            if prev_judge_answer is not None:
                if self._final_consistency(prev_judge_answer, judge_answer):
                    print("Answer stabilized, stopping early")

                    final_answer_pool.append({
                        "sub_query": sub_query,
                        "source": "external",
                        "evidence_id": "judge_direct",
                        "evidence": judge_answer,
                        "document": None,
                        "doc_id": None,
                        "reason": f"Stable answer after {round_id + 1} rounds of debate"
                    })
                    return final_answer_pool

            prev_judge_answer = judge_answer


        final_answer_pool.append({
            "sub_query": sub_query,
            "source": "external",
            "evidence_id": "judge_final",
            "evidence": prev_judge_answer,
            "document": None,
            "doc_id": None,
            "reason": "Final answer after max rounds of debate"
        })
        return final_answer_pool

    def second_stage_fusion(self, item, filtered_results, con_answer, con_evidence_text, log_list=None):
        question = item.question
        docs = filtered_results["original_query"]["docs"]

        doc_text = "\n\n".join(d["contents"] for d in docs)
        if log_list is not None:
            log_list.append(f"\n{'=' * 60}\n")
            log_list.append(f"[Second-stage Debate - Global Question]\n")
            log_list.append(f"{'=' * 60}\n")

        prev_judge_answer = None


        pro_messages = []
        con_messages = []

        for round_id in range(3):
            print(f"\n===== Second Stage Round {round_id + 1} =====")
            if log_list is not None:
                log_list.append(f"\n--- Second Stage Round {round_id + 1} ---\n")
            if round_id == 0:

                pro_system_msg = {
                    "role": "system",
                    "content":
                        "First, clearly state the answer to the question, "
                        "then explain why this answer is correct based on the provided Relevant Documents. "
                        "Do not add extra content, focus on the answer and evidence-based explanation."
                }
                pro_user_msg = {
                    "role": "user",
                    "content": (
                        f"Original Question:\n{question}\n\n"
                        f"Relevant Documents:\n{doc_text}\n\n"
                    )
                }


                current_pro_messages = [pro_system_msg, pro_user_msg]
                pro_prompt = self.prompt_template.get_string(messages=current_pro_messages)
                pro_answer = self.generator.generate(pro_prompt)[0]
                print(pro_answer)


                pro_messages.extend(current_pro_messages)
                pro_messages.append({"role": "assistant", "content": pro_answer})

                if log_list is not None:
                    log_list.append(f"\n[Proponent Prompt]:\n{pro_prompt}\n")
                    log_list.append(f"[Proponent Answer]:\n{pro_answer}\n")


                con_system_msg = {
                    "role": "system",
                    "content":
                        "First, clearly state the answer to the question, "
                        "then explain why this answer is correct based on the provided verified evidence. "
                        "Do not add extra content, focus on the answer and evidence-based explanation."
                }
                con_user_msg = {
                    "role": "user",
                    "content":
                        f"Question:\n{question}\n\n"
                        f"Answer to the question:\n{con_answer}\n\n"
                        f"Verified Evidences to support the answer:\n{con_evidence_text}"
                }


                current_con_messages = [con_system_msg, con_user_msg]
                con_prompt = self.prompt_template.get_string(messages=current_con_messages)
                con_answer = self.generator.generate(con_prompt)[0]


                con_messages.extend(current_con_messages)
                con_messages.append({"role": "assistant", "content": con_answer})

                if log_list is not None:
                    log_list.append(f"\n[Opponent Prompt]:\n{con_prompt}\n")
                    log_list.append(f"[Opponent Answer]:\n{con_answer}\n")
            else:

                last_con_answer = con_answer
                last_pro_answer = pro_answer

                con_refute_user_msg = {
                    "role": "user",
                    "content": (
                        f"You are a participant. Based on your verified evidences, and also considering another participant's last answer and your own last answer, "
                        "provide your final answer to the question and explain the reasoning. "
                        "Output in the format: Answer: <your answer> Reason: <your reasoning>. "
                        "Do not add extra irrelevant content.\n\n"
                        f"Question:\n{question}\n\n"
                        f"Another participant's Last Answer:\n{last_pro_answer}\n\n"
                        f"Your Last Answer:\n{last_con_answer}\n\n"
                    )
                }


                con_messages.append(con_refute_user_msg)
                con_refute_prompt = self.prompt_template.get_string(messages=con_messages)
                con_answer = self.generator.generate(con_refute_prompt)[0]
                con_messages.append({"role": "assistant", "content": con_answer})

                if log_list is not None:
                    log_list.append(f"\n[Opponent Refutation Prompt]:\n{con_refute_prompt}\n")
                    log_list.append(f"[Opponent Refutation Answer]:\n{con_answer}\n")


                pro_response_user_msg = {
                    "role": "user",
                    "content": (
                        f"You are a participant. Based on the original documents, and also considering another participant's last answer and your own last answer, "
                        "provide your final answer to the question and explain the reasoning. "
                        "Output in the format: Answer: <your answer> Reason: <your reasoning>. "
                        "Do not add extra irrelevant content.\n\n"
                        f"Question:\n{question}\n\n"
                        f"Your Last Answer:\n{last_pro_answer}\n\n"
                        f"Another participant's Last Answer:\n{last_con_answer}\n\n"
                    )
                }


                pro_messages.append(pro_response_user_msg)
                pro_response_prompt = self.prompt_template.get_string(messages=pro_messages)
                pro_answer = self.generator.generate(pro_response_prompt)[0]
                pro_messages.append({"role": "assistant", "content": pro_answer})

                if log_list is not None:
                    log_list.append(f"\n[Proponent Response Prompt]:\n{pro_response_prompt}\n")
                    log_list.append(f"[Proponent Response Answer]:\n{pro_answer}\n")


            judge_prompt = self.prompt_template.get_string(messages=[
                {
                    "role": "system",
                    "content": (
                        "Your task is to output the EXACT, MINIMAL final answer based on the views. "
                        "Follow these rules strictly:\n"
                        "1. Output ONLY the core answer - NO extra words, NO explanations, NO markdown, NO context from the question.\n"
                        "2. Do not add any punctuation unless it is part of the answer itself.\n"
                        "If the correct answer to the original question cannot be determined, output 'unknown'."
                    )
                },
                {
                    "role": "user",
                    "content":
                        f"Question:\n{question}\n\n"
                        f"View One:\n{pro_answer}\n\n"
                        f"View Two:\n{con_answer}"
                }
            ])

            judge_answer = self.generator.generate(judge_prompt)[0]
            if log_list is not None:
                log_list.append(f"\n[Judge Prompt]:\n{judge_prompt}\n")
                log_list.append(f"[Judge Decision]: {judge_answer}\n")


            if prev_judge_answer is not None:
                if self._final_consistency(prev_judge_answer, judge_answer):
                    print("Answer stabilized, stopping early")

                    invalid_keywords = ["no answer", "no valid evidence", "unknown", "no valid answer"]
                    if any(keyword in judge_answer.lower() for keyword in invalid_keywords):
                        print(f"[Fallback] Invalid answer detected ({judge_answer}), forcing the use of internal knowledge...")
                        fallback_prompt_messages = [
                            {
                                "role": "system",
                                "content": "Answer the question using your internal knowledge. Output ONLY the final answer here - NO reasoning words, NO explanations, NO markdown, NO extra text."
                            },
                            {
                                "role": "user",
                                "content": f"Question: {question}"
                            }
                        ]
                        fallback_prompt = self.prompt_template.get_string(messages=fallback_prompt_messages)
                        judge_answer = self.generator.generate(fallback_prompt)[0].strip()
                    return judge_answer

            prev_judge_answer = judge_answer


        invalid_keywords = ["no answer", "no valid evidence", "unknown", "no valid answer"]
        if any(keyword in prev_judge_answer.lower() for keyword in invalid_keywords):
            print(f"[Fallback] Invalid answer detected ({prev_judge_answer}), forcing the use of internal knowledge...")
            fallback_prompt_messages = [
                {
                    "role": "system",
                    "content": "Answer the question using your internal knowledge. Output ONLY the final answer here - NO reasoning words, NO explanations, NO markdown, NO extra text."
                },
                {
                    "role": "user",
                    "content": f"Question: {question}"
                }
            ]
            fallback_prompt = self.prompt_template.get_string(messages=fallback_prompt_messages)
            prev_judge_answer = self.generator.generate(fallback_prompt)[0].strip()
        return prev_judge_answer

    def _final_consistency(self, ans1, ans2):
        prompt = self.prompt_template.get_string(messages=[
            self._final_judge_system(),
            {
                "role": "user",
                "content":
                    f"Answer A:\n{ans1}\n\n"
                    f"Answer B:\n{ans2}"
            }
        ])

        result = self.generator.generate(prompt)[0].strip()
        result = self._process_judge_output(result)
        return result == "SAME"

    def _sequential_sub_query_replace_system_message(self):
        return {
            "role": "system",
            "content": (
                "You are a professional query rephrasing expert for sequential question decomposition.\n\n"
                "Core Task:\n"
                "1. You will receive a CURRENT sub-question, and the ANSWERS to PREVIOUS sub-questions in the sequential decomposition.\n"
                "2. Rephrase the CURRENT sub-question to incorporate the answers of previous sub-questions (synonymous replacement).\n"
                "3. Ensure the rephrased question has the SAME semantic meaning as the original, but is self-contained (no dependency on previous questions).\n\n"
                "Strict Rules:\n"
                "- ONLY output the rephrased sub-question (one line, no extra text, no explanations).\n"
                "- Do NOT change the core intent of the original sub-question.\n"
                "- Do NOT add new information beyond the previous answers.\n"
                "- If no rephrasing is needed (already self-contained), output the original sub-question verbatim.\n\n"
                "Examples:\n"
                "Previous Sub-questions & Answers:\n- What is the capital of France?: Paris\n"
                "Current Sub-question: What is the population of this city?\n"
                "Rephrased Output: What is the population of Paris?\n\n"
                "Previous Sub-questions & Answers:\n- Who wrote 'Hamlet?: William Shakespeare\n"
                "Current Sub-question: When was this author born?\n"
                "Rephrased Output: When was William Shakespeare born?"
            )
        }

    def _document_filter_system_message(self):
        return {
            "role": "system",
            "content": (
                "You are a document filtering agent for a retrieval-augmented generation (RAG) system.\n"
                "Your task is to judge if the document contains FACTS that directly/indirectly support answering the question.\n\n"
                "Decision Rules (MUST FOLLOW):\n"
                "1. Output Yes: If document has ANY factual info (statements/attributes/contexts) related to the core entity/topic of the question.\n"
                "2. Output No: ONLY if document has NO overlap with the core entity/topic of the question (completely irrelevant).\n\n"
                "Strict Output Rules:\n"
                "- Exactly one word: Yes or No (capitalized first letter)\n"
                "- No explanations/extra text, base judgment ONLY on the document content (no hallucination).\n\n"
                "Examples:\n"
                "Question: When was Iron Man created?\n"
                "Document 1 (Yes): Iron Man was created by Stan Lee in 1963.\n"
                "Document 2 (Yes): Stan Lee, the creator of Iron Man, was born in 1922.\n"
                "Document 3 (No): Spider-Man is a fictional character created by Stan Lee and Steve Ditko.\n\n"
                "Question: What is the boiling point of water?\n"
                "Document 1 (Yes): Water boils at 100°C at standard atmospheric pressure.\n"
                "Document 2 (No): The freezing point of ethanol is -114°C.\n"
            )
        }

    def _final_judge_system(self):
        return {
            "role": "system",
            "content":
                "Judge whether two answers have the same meaning.\n"
                "Output SAME or DIFFERENT only."
        }

    def _process_judge_output(self, raw_output):

        if not raw_output:
            return "DIFFERENT"


        processed = raw_output.strip().upper()


        if "SAME" in processed:
            return "SAME"
        elif "DIFFERENT" in processed:
            return "DIFFERENT"
        else:

            print(f"Warning: abnormal final judgment output, raw content: {raw_output}, returning DIFFERENT by default")
            return "DIFFERENT"

    def generate_final_answer_with_llm(self, original_question, final_answer_pool):

        structured_evidence_text = ""
        if not final_answer_pool:
            final_answer = "No valid evidence found to answer the question."
        else:

            sub_query_evidence_map = defaultdict(list)
            for item in final_answer_pool:
                sq = item["sub_query"]
                sub_query_evidence_map[sq].append(item)


            structured_evidence_text = ""
            for sq_idx, (sub_q, evidences) in enumerate(sub_query_evidence_map.items(), 1):
                structured_evidence_text += f"Sub-question {sq_idx}: {sub_q}\n\n"
                for ev_idx, ev_item in enumerate(evidences, 1):
                    source = "Internal Knowledge" if ev_item["source"] == "internal" else "External Document"
                    doc_id = ev_item["doc_id"] or "N/A"
                    content = ev_item["evidence"]

                    structured_evidence_text += (

                        f"  - candidate answer {sq_idx}.{ev_idx} (Source: {source}, Doc ID: {doc_id}):\n"
                        f"    {content}\n\n"
                    )


            prompt_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional answer synthesis agent. You will be given an original question, "
                        "a list of decomposed sub-questions, and verified candidate answers corresponding to each sub-question.\n\n"
                        "Core Rules:\n"
                        "1. **No Hallucination**: You MUST NOT use any knowledge outside of the provided candidate answers.\n"
                        "2. **Completeness**: Synthesize information from ALL relevant sub-questions to form a comprehensive answer.\n\n"
                        "【CRITICAL】Output Format (MUST STRICTLY FOLLOW - NO DEVIATION ALLOWED):\n"
                        "```\n"
                        "[Reasoning Process]\n"
                        "1. Analysis of the original question: <Your analysis>\n"
                        "2. Sufficiency check of sub-questions: <Check if all sub-questions are covered>\n"
                        "3. Conflict resolution (if any): <How you resolve conflicts>\n"
                        "\n"
                        "[Final Answer]\n"
                        "<Output ONLY the final answer here - NO reasoning words, NO explanations, NO markdown, NO extra text>\n"
                        "```\n"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"## Original Question\n{original_question}\n\n"
                        f"## Decomposed Sub-questions & Verified Candidate Answers\n"
                        f"{structured_evidence_text}\n"
                        f"Please follow the output format EXACTLY to provide the reasoning process and final answer. Do NOT modify any section titles or add extra content outside the specified structure."
                    )
                }
            ]


            prompt = self.prompt_template.get_string(messages=prompt_messages)
            raw_output = self.generator.generate(prompt)[0].strip()
            print(raw_output)


            final_answer = self._extract_final_answer(raw_output)

        return final_answer, structured_evidence_text

    def _extract_final_answer(self, raw_output):

        import re


        cleaned_output = raw_output.strip()
        if not cleaned_output:
            return "No valid answer generated."


        patterns = [

            r"\*\*Final Answer\*\*\s*(.*?)(?:\n\n|$)",

            r"\[Final Answer\]\s*(.*?)(?:\n\n|$)",

            r"Final Answer\s*:\s*(.*?)(?:\n\n|$)",

            r"Final Answer\s*\n\s*(.*?)(?:\n\n|$)",
        ]


        for pattern in patterns:

            matches = re.findall(pattern, cleaned_output, re.DOTALL | re.IGNORECASE)
            if matches:

                answer = matches[-1].strip()

                answer = re.sub(r"^```|```$", "", answer).strip()

                if answer:
                    return answer


        paragraphs = [p.strip() for p in cleaned_output.split("\n\n") if p.strip()]
        if paragraphs:
            return paragraphs[-1]


        return "No valid answer found in the response."


    def __del__(self):
        if hasattr(self, "qa_file") and self.qa_file is not None:
            self.qa_file.write("\n========== Log End ==========\n")
            self.qa_file.close()
            print("Log file closed safely.")

    def _format_filtered_results(self, filtered_results):

        lines = []
        orig_query = filtered_results.get("original_query", {}).get("query", "")
        orig_docs = filtered_results.get("original_query", {}).get("docs", [])
        lines.append(f"Original Question: {orig_query}")
        lines.append(f"Filtered documents retained: {len(orig_docs)}")
        for i, doc in enumerate(orig_docs, 1):
            doc_content = doc.get('contents', '')
            doc_id = doc.get('id', f'doc_{i}')
            lines.append(f"\nDocument {i} (ID: {doc_id}):")
            lines.append(f"{doc_content}")
            lines.append(f"-" * 50)
        return "\n".join(lines)

    def judge_question_type(self, input_query):

        type_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a question type classification expert. You must return ONLY one of these keywords: parallel/serial/none\n"
                    "Classification Rules (MUST FOLLOW):\n"
                    "1. parallel: The question requires ≥2 independent facts, sub-questions have NO dependencies (can be answered at the same time).\n"
                    "   Example: Are Iron Man and Batman both billionaires?\n"
                    "2. serial: The question requires facts to be obtained in order — LATER SUB-QUESTIONS CAN ONLY BE ANSWERED WITH THE ANSWER OF EARLIER ONES.\n"
                    "   Example: Who is the author of the book that won the 2023 Nobel Prize in Literature?\n"
                    "3. none: The question only needs one fact to answer, no need to decompose.\n"
                    "   Example: What has David Bowie done in The Lodge?\n"
                    "Note: Return only the keyword, no extra words, no explanations."
                )
            },
            {"role": "user", "content": f"Question: {input_query}"}
        ]

        prompt_str = self.prompt_template.get_string(messages=type_prompt)
        try:
            type_result = self.generator.generate(prompt_str)[0].strip().lower()
            if type_result not in ["parallel", "serial", "none"]:
                type_result = "none"
            return type_result
        except Exception as e:
            return "none"

    def _decompose_parallel(self, input_query, idx):

        parallel_prompt = [
            self._parallel_decompose_system_message(),
            {"role": "user", "content": f"Question: {input_query}"}
        ]
        prompt_str = self.prompt_template.get_string(messages=parallel_prompt)
        output = self.generator.generate(prompt_str)[0]
        sub_queries = [q.strip() for q in output.split("\n") if q.strip()]
        return sub_queries

    def _decompose_serial(self, input_query, idx):

        serial_prompt = [
            self._serial_decompose_system_message(),
            {"role": "user", "content": f"Question: {input_query}"}
        ]
        prompt_str = self.prompt_template.get_string(messages=serial_prompt)
        output = self.generator.generate(prompt_str)[0]
        sub_queries = [q.strip() for q in output.split("\n") if q.strip()]
        return sub_queries

    def _parallel_decompose_system_message(self):

        return {
            "role": "system",
            "content": (
                "You are a parallel question decomposition expert. Decompose the question into COMPLETELY INDEPENDENT sub-questions.\n"
                "Core Rules:\n"
                "1. Max 2-3 sub-questions, each requires only one independent fact.\n"
                "2. Sub-questions have NO dependencies — each can be answered without knowing the answer of others.\n"
                "3. Output format: One sub-question per line, no numbering/explanations.\n"
                "Example:\n"
                "Original Question: What is the population of Beijing and the GDP of Shanghai in 2023?\n"
                "Output:\nWhat is the population of Beijing in 2023?\nWhat is the GDP of Shanghai in 2023?"
            )
        }

    def _serial_decompose_system_message(self):

        return {
            "role": "system",
            "content": (
                "You are a serial question decomposition expert. Decompose the question into DEPENDENT sub-questions (strict order).\n"
                "Core Rules (MUST FOLLOW):\n"
                "1. Max 2-3 sub-questions, LATER SUB-QUESTIONS CAN ONLY BE ANSWERED IF YOU KNOW THE ANSWER OF EARLIER ONES.\n"
                "2. Each sub-question requires only one fact, ordered by the necessary answering sequence.\n"
                "3. No redundant/repeated sub-questions, and avoid irrelevant steps.\n"
                "4. Sub-questions must strictly match the core attribute of the original question.\n"
                "5. Only use serial decomposition for attribute query questions (do not mislabel as parallel).\n"
                "6. Use accurate interrogative words and clear expressions (e.g., 'Who' for people, 'What' for objects) with no grammatical errors.\n"
                "7. Output format: One sub-question per line, no numbering/explanations.\n"
                "Example (critical: second sub-question depends on first):\n"
                "Original Question: Who is the author of the book that won the 2023 Nobel Prize in Literature?\n"
                "Output:\nWhich book won the 2023 Nobel Prize in Literature?\nWho is the author of this book?"
            )
        }


    def _answer_generation_system_message(self):

        return {
            "role": "system",
            "content": (
                "You are a document-based QA agent. Follow these rules STRICTLY:\n"
                "1. Answer the sub-question ONLY using the content of the provided document.\n"
                "2. If the document does NOT contain enough information to answer the question, output EXACTLY 'unknown' (all lowercase).\n"
                "3. If you can answer, output ONLY the core answer (no extra explanation, no reasoning, no markdown).\n"
                "4. Do NOT add any punctuation unless it is part of the answer itself.\n"
            )
        }

    def _internal_knowledge_system_message(self):
        return {
            "role": "system",
            "content": (
                "You are an internal knowledge agent of a large language model.\n\n"
                "Your task:\n"
                "- Answer the given sub-question using ONLY your internal (parametric) knowledge.\n\n"
                "Rules (STRICT):\n"
                "- If you are confident you know the answer, output:\n"
                "  Answer: <your answer>\n"
                "  Reason: <brief justification>\n"
                "- If you are NOT confident or do not know, output EXACTLY:\n"
                "  unknown\n\n"
                "Constraints:\n"
                "- Do NOT assume access to documents or external sources\n"
                "- Do NOT hallucinate\n"
                "- Be concise and factual\n"
            )
        }

    def _process_llm_answer_output(self, llm_raw_output: str):

        cleaned_answer = llm_raw_output.strip()


        judge_text = cleaned_answer.lower()


        import string
        judge_text = judge_text.rstrip(string.punctuation).strip()

        unknown_keywords = [
            "unknown",
            "i don't know",
            "i do not know",
            "no answer",
            "no valid answer",
            "cannot answer",
            "unable to answer"
        ]


        is_unknown = any(keyword in judge_text for keyword in unknown_keywords)

        return is_unknown

    def _append_log_section(self, log_list, title, content):
        log_list.append(f"\n{'=' * 80}\n")
        log_list.append(f"【{title}】\n")
        log_list.append(f"{'=' * 80}\n")
        log_list.append(content if content else "No content\n")
