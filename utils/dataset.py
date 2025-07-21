import pdstools.infinity.client as client
import json
from json import JSONDecodeError
import ast
from abc import abstractmethod
import os
import tqdm
import asyncio
import aiofiles
from collections import defaultdict
from utils.generator import Generator
import numpy as np
from ragas import evaluate, EvaluationDataset
import yaml
import re
from pdstools.infinity.resources.knowledge_buddy.knowledge_buddy import BuddyResponse
import ast
import itertools
import time
import random

class MultiDocDataSet:
    def __init__(self,
                name: str,
                experiment_name: str,
                **kwargs):
        project_root = os.environ.get("PROJECT_ROOT")
        self.data_path = os.path.join(project_root, "data", name)
        self.output_path = os.path.join(project_root, "output", name, experiment_name)
        self.result_path = os.path.join(self.output_path, "results")

        self.data = json.load(
            open(
                os.path.join(self.data_path, "corpus.json")
            )
        )

        self.kwargs=kwargs
        self.data_type = "NODATALOADED"
        # self.get_descriptives(save=True, **kwargs)
        self.name=name

    async def process_question(self, question, reference=None, **kwargs):
        out = await self.knowledge_buddy_client.knowledge_buddy.question_async(
            question=question,
            buddy=self.buddy_name,
            include_search_results=self.include_search_results
        )
        if type(out) != BuddyResponse:
            print("Logging error...")
            return {
                "response" : "error",
                "user_input" : question,
                "reference": reference,
                "retrieved_contexts": None,
                "status_code": out.status_code,
                # "error_details": ast.literal_eval(out.content.decode("utf8"))
            }

        #FUTURE WORK - add optional JSON parser for JSON output
        answer = out.answer
        result = {"response" : answer if type(answer)==str else str(answer),
                "user_input" : question,
                "reference" : reference}
        
        if "don't know" in str(answer).lower() or "I do not have enough information" in str(answer).lower() or out.status != "Results found":
            #FUTURE WORK - add better way of recognizing rejected answers
            result["response"] = "rejected"

        if self.include_search_results:
            if out.status == "Results found":
                N_retrieved = len(ast.literal_eval(out.search_results[0].value))
                retrieved_context = [ast.literal_eval(out.search_results[0].value)[i]['content'] for i in range(N_retrieved)]
            else:
                retrieved_context = []
            result['retrieved_contexts'] = retrieved_context  

        return result
    
    async def predict_and_save(self, 
                               knowledge_buddy_client,
                               buddy_name,
                               include_search_results=True,
                               allow_overwrite=False,
                               batching=None):
        """
        This method generates answers for the currently loaded questions by prompting the Buddy {buddy_name} through the knowledge_buddy_client.
        Use the data_type argument to specify which questions are currently loaded.
        """
        # Set prediction args
        self.include_search_results = include_search_results
        self.knowledge_buddy_client = knowledge_buddy_client
        self.buddy_name = buddy_name

        # Set data type
        if self.data_type == "NODATALOADED":
            print("Warning: No data type specified / no data loaded.")
            raise Warning
        
        self.predictions = {}
        self.rejections = {}
        self.errors = {}
        save_path = os.path.join(self.output_path, self.data_type, buddy_name)
        os.makedirs(save_path, exist_ok=allow_overwrite)

        async def run(questions, start_id, end_id):
            tasks = [self.process_question(**qa) for qa in questions]
            batch_doc_results = await tqdm.asyncio.tqdm.gather(*tasks, desc=f"Processing Questions {start_id}-{end_id}", total=len(tasks))
            return batch_doc_results

        doc_results = []

        # Optional: split up in batches and wait inbetween to prevent overloading the API resulting in errors.
        if batching:
            for i in range(0,len(self.questions), batching):
                end = min(i+batching, len(self.questions))
                questions = self.questions[i:end]
                doc_results += await run(questions=questions, start_id=i, end_id=end)
                time.sleep(5)
        else:
            doc_results = await run(questions=self.questions, start_id=0, end_id=len(self.questions))
    
        doc_rejections = [doc_result for doc_result in doc_results if doc_result["response"]=="rejected"]
        doc_answers = [doc_result for doc_result in doc_results if doc_result["response"] not in ["rejected", "error"]]
        doc_errors = [doc_result for doc_result in doc_results if doc_result["response"]=="error"]
        
        self.predictions= doc_answers
        self.rejections = doc_rejections
        self.errors = doc_errors

        async with aiofiles.open(save_path + f"/answers.json", "w", encoding="utf-8") as f:
            await f.write(json.dumps(doc_answers, indent=3, ensure_ascii=False))

        async with aiofiles.open(save_path + f"/rejections.json", "w", encoding="utf8") as f:
            await f.write(json.dumps(doc_rejections, indent=3, ensure_ascii=False))

        async with aiofiles.open(save_path + f"/errors.json", "w", encoding="utf8") as f:
            await f.write(json.dumps(doc_errors, indent=3, ensure_ascii=False))

        # retry errors after a bit of time to circumvent rate limits
        if len(doc_errors) > 0:
            print("Some questions raised (internal) errors, possibly due to rate limits. We retry asking these questions after a short wait of 20s...")
            time.sleep(20)
            error_questions = [{
                "question" : doc['user_input'],
                "reference": doc['reference']
            } for doc in doc_errors]

            error_doc_results = await run(error_questions, start_id=0, end_id=len(error_questions))

            doc_rejections += [doc_result for doc_result in error_doc_results if doc_result["response"]=="rejected"]
            doc_answers += [doc_result for doc_result in error_doc_results if doc_result["response"] not in ["rejected", "error"]]
            doc_errors = [doc_result for doc_result in error_doc_results if doc_result["response"]=="error"]

        self.predictions= doc_answers
        self.rejections = doc_rejections
        self.errors = doc_errors

        async with aiofiles.open(save_path + f"/answers.json", "w", encoding="utf-8") as f:
            await f.write(json.dumps(doc_answers, indent=3, ensure_ascii=False))

        async with aiofiles.open(save_path + f"/rejections.json", "w", encoding="utf8") as f:
            await f.write(json.dumps(doc_rejections, indent=3, ensure_ascii=False))

        async with aiofiles.open(save_path + f"/errors.json", "w", encoding="utf8") as f:
            await f.write(json.dumps(doc_errors, indent=3, ensure_ascii=False))


    # @abstractmethod
    def load_human_questions(self, eval_set_name="human", use_doc_ids=False, N=None):
        """Loads human questions from a JSON file.
        - use_doc_ids: set this to True if your questions are structured per document in a nested json file.
        - N: Set a subset amount of questions.
        """
        with open(os.path.join(self.data_path, eval_set_name, "questions.json"), "r", encoding="utf8") as f:
            self.questions = sum(list(json.load(f).values()), []) if use_doc_ids else json.load(f)
        
        if N:
            assert N <= len(self.questions), f"N is larger than the number of questions ({N} > {len(self.questions)}). Consider specifying `num_human_questions` in the config file."
            self.questions = self.questions[:N]
        
        self.data_type = "Human"

    # @abstractmethod
    # def load_contexts(self):
    #     self.contexts = self.data
    #     pass

    def init_generator(self, 
                       prompt:str,
                       prompt_keys:list,
                       **kwargs):
        """
        Initialize the generator to generate questions based on the corpus.
        The prompt may include {variables}. All variable names should be specified in the prompt_keys list.
        Kwargs may include:

            Seed = 0
            Temperature = 0
            top_p = 0.95
            max_tokens = 800

        For more options, refer to the AzureChatOpenAI documentation.
        """
        self.generator = Generator(
            prompt=prompt,
            prompt_keys=prompt_keys,
            **kwargs
        )

    def generate_and_save(self, 
                          prompt_args:dict,
                          num_source_docs: int,
                          data_type:str="NODATATYPE_GENERATED",
                          allow_overwrite=False,
                          min_doc_length=0):
        """
        Generates and saves data based on the context documents. Questions are saved in the structure "{doc_id : [question 1, question 2, ...]}"
        """
        # self.load_contexts()
        questions = defaultdict(lambda:[])
        save_path = os.path.join(self.data_path, data_type)
        os.makedirs(save_path, exist_ok=allow_overwrite)
        if 'num_questions_per_document' in prompt_args.keys():
            print(f"Generating {prompt_args['num_questions_per_document'] * num_source_docs} questions from {num_source_docs} documents with at least {min_doc_length} words...")


        # SAMPLING DOCUMENTS
        # Step 1: Filter document IDs by length
        eligible_doc_ids = [doc_id for doc_id, text in self.data.items() if len(text.split()) >= min_doc_length]
        # Step 2: Check if enough documents are available
        if len(eligible_doc_ids) < num_source_docs:
            raise ValueError(
                f"Not enough documents with at least {min_doc_length} words. "
                f"Found {len(eligible_doc_ids)}, but need {num_source_docs}."
            )
        # Step 3: Sample from eligible documents
        doc_ids = random.sample(eligible_doc_ids, num_source_docs)

        # GENERATE QUESTIONS
        for doc_id in tqdm.tqdm(doc_ids):
            doc = self.data[doc_id]
            prompt_args["context"] = doc
            assert set(self.generator.prompt_keys) == set(prompt_args.keys()), f"prompt_args should contain {self.generator.prompt_keys}. Received {list(prompt_args.keys())}"
            output = self.generator.chain.invoke(prompt_args).content
            try:
                questions[doc_id] = json.loads(output)
            except:
                try:
                    output = output.replace("\\", "\\\\")
                    questions[doc_id] = json.loads(output[7:-3])
                except:
                    print(f"No json output created: {output}")
                    questions[doc_id] = output


        with open(os.path.join(save_path, "questions.json"), "w", encoding="utf8") as f:
            json.dump(questions, f, indent=2)

        print(f"Questions generated and saved to {save_path}/questions.json")

    def load_generated_questions(self, data_type: str, use_doc_ids: bool = True):
        """Loads generated questions from a JSON file."""
        question_path = os.path.join(self.data_path, data_type, "questions.json")
        
        with open(question_path, "r", encoding="utf8") as f:
            questions = json.load(f)  # {doc_id: [question 1, question 2, ...]}

        self.data_type = data_type

        if use_doc_ids:
            self.questions = [q for q_list in questions.values() for q in q_list]  # Flatten list
            self.source_doc_ids = list(questions.keys())
        else:
            self.questions = questions

        return self.questions


    def get_descriptives(self, save=True, **kwargs):
        N = len(self.data)
        doc_sizes = [len(doc.split(" ")) for doc in self.data.values()]
        descriptives = {
            "N documents" : N,
            "mean size (tokens)" : np.mean(doc_sizes),
            "size SD" : np.std(doc_sizes),
            "minimum size (tokens)" : np.min(doc_sizes),
            "maximum size (tokens)" : np.max(doc_sizes)
        }
        descriptives.update(kwargs)
        descriptives = {k: (v.item() if isinstance(v, (np.int64, np.float64)) else v) for k, v in descriptives.items()}
        if save:
            os.makedirs(self.result_path, exist_ok=True)
            with open(self.result_path + "/descriptives.json", "w") as f:
                json.dump(descriptives, f, indent=3)
            print(f"saved descriptives to {self.result_path}")
        self.descriptives = descriptives
        return descriptives

    def get_question_descriptives(self):
        if not self.questions:
            raise RuntimeError("No questions loaded")
        questions = [qa["question"] for qa in self.questions]
        answers = [qa["reference"] for qa in self.questions]
        N = len(questions)
        question_sizes = [len(q.split(" ")) for q in questions]
        answer_sizes = [len(a.split(" ")) for a in answers]
        descriptives= {
            "N questions" : N,
            "Mean question length" : np.mean(question_sizes),
            "SD question length" : np.std(question_sizes),
            "Shortest question": np.min(question_sizes),
            "Longest question": np.max(question_sizes),
            "Mean answer length" : np.mean(answer_sizes),
            "SD answer length" : np.std(answer_sizes),
            "Shortest answer": np.min(answer_sizes),
            "Longest answer": np.max(answer_sizes),
        }
        descriptives = {k:float(v) for k,v in descriptives.items()}
        save_path = os.path.join(self.output_path, self.data_type, "question_descriptives.json")
        os.makedirs(os.path.join(self.output_path, self.data_type), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(descriptives, f, indent=3)
        return descriptives

    def _load_subset(self, buddy_name:str, file_name:str):
        dataset=[]
        self.predictions_path = os.path.join(self.output_path, self.data_type, buddy_name)
        for f in os.listdir(self.predictions_path):
            if f.startswith(file_name):
                with open(os.path.join(self.predictions_path, f), 'r', encoding='utf8') as f:
                    dataset+=json.load(f)
        dataset = EvaluationDataset.from_list(dataset)
        print(f"Loaded {buddy_name} {file_name} and updated predictions path")

        return dataset

    def load_evaluation_dataset(self, buddy_name):
        """
        Load the evaluation dataset of {buddy_name}'s predictions in the current experiment.
        """
        self.evaluation_dataset=self._load_subset(buddy_name=buddy_name, file_name="answers")
        self.predictions_path = os.path.join(self.output_path, self.data_type, buddy_name)
        return self.evaluation_dataset

    def load_rejection_dataset(self, buddy_name):
        """
        Load the dataset of questions that were not answered by the given buddy.
        """
        self.rejection_dataset=self._load_subset(buddy_name=buddy_name, file_name="rejections")
        self.predictions_path = os.path.join(self.output_path, self.data_type, buddy_name)
        return self.rejection_dataset

    def evaluate(self, buddy_name:str, metrics:list, llm=None, embeddings=None, token_usage_parser=None, overwrite=False):
        if any([hasattr(m,"embeddings") for m in metrics]) and embeddings==None:
            raise AttributeError("No embeddings specified while including at least one metric that require embeddings.")
        if any([hasattr(m,"llm") for m in metrics]) and embeddings==None:
            raise AttributeError("No llm specified while including at least one metric that requires an llm.")

        self.load_evaluation_dataset(buddy_name=buddy_name)
        self.load_rejection_dataset(buddy_name=buddy_name)
        result_dir = os.path.join(self.predictions_path, "results")
        os.makedirs(result_dir, exist_ok=overwrite)    
        results = evaluate(
            dataset=self.evaluation_dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            token_usage_parser=token_usage_parser
        )

        answers = [self.evaluation_dataset.samples[i].response for i in range(len(self.evaluation_dataset))]
        answer_lengths = [len(a.split(" ")) for a in answers]
        mean_length = float(np.mean(answer_lengths))
        sd_length = float(np.std(answer_lengths))
        min_length = int(np.min(answer_lengths))
        max_length = int(np.max(answer_lengths))
        
        # Add 0 valued results for false negatives
        results.dataset.samples += self.rejection_dataset.samples
        false_negatives_n = len(self.rejection_dataset)
        zero_metrics = [{k:0.0 for k in results.scores[0].keys()} for _ in range(false_negatives_n)]
        results.scores += zero_metrics
        results.__post_init__()

        
        false_negative = len(self.rejection_dataset) / (len(self.evaluation_dataset)) #MOTE: since we added rejections to the evaluation dataset, this is correct.
        results._repr_dict["false_negatives"] = false_negative
        results._repr_dict["answer_length_mean"] = mean_length
        results._repr_dict["answer_length_sd"] = sd_length
        results._repr_dict["answer_length_min"] = min_length
        results._repr_dict["answer_length_max"] = max_length

        self.save_results(results, overwrite=overwrite)
        return results
    
    def save_results(self, results, overwrite=False):
        simple_results = results._repr_dict
        results_dict = results.to_pandas().to_dict()
        result_dir = os.path.join(self.predictions_path, "results")
        with open(result_dir + "/results.json", "w", encoding="utf8") as f:
            json.dump(results_dict, f, indent=3)
        with open(result_dir + "/simple_results.json", "w", encoding="utf8") as f:
            json.dump(simple_results, f, indent=3)
        print("Saved results succesfully.")

    def load_results(self, buddy_name):
        #FUTURE WORK - save/load full results as pickle objects instead so we can still calculate cost afterwards.
        self.predictions_path = os.path.join(self.output_path, self.data_type, buddy_name)
        result_dir = os.path.join(self.predictions_path, "results")
        with open(result_dir + "/results.json", "r", encoding="utf8") as f:
            results = json.load(f)
        with open(result_dir + "/simple_results.json", "r", encoding="utf8") as f:
            simple_results = json.load(f)
        print("loaded results succesfully.")
        return results, simple_results
