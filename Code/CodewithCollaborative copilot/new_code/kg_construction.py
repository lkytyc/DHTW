from itertools import combinations
from openai import OpenAI
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from tqdm import tqdm
import ast
import time

class KGConstruction:
    def __init__(self, **constructor_config) -> None:
        # API_base
        self.API_Base = constructor_config["OpenAI_API_Base"]
        # API_key
        self.key_list = constructor_config["API_key_list"]
        # Model name
        self.model_name = "gpt-4o-2024-08-06"
        # all_file_path
        self.all_file_path = constructor_config["all_file_path"]
        # prompt_path
        self.entity_extratction_prompt_path = constructor_config["kc_entity_extratction_prompt"]
        self.relation_extratction_prompt_path = constructor_config["kc_relation_extratction_prompt"]
        # type_path
        self.entity_type_path = constructor_config["save_ke_entity_type_path"]
        self.relation_type_path = constructor_config["save_ke_relation_type_path"]
        # save path
        self.entity_file_path = constructor_config["save_kc_entity_path"]
        self.relation_file_path = constructor_config["save_kc_relation_path"]
        self.save_kg_schema_path = constructor_config["save_kg_schema_path"]
        
        # Token and time statistics
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.llm_call_count = 0
        self.llm_response_times = []  # LLM response time list

    def load_message(self, prompt_content):
        instruct_content = "Please complete the task."
        message = [{"role": "system", "content": instruct_content}]
        message.append({"role": "user", "content": prompt_content})
        return message

    def call_llm(self, llm_input, api_key):
        client = OpenAI(api_key=api_key, base_url=self.API_Base)
        
        start_time = time.time()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=llm_input,
            temperature=0,
            max_tokens=4096
        )
        end_time = time.time()
        
        # Token accounting
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        
        # Accumulate statistics
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.llm_call_count += 1
        
        # Record response time
        response_time = end_time - start_time
        self.llm_response_times.append(response_time)
        
        print(f"Token stats - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}, Time: {response_time:.2f}s")
        
        return response.choices[0].message.content.strip().lower()
    
    def print_statistics(self):
        print("\n" + "="*80)
        print("KGConstruction Statistics Report")
        print("="*80)
        print("Token Statistics:")
        print(f"  - Total Prompt Tokens: {self.total_prompt_tokens:,}")
        print(f"  - Total Completion Tokens: {self.total_completion_tokens:,}")
        print(f"  - Total Tokens: {self.total_tokens:,}")
        print("\nTime Statistics:")
        print(f"  - LLM Call Count: {self.llm_call_count}")
        if self.llm_response_times:
            print(f"  - Average Response Time: {sum(self.llm_response_times)/len(self.llm_response_times):.2f}s")
            print(f"  - Fastest Response Time: {min(self.llm_response_times):.2f}s")
            print(f"  - Slowest Response Time: {max(self.llm_response_times):.2f}s")
            print(f"  - Total LLM Response Time: {sum(self.llm_response_times):.2f}s")
        print("="*80 + "\n")

    def save_statistics_to_file(self, save_path="../output/kg_construction/statistics.txt"):
        """Save statistics to a file."""
        # Ensure the output directory exists.
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("KGConstruction Statistics Report\n")
            f.write("="*80 + "\n")
            f.write("Token Statistics:\n")
            f.write(f"  - Total Prompt Tokens: {self.total_prompt_tokens:,}\n")
            f.write(f"  - Total Completion Tokens: {self.total_completion_tokens:,}\n")
            f.write(f"  - Total Tokens: {self.total_tokens:,}\n")
            f.write("\nTime Statistics:\n")
            f.write(f"  - LLM Call Count: {self.llm_call_count}\n")
            if self.llm_response_times:
                f.write(f"  - Average Response Time: {sum(self.llm_response_times)/len(self.llm_response_times):.2f}s\n")
                f.write(f"  - Fastest Response Time: {min(self.llm_response_times):.2f}s\n")
                f.write(f"  - Slowest Response Time: {max(self.llm_response_times):.2f}s\n")
                f.write(f"  - Total LLM Response Time: {sum(self.llm_response_times):.2f}s ({sum(self.llm_response_times)/60:.2f} min)\n")
            f.write("="*80 + "\n")

        print(f"Statistics saved to: {save_path}")

    def load_existing_schema(self):
        existing_schema_triples = set()
        save_kg_schema_path = self.save_kg_schema_path


        if os.path.exists(save_kg_schema_path):
            try:
                with open(save_kg_schema_path, mode='r', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader, None)
                    for row in reader:
                        if len(row) == 3:

                            existing_schema_triples.add(tuple(map(str.strip, row)))
            except Exception as e:
                print(f"Failed to read existing KG Schema: {e}")

        return existing_schema_triples

    def read_all_files(self):
        # check seed_path
        if os.path.isdir(self.all_file_path):
            files = os.listdir(self.all_file_path)
            files_list = [os.path.join(self.all_file_path, file) for file in files]
        else:
            files_list = []
            print("Invalid seed path")
        # read file_content
        file_content_list = []
        for path in files_list:
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    result = list(reader)
                    for i in range(len(result)):
                        file_content_list.append(result[i][0])
            except Exception as e:
                print(e)
        return file_content_list

    import csv
    def get_entity_type(self):
        entity_type_list = []
        with open(self.entity_type_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 3:
                    entity_type_list.append({row[0]: row[2]})
        result_dict = {}
        for item in entity_type_list:
            result_dict.update(item)

        entity_type_str = '\n'.join([f"'{k}': '{v}'" for k, v in result_dict.items()])
        return entity_type_str
    def entity_extraction(self, chunk, entity_type_str, api_key, index):
        try:
            # entity_type_str = self.get_entity_type()
            prompt_content = open(self.entity_extratction_prompt_path, 'r', encoding='utf-8').read()
            prompt_content = prompt_content.replace("${text}$", '"' + chunk + '"').replace("${entity types and their definitions}$", '\n' + entity_type_str)
            llm_input = self.load_message(prompt_content)
            entity_result = self.call_llm(llm_input, api_key)
            if ":" in entity_result:
                entities = entity_result.split("\n")[1]
            else:
                entities = ""
        except:
            entities = ""
        return index, entities

    def save_extracted_entities(self, chunk_list, entity_list, entity_pair_list, save_path):
        result = []
        for idx in range(len(chunk_list)):
            task = {
                "chunk": chunk_list[idx],
                "entities": entity_list[idx],
                "entity_pair": entity_pair_list[idx],
            }
            result.append(task)
        return result

    def process_entity_extraction(self, chunk_list, entity_type_str, save_path):
        all_result = []
        entity_list = [None] * len(chunk_list)
        entity_pair_list = [None] * len(chunk_list)
        with ThreadPoolExecutor(max_workers=len(self.key_list)) as executor:
            futures = {
                executor.submit(self.entity_extraction, chunk, entity_type_str,
                                self.key_list[idx % len(self.key_list)], idx): idx for idx, chunk in
                enumerate(chunk_list)
            }
            for future in tqdm(as_completed(futures), desc="Entity Extraction", total=len(futures)):
                try:
                    index, entities = future.result()
                    entity_list[index] = entities

                    # Safely split the entity list for better robustness.
                    entity_split_list = []
                    if entities and entities.strip():
                        try:
                            for item in entities.split("; "):
                                if item and item.strip() and ": " in item:
                                    entity_split_list.append(item.split(": ")[0].strip())
                        except Exception as e:
                            print(f"[WARNING] Failed to split entities for row {index}; skipping entity parsing for that row: {e}")

                    entity_pair_list[index] = list(combinations(entity_split_list, 2)) if entity_split_list else []
                except Exception as e:
                    print(f"[WARNING] Row processing failed; preserving raw text only for row {index}. Error: {e}")
                    entity_list[index] = ""
                    entity_pair_list[index] = []

        # Rebuild results in the original text order to keep row counts aligned.
        for idx in range(len(chunk_list)):
            result = self.save_extracted_entities(
                [chunk_list[idx]], [entity_list[idx]], [entity_pair_list[idx]], save_path
            )
            all_result.extend(result)

        df = pd.DataFrame(all_result)
        all_columns = df.columns.tolist()
        df = df[all_columns[:3]]
        df.to_csv(save_path, index=False, header=False)
        print(f"[OK] Entity extraction complete. Processed {len(chunk_list)} rows.")

    def get_relation_type(self):
        relation_type_list = []
        with open(self.relation_type_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                relation_type_list.append({row[0]: row[1]})
        result_dict = {}
        for item in relation_type_list:
            result_dict.update(item)
        relation_type_str = '\n'.join([f"'{k}': '{v}'" for k, v in result_dict.items()])
        return relation_type_str

    import ast

    def relation_extraction(self, chunk, relation_type_str, entity_pair, api_key, index):
        try:

            if isinstance(entity_pair, str):
                entity_pair = ast.literal_eval(entity_pair)

            if entity_pair and entity_pair != "[]":

                entity_pair_str = "; ".join([f"({a}, {b})" for a, b in entity_pair])


                prompt_content = open(self.relation_extratction_prompt_path, 'r', encoding='utf-8').read()
                prompt_content = prompt_content.replace("${text}$", '"' + chunk + '"')
                prompt_content = prompt_content.replace("${entity pairs}$", entity_pair_str)
                prompt_content = prompt_content.replace("${relation types and their definitions}$",
                                                        "\n" + relation_type_str.lower())


                print(f"Generated prompt for relation extraction:\n{prompt_content}")


                llm_input = self.load_message(prompt_content)
                relation_result = self.call_llm(llm_input, api_key)


                print(f"LLM relation extraction result for index {index}:\n{relation_result}")


                relations = relation_result.split("\n")[1:]
                if not relations or all(not rel.strip() for rel in relations):

                    relations = ["No relations extracted"]
                else:

                    valid_relations = []
                    for rel in relations:
                        try:
                            relation_type, entity_triplet = rel.split(": ", 1)

                            if entity_triplet.startswith('(') and entity_triplet.endswith(')'):
                                new_entity_triple = entity_triplet[1:-1]
                                entity1, relation_name, entity2 = new_entity_triple.split(", ")
                                valid_relations.append(f"{relation_type}: ({entity1}, {relation_name}, {entity2})")
                            else:
                                raise ValueError("Invalid entity triplet format")
                        except ValueError as e:
                            print(f"Error parsing relation at index {index}: {e}")
                            continue

                    relations = valid_relations
            else:

                relations = ["No entity pairs provided"]
        except Exception as e:

            print(f"Error extracting relations for index {index}: {e}")
            relations = [f"Error extracting relations: {str(e)}"]


        return index, relations

    def convert2_relation_triples(self, entity_list, relation_list):
        existing_schema_triples = self.load_existing_schema()

        final_output_list = []
        for idx, (entities_str, relations) in enumerate(zip(entity_list, relation_list)):
            entity_type_mapping = {}

            # Safely split entities for better robustness.
            if entities_str and entities_str.strip():
                try:
                    entities = entities_str.split('; ')
                    for entity in entities:
                        if entity and entity.strip() and ": " in entity:
                            entity_name, entity_type = entity.split(': ', 1)  # Split only on the first delimiter.
                            entity_type_mapping[entity_name] = entity_type
                except Exception as e:
                    print(f"[WARNING] Failed to build the entity-type mapping for row {idx}: {e}")
            else:
                # Skip directly if entities_str is empty.
                final_output_list.append([])
                continue

            output_per_relation = []
            for relation in relations:
                try:
                    # Validate relation format.
                    if not relation or not relation.strip() or ": " not in relation:
                        continue

                    relation_type, entity_triplet = relation.split(': ', 1)

                    # Ensure the relation type is lowercase and format-consistent.
                    relation_type = relation_type.lower()

                    # Validate the entity triple format.
                    if not entity_triplet or not entity_triplet.strip():
                        continue

                    if entity_triplet.startswith('(') and entity_triplet.endswith(')'):
                        new_entity_triple = entity_triplet[1:-1]
                    else:
                        new_entity_triple = entity_triplet

                    # Validate the triple delimiter.
                    if ", " not in new_entity_triple:
                        continue

                    parts = new_entity_triple.split(', ')
                    if len(parts) != 3:
                        continue

                    entity1, relation_name, entity2 = parts
                    type1 = entity_type_mapping.get(entity1, "unknown").replace(";", "")
                    type2 = entity_type_mapping.get(entity2, "unknown").replace(";", "")
                    type_triple = (type1, relation_type, type2)

                    if type_triple in existing_schema_triples:
                        instance_triple = f"({entity1}, {relation_name}, {entity2})"
                        output_per_relation.append(f"{type_triple}: {instance_triple}")

                except Exception as e:
                    # Skip malformed relations without affecting the others.
                    continue

            final_output_list.append(output_per_relation)

        return final_output_list

    def save_extracted_relations(self, chunk_list, entity_list, relation_list):
        result = []
        try:
            triple_with_type = self.convert2_relation_triples(entity_list, relation_list)
            for idx in range(len(chunk_list)):
                task = {
                    "chunk": chunk_list[idx],
                    "relations": relation_list[idx] if relation_list[idx] else "",
                    "relation_with_type": triple_with_type[idx] if idx < len(triple_with_type) else "[]",
                }
                result.append(task)
        except Exception as e:
            # If convert2_relation_triples fails, keep only the raw text.
            print(f"[WARNING] Relation conversion failed; preserving raw text only: {e}")
            for idx in range(len(chunk_list)):
                result.append({
                    "chunk": chunk_list[idx],
                    "relations": "",
                    "relation_with_type": "[]",
                })
        return result

    def process_relation_extraction(self, chunk_list, relation_type_str, entity_list, entity_pair_list, save_path):
        all_result = []
        relation_list = [None] * len(chunk_list)

        with ThreadPoolExecutor(max_workers=len(self.key_list)) as executor:
            futures = {
                executor.submit(
                    self.relation_extraction, chunk, relation_type_str, entity_pair,
                    self.key_list[idx % len(self.key_list)], idx): idx for idx, (chunk, entity_pair) in
                enumerate(zip(chunk_list, entity_pair_list))
            }

            for future in tqdm(as_completed(futures), desc="Relation Extraction", total=len(futures)):
                try:
                    index, relation_triples = future.result()
                    relation_list[index] = relation_triples
                except Exception as e:
                    print(f"[WARNING] Relation extraction failed for row {index}; skipping that row's relations: {e}")
                    relation_list[index] = []

        # Rebuild results in the original text order to keep row counts aligned.
        for idx in range(len(chunk_list)):
            try:
                result = self.save_extracted_relations(
                    [chunk_list[idx]], [entity_list[idx]], [relation_list[idx]]
                )
                all_result.extend(result)
            except Exception as e:
                # If save_extracted_relations fails, keep only the raw text.
                print(f"[WARNING] Saving relations failed for row {idx}; preserving raw text only: {e}")
                all_result.extend([{"chunk": chunk_list[idx], "relations": "", "relation_with_type": "[]"}])

        df = pd.DataFrame(all_result)
        df.to_csv(save_path, index=False, header=False)
        print(f"[OK] Relation extraction complete. Processed {len(chunk_list)} rows.")
        return relation_list

