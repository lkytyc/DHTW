import itertools
import re
from openai import OpenAI
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import ast
from collections import Counter
from tqdm import tqdm


class KGExploration:
    def __init__(self, **explorer_config) -> None:
        # API_base
        self.API_Base = explorer_config["OpenAI_API_Base"]
        # API_key
        self.key_list = explorer_config["API_key_list"]
        # chunk_size
        self.chunk_size = explorer_config["chunk_size"]
        # Model name
        self.model_name = "gpt-4o-2024-08-06"
        # seed_path
        self.seed_file_path = explorer_config["seed_file_path"]
        # prompt_path
        self.entity_extratction_prompt_path = explorer_config["ke_entity_extratction_prompt"]
        self.relation_extratction_prompt_path = explorer_config["ke_relation_extratction_prompt"]
        self.entity_labeling_prompt_path = explorer_config["ke_entity_labeling_prompt"]
        self.entity_type_fusion_prompt = explorer_config["ke_entity_type_fusion_prompt"]
        self.relation_type_fusion_prompt = explorer_config["ke_relation_type_fusion_prompt"]
        # save_path
        self.entity_file_path = explorer_config["save_ke_entity_path"]
        self.relation_file_path = explorer_config["save_ke_relation_path"]
        self.entity_label_path = explorer_config["save_ke_entity_label_path"]
        self.entity_type_path = explorer_config["save_ke_entity_type_path"]
        self.relation_type_batch_path = explorer_config["save_ke_relation_type_path"]
        self.save_kg_schema_path = explorer_config["save_kg_schema_path"]
        self.suggestion_file_path = explorer_config["save_suggestions_path"]
        self.suggestion_prompt = explorer_config["Suggestion_prompt"]

    def load_message(self, prompt_content):
        instruct_content = ""
        message = [{"role": "system", "content": instruct_content}]
        message.append({"role": "user", "content": prompt_content})
        return message

    def read_suggestions(self):
        suggestions = []
        if os.path.isfile(self.suggestion_file_path):
            try:
                with open(self.suggestion_file_path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        if row and row[0].strip():
                            suggestions.append(row[0].strip())
            except Exception as e:
                print(f"Error reading suggestions.csv: {e}")
        else:
            print("Suggestion file not found. Returning an empty list.")


        return "\n".join(suggestions) if suggestions else "None"

    def call_llm(self, llm_input, api_key):
        client = OpenAI(api_key=api_key, base_url=self.API_Base)
        print("LLM Input:", llm_input)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=llm_input,
            temperature=0,
            max_tokens=4096
        )
        print("Response:", response)
        return response.choices[0].message.content.strip()

    def split_list(self, lst):
        return [lst[i:i + self.chunk_size] for i in range(0, len(lst), self.chunk_size)]

    def read_seed_files(self):
        # Check seed_path
        if os.path.isdir(self.seed_file_path):
            files = os.listdir(self.seed_file_path)
            files_list = [os.path.join(self.seed_file_path, file) for file in files if file.endswith('.csv')]
        else:
            print("Invalid seed path")
            return []


        file_content_list = []
        for path in files_list:
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    for line in file:
                        line_content = line.strip()
                        if line_content:
                            file_content_list.append(line_content)
            except Exception as e:
                print(f"Error reading {path}: {e}")

        return file_content_list

    def entity_extraction(self, chunk, prompt_content, suggestions, api_key, index):

        if isinstance(suggestions, list):
            suggestions_str = "\n".join(suggestions) if suggestions else "No Suggestions"
        else:
            suggestions_str = str(suggestions)
        if isinstance(chunk, list):
            chunk = "\n".join(chunk)

        prompt_content = str(prompt_content).replace("${text}$", chunk).replace("${suggestions.csv}$", suggestions_str)
        llm_input = self.load_message(prompt_content)


        entity_result_form_1 = self.call_llm(llm_input, api_key)


        entity_result_form_2 = re.sub(r'\([^)]*\)', '()', entity_result_form_1)
        entity_result = re.sub(r'<[^>]*>', '<>', entity_result_form_2)

        try:

            entity_list = entity_result.split("\n")[1].split(", ")
            entities = ', '.join(entity_list)
        except:
            entities = ""

        return index, entities

    def process_entity_extraction(self, chunk_list, suggestions):

        entity_list = [None] * len(chunk_list)


        entity_extraction_prompt_content = open(self.entity_extratction_prompt_path, 'r', encoding='utf-8').read()


        with ThreadPoolExecutor(max_workers=len(self.key_list)) as executor:

            futures = {
                executor.submit(self.entity_extraction, chunk, entity_extraction_prompt_content,
                                suggestions, self.key_list[idx % len(self.key_list)], idx): idx for idx, chunk in
                enumerate(chunk_list)
            }


            for future in tqdm(as_completed(futures), desc="Entity Extraction", total=len(futures)):
                index, entities = future.result()
                entity_list[index] = entities

        return chunk_list, entity_list

    def relation_extraction(self, chunk, prompt_content, entity_pairs, suggestions, api_key, index):

        formatted_pairs = ast.literal_eval(entity_pairs)
        string_pairs = "; ".join(f"({item[0]}, {item[1]})" for item in formatted_pairs)


        suggestions_str = "\n".join(suggestions) if suggestions else "No Suggestions"
        if isinstance(chunk, list):
            chunk = "\n".join(chunk)

        prompt_content = prompt_content.replace("${text}$", chunk).replace('${entity_pairs}$', string_pairs).replace(
            '${suggestions.csv}$', suggestions_str)

        llm_input = self.load_message(prompt_content)
        relation_result = self.call_llm(llm_input, api_key)
        try:

            relation_triples = relation_result.split("\n")[1]
        except:
            relation_triples = ""
        return index, relation_triples

    def process_relation_extraction(self, chunk_list, entity_pair_list, suggestions):


        relation_triple_list = [None] * len(chunk_list)


        with open(self.relation_extratction_prompt_path, 'r', encoding='utf-8') as file:
            relation_extraction_prompt_content = file.read()


        with ThreadPoolExecutor(max_workers=len(self.key_list)) as executor:
            futures = {
                executor.submit(
                    self.relation_extraction, chunk, relation_extraction_prompt_content, entity_pair,
                    suggestions, self.key_list[idx % len(self.key_list)], idx): idx
                for idx, (chunk, entity_pair) in enumerate(zip(chunk_list, entity_pair_list))
                if entity_pair != []
            }


            for future in tqdm(as_completed(futures), desc="Relation Extraction", total=len(futures)):
                try:
                    index, relation_triples = future.result()


                    if not relation_triples or relation_triples == []:
                        continue


                    relation_triple_list[index] = relation_triples
                except Exception as e:

                    print(f"Error processing relation at index {index}: {e}")

        return relation_triple_list

    def read_entity_infos(self):
        chunk_list = []
        entity_pair_list = []
        with open(self.entity_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            result = list(reader)
            for i in range(len(result)):
                chunk_list.append(result[i][0])
                entity_pair_list.append(result[i][2])
        return chunk_list, entity_pair_list

    def entity_type_labeling(self, chunk, prompt_content, entity, api_key, index):
        prompt_content = prompt_content.replace("${text}$", chunk).replace('${entities}$', entity)
        llm_input = self.load_message(prompt_content)
        entity_label_result = self.call_llm(llm_input, api_key)
        try:

            entity_types = entity_label_result.strip().split("\n")
            if len(entity_types) > 1:
                entity_types = entity_types[1]
            else:
                entity_types = ""
        except Exception as e:
            print(f"Error parsing entity types result at index {index}: {e}")
            entity_types = ""
        return index, entity_types

    def process_entity_type_labeling(self, chunk_list, entity_list):
        entity_type_list = [None] * len(chunk_list)
        entity_type_label_prompt_content = open(self.entity_labeling_prompt_path, 'r', encoding='utf-8').read()

        with ThreadPoolExecutor(max_workers=len(self.key_list)) as executor:
            futures = {
                executor.submit(
                    self.entity_type_labeling, chunk, entity_type_label_prompt_content, entity,
                    self.key_list[idx % len(self.key_list)], idx): idx for idx, (chunk, entity) in
                enumerate(zip(chunk_list, entity_list))
            }

            for future in tqdm(as_completed(futures), desc="Entity Type Labeling", total=len(futures)):
                try:
                    index, entity_types = future.result()


                    labeled_entities = {}
                    if entity_types:
                        try:

                            pairs = entity_types.split('; ')
                            for pair in pairs:
                                if ": " in pair:
                                    entity, entity_type = pair.split(": ", 1)
                                    labeled_entities[entity.strip()] = entity_type.strip()
                                else:
                                    print(f"Unexpected format in pair: '{pair}'")
                        except Exception as e:
                            print(f"Error parsing entity types at index {index}: {e}")


                    all_entities = [ent.strip() for ent in entity_list[index].split(', ') if ent.strip()]
                    complete_entity_labels = []

                    for entity in all_entities:
                        entity_type = labeled_entities.get(entity, "none")
                        complete_entity_labels.append(f"{entity}: {entity_type}")

                    entity_type_list[index] = "; ".join(complete_entity_labels)

                except Exception as e:
                    print(f"Error processing entity type labeling at index {index}: {e}")

                    all_entities = [ent.strip() for ent in entity_list[index].split(', ') if ent.strip()]
                    entity_type_list[index] = "; ".join([f"{entity}: none" for entity in all_entities])

        return entity_type_list



    def read_entity_types(self):
        entity_type_list = []
        with open(self.entity_label_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:

                for cell in row[1:]:
                    if cell:
                        try:
                            parts = cell.split('; ')
                            results = [part.split(': ')[1].strip().replace(';', '') for part in parts if ': ' in part]
                            entity_type_list.extend(results)
                        except (ValueError, SyntaxError) as e:
                            print(f"Error parsing row: {cell}. Error: {e}")

        counter = Counter(entity_type_list)
        unique_entity_type_list = sorted(counter.keys(), key=lambda x: counter[x], reverse=True)


        elements_to_remove = ['none', 'none;', 'none.']
        for element in elements_to_remove:
            if element in unique_entity_type_list:
                unique_entity_type_list.remove(element)


        seen = set()
        final_entity_type_list = []
        for item in unique_entity_type_list:
            if item.lower() not in seen:
                seen.add(item.lower())
                final_entity_type_list.append(item)

        return final_entity_type_list

    def load_previous_entity_fusion(self):

        if os.path.exists(self.entity_type_path):
            try:

                df = pd.read_csv(self.entity_type_path, header=None)

                if df.empty:
                    print(f"File {self.entity_type_path} is empty.")
                    return [], []

                previous_entity_definitions = []
                previous_entity_types = []


                for _, row in df.iterrows():

                    if len(row) >= 2:
                        typename = row[0]
                        definition = row[1]
                        subtypes = row[2] if len(row) > 2 else ""
                        previous_entity_definitions.append(f"{typename}: {definition}")
                        previous_entity_types.append(f"{typename}: {subtypes}")
                    else:
                        print(f"Skipping incomplete row: {row}")

                return previous_entity_definitions, previous_entity_types

            except pd.errors.EmptyDataError:
                print(f"File {self.entity_type_path} is empty or has no data.")
                return [], []

        else:
            print(f"File {self.entity_type_path} does not exist.")
            return [], []
    def entity_type_fusion(self, entity_type_def_list):
        print("************************* Entity Type Fusion *************************")

        prev_entity_types, prev_entity_definitions = self.load_previous_entity_fusion()

        with open(self.entity_type_fusion_prompt, 'r', encoding='utf-8') as file:
            prompt_content = file.read()
        old_entity_types = '\n'.join(f"{defn}: {prev_type}" for defn, prev_type in zip(prev_entity_definitions, prev_entity_types))
        prompt_content = prompt_content.replace('${old entity types}$', old_entity_types)
        prompt_content = prompt_content.replace('${entity types}$', '\n'.join(entity_type_def_list))
        llm_input = self.load_message(prompt_content)
        entity_fuse_result = self.call_llm(llm_input, self.key_list[0])

        entity_types = []
        entity_type_definitions = []

        try:

            parts = entity_fuse_result.split("\n\n", 1)


            if len(parts) >= 2:

                entity_types_raw = parts[0].strip().split("\n")[1:]
                entity_types_raw = [line for line in entity_types_raw if line.strip()]


                entity_type_definitions_raw = parts[1].strip().split("\n")[1:]
                entity_type_definitions_raw = [line for line in entity_type_definitions_raw if line.strip()]


                for line in entity_types_raw:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        subtypes = [v.strip() for v in value.strip().strip('[]').split(',')]
                        entity_types.append(f"{key.strip()}: [{', '.join(subtypes)}]")


                for line in entity_type_definitions_raw:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        entity_type_definitions.append(f"{key.strip()}: {value.strip()}")

        except Exception as e:
            print(f"An error occurred during processing: {e}")

        return entity_type_definitions, entity_types

    def load_previous_relation_fusion(self):

        if os.path.exists(self.relation_type_batch_path):
            try:
                df = pd.read_csv(self.relation_type_batch_path, header=None)

                if df.empty:
                    print(f"File {self.relation_type_batch_path} is empty.")
                    return [], []

                previous_relation_definitions = []
                previous_relation_types = []


                for _, row in df.iterrows():

                    if len(row) >= 3:
                        typename = row[0]
                        definition = row[1]
                        subtypes = row[2]
                        previous_relation_definitions.append(f"{typename}: {definition}")
                        previous_relation_types.append(f"{typename}: {subtypes}")
                    else:
                        print(f"Skipping incomplete row: {row}")

                return previous_relation_definitions, previous_relation_types

            except pd.errors.EmptyDataError:
                print(f"File {self.relation_type_batch_path} is empty or has no data.")
                return [], []

        else:
            print(f"File {self.relation_type_batch_path} does not exist.")
            return [], []

    def read_relation_types(self):
        relation_type_list = []
        relation_instance_list = []


        with open(self.relation_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) < 3 or not row[2]:
                    continue

                try:

                    triplets = row[2].split('; ')


                    for triplet in triplets:
                        triplet = triplet.strip('()')
                        entity1, relation, entity2 = [element.strip() for element in triplet.split(', ')]


                        relation_type_list.append(relation)


                        relation_instance_list.append(f"({entity1}, {relation}, {entity2})")
                except Exception as e:
                    print(f"Error processing row: {row}. Error: {e}")


        unique_relation_type_list = list(dict.fromkeys(relation_type_list))
        unique_relation_instance_list = list(dict.fromkeys(relation_instance_list))

        return unique_relation_type_list, unique_relation_instance_list

    def relation_type_fusion(self, relation_type_def_list, relation_instance_list):

        print("************************* Starting Relation Type Fusion *************************")


        prev_relation_definitions, prev_relation_types = self.load_previous_relation_fusion()


        relation_fusion_map = {}
        for definition in prev_relation_definitions:
            try:
                typename, desc = definition.split(": ", 1)
                relation_fusion_map[typename.strip()] = {
                    "definition": desc.strip(),
                    "subtypes": set(),
                    "instances": []
                }
            except ValueError:
                print(f"Skipping invalid definition entry: {definition}")


        prev_relation_types_str = '\n'.join(prev_relation_definitions)
        new_relation_types_str = '\n'.join(relation_type_def_list)
        relation_instances_str = '\n'.join(relation_instance_list)


        try:
            with open(self.relation_type_fusion_prompt, 'r', encoding='utf-8') as file:
                prompt_content = file.read()

            prompt_content = prompt_content.replace('${previous relation types}$', prev_relation_types_str)
            prompt_content = prompt_content.replace('${relation types}$', new_relation_types_str)
            prompt_content = prompt_content.replace('${relation instances}$', relation_instances_str)
        except FileNotFoundError:
            print(f"Error: Prompt file '{self.relation_type_fusion_prompt}' not found.")
            return [], [], [], []

        llm_input = self.load_message(prompt_content)
        relation_fuse_result = self.call_llm(llm_input, self.key_list[0])


        print("LLM Fusion Result:", relation_fuse_result)

        try:

            relation_fusion_map = {}


            result_lines = [line.strip() for line in relation_fuse_result.splitlines() if line.strip()]

            for segment in result_lines:
                if segment.strip():

                    split_segment = segment.split(":", 3)

                    if len(split_segment) == 4:
                        typename = split_segment[0].strip()
                        definition = split_segment[1].strip()
                        subtype_str = split_segment[2].strip()
                        instance_str = split_segment[3].strip()

                        subtypes_list = [subtype.strip() for subtype in subtype_str.split(",") if
                                         subtype.strip()] if subtype_str else []
                        instances_list = [inst.strip() for inst in instance_str.split(";") if
                                          inst.strip()] if instance_str else []


                        relation_fusion_map[typename] = {
                            "definition": definition,
                            "subtypes": set(subtypes_list),
                            "instances": instances_list
                        }
                    else:
                        print(f"Error parsing line: {segment}")
        except Exception as e:
            print(f"An error occurred during parsing: {e}")
            relation_fusion_map = {}

        final_relation_definitions = []
        final_relation_types = []
        final_relation_instances = []
        final_relation_subtypes = []
        for typename, info in relation_fusion_map.items():
            definition = info["definition"]
            sorted_subtypes = ", ".join(sorted(info["subtypes"])) if info["subtypes"] else ""

            if sorted_subtypes:
                final_relation_definitions.append(f"{typename}: {definition}")
            else:
                final_relation_definitions.append(f"{typename}: {definition}")

            final_relation_types.append(f"{typename}: [{sorted_subtypes}]")


            if info["instances"]:
                instances_str = "; ".join(info["instances"])
                final_relation_instances.append(f"{typename}: {instances_str}")
            else:
                final_relation_instances.append(f"{typename}:")


            final_relation_subtypes.append(f"{typename}: {sorted_subtypes}")


        return final_relation_types, final_relation_definitions,  final_relation_instances, final_relation_subtypes

    def save_labeled_entity_types(self, chunk_list, entity_list, entity_type_list):
        result = []

        for index in range(len(chunk_list)):
            chunk = chunk_list[index].strip()
            entities = entity_list[index].strip() if entity_list[index] else ""
            entity_types = entity_type_list[index].strip() if entity_type_list[index] else ""


            entity_types_split = entity_types.split('; ') if entity_types else []
            entity_types_cleaned = [et.strip() for et in entity_types_split if et.strip()]


            cleaned_entity_types = "; ".join(entity_types_cleaned)
            cleaned_entity_types = f"; {'; '.join(entity_types_cleaned)}" if entity_types_cleaned else ""

            task = {
                "chunk": chunk,
                "entity_types": cleaned_entity_types
            }
            result.append(task)


        df = pd.DataFrame(result, columns=["chunk", "entity_types"])
        df = df.dropna(subset=["chunk", "entity_types"])


        print(df)
        df.to_csv(self.entity_label_path, index=False, header=False)

    import os
    import pandas as pd

    def save_fused_entity_types(self, entity_type_definitions, entity_types):
        result = []
        entity_def_dict = {}


        for item in entity_type_definitions:
            typename, definition = item.split(': ', 1)
            entity_def_dict[typename.strip()] = definition.strip()

        entity_type_dict = {}
        for item in entity_types:
            typename, subtypes = item.split(': ', 1)
            entity_type_dict[typename.strip()] = subtypes.replace(';', '').strip('[]')


        existing_typenames = set()
        if os.path.exists(self.entity_type_path) and os.path.getsize(self.entity_type_path) > 0:
            try:

                df_old = pd.read_csv(self.entity_type_path, header=None, names=["typename", "subtypes", "definition"])
                for _, row in df_old.iterrows():
                    old_typename = str(row['typename']).strip()
                    old_definition = str(row['definition']).strip()
                    old_subtypes = [
                        subtype.strip() for subtype in str(row['subtypes']).strip("[]").split(',') if subtype.strip()
                    ]
                    existing_typenames.add(old_typename)


                    result.append({
                        "typename": old_typename,
                        "subtypes": ', '.join(sorted(old_subtypes)),
                        "definition": old_definition
                    })
            except Exception as e:
                print(f"Error reading the existing file: {e}")


        for typename, definition in entity_def_dict.items():
            if typename not in existing_typenames:
                subtypes = entity_type_dict.get(typename, '')
                if subtypes:
                    subtypes_list = list(set(subtypes.split(', ')))
                    if typename in subtypes_list:
                        subtypes_list.remove(typename)
                    subtypes = ', '.join(sorted(subtypes_list))
                else:
                    subtypes = ''
                result.append({
                    "typename": typename,
                    "subtypes": subtypes,
                    "definition": definition
                })

        df_result = pd.DataFrame(result).drop_duplicates(subset=["typename"])
        df_result.to_csv(self.entity_type_path, index=False, header=False, sep=',', encoding='utf-8')

    def save_fused_relation_types(self, relation_types, relation_type_definitions, relation_instances,
                                  relation_subtypes):

        result = []


        definitions = {item.split(": ", 1)[0].strip(): item.split(": ", 1)[1].strip() for item in
                       relation_type_definitions}
        subtypes = {}
        instances = {}


        for entry in relation_types:
            if entry.strip():
                typename, subtype_str = entry.split(": ", 1)
                typename = typename.strip()
                subtype_str = subtype_str.strip()
                if typename not in subtypes:
                    subtypes[typename] = set()
                subtypes[typename].update(subtype_str.strip("[]").split(", "))


        for instance_entry in relation_instances:
            if instance_entry.strip():

                if ": " in instance_entry:
                    typename, instance_str = instance_entry.split(": ", 1)
                    typename = typename.strip()
                    if typename not in instances:
                        instances[typename] = []

                    if instance_str.strip():
                        instances[typename].extend([inst.strip() for inst in instance_str.split(";") if inst.strip()])
                else:

                    typename = instance_entry.strip().replace(":", "")
                    if typename not in instances:
                        instances[typename] = []
                    print(
                        f"Warning: The instance entry '{instance_entry}' does not contain instances and will be processed as an empty list.")


        for subtype_entry in relation_subtypes:
            if subtype_entry.strip():
                typename, subtype_str = subtype_entry.split(": ", 1)
                typename = typename.strip()
                if typename not in subtypes:
                    subtypes[typename] = set()
                subtypes[typename].update(subtype_str.strip("[]").split(", "))

        for typename, definition in definitions.items():
            subtype_list = sorted(subtypes.get(typename, []))
            instance_list = instances.get(typename, [])


            result.append({
                "relation_type": typename,
                "definition": definition,
                "subtypes": ", ".join(subtype_list) if subtype_list else "",
                "instances": "; ".join(instance_list) if instance_list else ""
            })

        df = pd.DataFrame(result, columns=["relation_type", "definition", "subtypes", "instances"])

        with open(self.relation_type_batch_path, mode='w', newline='', encoding='utf-8') as file:
            df.to_csv(file, index=False, header=False, columns=["relation_type", "definition", "subtypes", "instances"])

    def save_extracted_entities(self,chunk_list, entity_list):
        result = []
        for index in range(len(chunk_list)):
            pairs = list(itertools.combinations(entity_list[index].split(', '), 2))
            task = {
                "chunk": chunk_list[index],
                "entities": entity_list[index],
                "entity_pairs": pairs,
            }
            result.append(task)
        df = pd.DataFrame(result)
        df.to_csv(self.entity_file_path, index=False,header=False)
    def save_extracted_relations(self, chunk_list, entity_pair_list, relation_triple_list):
        result = []
        for index in range(len(chunk_list)):
            task = {
                "chunk": chunk_list[index],
                "entities": entity_pair_list[index],
                "relation_triples": relation_triple_list[index],
            }
            result.append(task)
        df = pd.DataFrame(result)
        df.to_csv(self.relation_file_path, index=False,header=False)

    def generate_kg_schema(self):

        entity_label_map = {}
        entity_type_map = {}
        final_triples = set()

        with open(self.entity_label_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) < 1:
                    continue
                for col_idx in range(1, len(row)):
                    entity_info = row[col_idx].strip()


                    pairs = entity_info.split(";")
                    for pair in pairs:
                        pair = pair.strip()
                        if ": " in pair:
                            entity_name, label = map(str.strip, pair.split(": ", 1))
                            entity_label_map[entity_name] = label


        with open(self.entity_type_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) < 2:
                    continue
                final_class = row[0].strip()
                subtypes = [subtype.strip() for subtype in row[1].strip("[]").split(',') if subtype.strip()]

                entity_type_map[final_class] = final_class
                for subtype in subtypes:
                    entity_type_map[subtype] = final_class

        with open(self.relation_type_batch_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) < 4:
                    continue
                relation = row[0].strip()
                raw_instances = row[3].strip()
                instance_triples = raw_instances.split("; ")
                # Parse triples
                for instance in instance_triples:
                    if instance.startswith("(") and instance.endswith(")"):
                        instance = instance[1:-1]
                    instance_parts = instance.split(", ")
                    if len(instance_parts) == 3:
                        entity1, raw_relation, entity2 = map(str.strip, instance_parts)
                        fused_relation = relation


                        entity1_label = entity_label_map.get(entity1, None)
                        entity2_label = entity_label_map.get(entity2, None)
                        if not entity1_label or not entity2_label:
                            continue

                        entity1_type = entity_type_map.get(entity1_label, None)
                        entity2_type = entity_type_map.get(entity2_label, None)
                        if entity1_type is None or entity2_type is None:
                            continue

                        final_triples.add((entity1_type, fused_relation, entity2_type))
        return list(final_triples)

    def save_kg_schema(self, kg_schema):
        save_kg_schema_path = self.save_kg_schema_path
        existing_triples = set()


        entity_type_map = {}
        with open(self.entity_type_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) < 2:
                    continue
                final_class = row[0].strip()
                subtypes = [subtype.strip() for subtype in row[1].strip("[]").split(',') if subtype.strip()]
                entity_type_map[final_class] = final_class
                for subtype in subtypes:
                    entity_type_map[subtype.strip()] = final_class


        relation_type_map = {}
        with open(self.relation_type_batch_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) < 3:
                    continue
                final_relation = row[0].strip()
                subtypes = [subtype.strip() for subtype in row[2].strip("[]").split(',') if subtype.strip()]
                relation_type_map[final_relation] = final_relation
                for subtype in subtypes:
                    relation_type_map[subtype.strip()] = final_relation


        if os.path.exists(save_kg_schema_path):
            try:
                with open(save_kg_schema_path, mode='r', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader, None)
                    for row in reader:
                        if len(row) == 3:
                            entity1_type, relation_type, entity2_type = map(str.strip, row)


                            updated_entity1_type = entity_type_map.get(entity1_type, entity1_type)
                            updated_entity2_type = entity_type_map.get(entity2_type, entity2_type)
                            updated_relation_type = relation_type_map.get(relation_type, relation_type)

                            existing_triples.add((updated_entity1_type, updated_relation_type, updated_entity2_type))
            except Exception as e:
                print(f"Failed to read existing KG Schema: {e}")


        updated_new_triples = set()
        for triple in kg_schema:
            entity1_type, relation_type, entity2_type = triple


            updated_entity1_type = entity_type_map.get(entity1_type.strip(), entity1_type.strip())
            updated_entity2_type = entity_type_map.get(entity2_type.strip(), entity2_type.strip())
            updated_relation_type = relation_type_map.get(relation_type.strip(), relation_type.strip())

            updated_new_triples.add((updated_entity1_type, updated_relation_type, updated_entity2_type))


        combined_triples = existing_triples.union(updated_new_triples)

        if not combined_triples:
            print("No new triples to save.")
            return


        try:
            with open(save_kg_schema_path, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Entity1', 'Relation', 'Entity2'])
                for triple in sorted(combined_triples):
                    writer.writerow(triple)
            print(f"New triples have been saved successfully to {save_kg_schema_path}")
        except Exception as e:
            print(f"Failed to save KG Schema: {e}")

    def read_file(self, file_path):
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            return ""

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ""

    def split_into_units(self, content):
        units = []
        current_unit = []

        for line in content:
            if line.strip():
                current_unit.append(line)
            elif current_unit:
                units.append(current_unit)
                current_unit = []


        return current_unit

    def extract_key_and_value(self, unit):

        parts = unit.split(',', 1)

        if len(parts) == 2:
            key = parts[0].strip().strip('"').strip('()')
            value = parts[1].strip()
        elif len(parts) == 1:
            key = parts[0].strip().strip('"').strip('()')
            value = ""
        else:
            key = ""
            value = ""
        return key, value

    def compare_file_changes(self, old_content_str, new_content_str):

        if not old_content_str and not new_content_str:
            return {"added": [], "removed": [], "changed": []}

        old_units = self.split_into_units(old_content_str.strip().split('\n'))
        new_units = self.split_into_units(new_content_str.strip().split('\n'))


        old_first_col = {}
        for unit in old_units:
            if unit:
                key, value = self.extract_key_and_value(unit)
                old_first_col[key] = ''.join(unit)

        new_first_col = {}
        for unit in new_units:
            if unit:
                key, value = self.extract_key_and_value(unit)
                new_first_col[key] = ''.join(unit)

        added = set(new_first_col) - set(old_first_col)
        removed = set(old_first_col) - set(new_first_col)
        changed = []

        for old_key, old_value in old_first_col.items():
            for new_key, new_value in new_first_col.items():
                if (self.extract_key_and_value(old_value)[1] == self.extract_key_and_value(new_value)[1] and
                        old_key != new_key):
                    changed.append((old_key, new_key))


        added -= {new_key for old_key, new_key in changed}
        removed -= {old_key for old_key, new_key in changed}

        return {
            "added": list(added),
            "removed": list(removed),
            "changed": changed
        }

    import csv

    def pause_and_process_csv(self):
        try:

            entity_types_old = self.read_file(self.entity_type_path)
            relation_types_old = self.read_file(self.relation_type_batch_path)


            input("Please modify the file, and press the Enter key to continue after completing the operation...")
            while input().strip() != '':
                print("Please press the Enter key directly to continue...")


            entity_types_new = self.read_file(self.entity_type_path)
            relation_types_new = self.read_file(self.relation_type_batch_path)


            entity_changes = self.compare_file_changes(entity_types_old, entity_types_new)
            relation_changes = self.compare_file_changes(relation_types_old, relation_types_new)


            suggestions = self.read_suggestions()
            prompt_template = self.read_file(self.suggestion_prompt)


            prompt_content = prompt_template.replace('${suggestions}$', suggestions)
            prompt_content = prompt_content.replace('${entity_changes}$', str(entity_changes))
            prompt_content = prompt_content.replace('${relation_changes}$', str(relation_changes))


            llm_input = self.load_message(prompt_content)
            result = self.call_llm(llm_input, self.key_list[0])


            if result:
                self.save_suggestions_to_csv(result)
            else:
                print("None")

        except Exception as e:
            print("error:", str(e))

    def save_suggestions_to_csv(self, result):
        try:
            with open(self.suggestion_file_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)


                suggestions_list = result.split(';')


                pattern = re.compile(r'(Implementation Level Suggestions:|Design Level Suggestions:)\s*(.+)$')

                for suggestion in suggestions_list:
                    suggestion = suggestion.strip()
                    if suggestion:

                        match = pattern.search(suggestion)
                        if match:
                            type_text = match.group(
                                1).strip()
                            text = match.group(2).strip()
                            writer.writerow([type_text, text])

            print("Processing complete. Results saved to", self.suggestion_file_path)
        except Exception as e:
            print("Error occurred while saving CSV:", str(e))
