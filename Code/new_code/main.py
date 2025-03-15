import csv
import yaml
import kg_exploration as KE
import kg_construction as KC
import os
import json

def kg_exploration(universal_config):
    kgexplorer = KE.KGExploration(**universal_config)
    file_content_list = kgexplorer.read_seed_files()
    chunks = kgexplorer.split_list(file_content_list)
    all_entity_lists = []
    all_relation_lists = []
    all_entity_types = []
    all_relation_types = []


    for chunk in chunks:
        suggestions = kgexplorer.read_suggestions()
        chunk_list, entity_list = kgexplorer.process_entity_extraction(chunk, suggestions)
        kgexplorer.save_extracted_entities(chunk_list, entity_list)
        all_entity_lists.append(entity_list)


        chunk_list, entity_pair_list = kgexplorer.read_entity_infos()


        relation_triple_list = kgexplorer.process_relation_extraction(chunk_list, entity_pair_list, suggestions)
        kgexplorer.save_extracted_relations(chunk_list, entity_pair_list, relation_triple_list)
        all_relation_lists.append(relation_triple_list)


        entity_type_list = kgexplorer.process_entity_type_labeling(chunk_list, entity_list)
        kgexplorer.save_labeled_entity_types(chunk_list, entity_list, entity_type_list)
        all_entity_types.append(entity_type_list)

        unique_entity_type_list = kgexplorer.read_entity_types()
        entity_type_definitions, entity_types = kgexplorer.entity_type_fusion(unique_entity_type_list)
        kgexplorer.save_fused_entity_types(entity_type_definitions, entity_types)

        unique_relation_type_list, relation_instances = kgexplorer.read_relation_types()
        relation_types, relation_type_definitions, relation_instances, final_relation_subtypes = kgexplorer.relation_type_fusion(unique_relation_type_list, relation_instances)
        kgexplorer.save_fused_relation_types(relation_types, relation_type_definitions, relation_instances, final_relation_subtypes)
        schema = kgexplorer.generate_kg_schema()
        kgexplorer.save_kg_schema(schema)
        kgexplorer.pause_and_process_csv()


def kg_construction(universal_config):
    kgconstructor = KC.KGConstruction(**universal_config)
    all_file_list = kgconstructor.read_all_files()
    entity_type_str = kgconstructor.get_entity_type()
    relation_type_str = kgconstructor.get_relation_type()
    kgconstructor.process_entity_extraction(all_file_list, entity_type_str, kgconstructor.entity_file_path)
    text_list = []
    entity_list = []
    entity_pairs = []
    with open('../output/kg_construction/entity.csv',encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            text_list.append(row[0])
            entity_list.append(row[1])
            entity_pairs.append(row[2])
    kgconstructor.process_relation_extraction(text_list, relation_type_str, entity_list, entity_pairs, kgconstructor.relation_file_path)

def main():
    with open('../config.yaml', 'r', encoding="utf-8") as config_file:
        universal_config = yaml.safe_load(config_file)
    kg_exploration(universal_config)
    kg_construction(universal_config)

if __name__ == "__main__":
     main()