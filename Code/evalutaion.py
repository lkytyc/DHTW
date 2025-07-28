import pandas as pd
import ast
import re
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F





def process_relation_string(relation_str):
    if pd.isna(relation_str):
        return ''
    try:

        relation_list = ast.literal_eval(relation_str)
        processed_parts = []
        for relation in relation_list:
            parts = relation.split(': ')
            if len(parts) > 1:
                processed_parts.append(parts[1].strip())
        return '; '.join(processed_parts)
    except (ValueError, SyntaxError):
        return relation_str



def parse_triplets(relation_str):
    triplet_pattern = r"\(([^,]+),\s*([^,]+),\s*([^\)]+)\)"
    return re.findall(triplet_pattern, relation_str)



def match_triplets(triplets_x, triplets_y):
    result = []
    for t1 in triplets_x:
        head1, rel1, tail1 = [x.lower().replace("(", "").replace(")", "") for x in t1]
        for t2 in triplets_y:
            head2, rel2, tail2 = [x.lower().replace("(", "").replace(")", "") for x in t2]
            if (head1 in head2 and tail1 in tail2) or (head2 in head1 and tail2 in tail1):
                result.append(f"({', '.join(t1)})")
                result.append(f"({', '.join(t2)})")
    final_result = []
    for i in range(0, len(result), 2):
        final_result.append(f"{{{result[i]}, {result[i + 1]}}}")
    return "; ".join(final_result)



def process_and_match_triplets(file1_path, file2_path):

    file1 = pd.read_csv(file1_path, header=None, names=['Text', 'Relations'])
    file2 = pd.read_csv(file2_path, header=None, names=['Text', 'Relations'])

    file1['Text'] = file1['Text'].str.lower()
    file2['Text'] = file2['Text'].str.lower()


    merged_df = pd.merge(file1, file2, on='Text', how='outer')


    merged_df['Relations_y'] = merged_df['Relations_y'].apply(process_relation_string)


    def count_triplets_in_column(df, column_name):
        return df[column_name].apply(lambda x: len(parse_triplets(x)) if pd.notna(x) else 0)


    merged_df['Relations_x_triplet_count'] = count_triplets_in_column(merged_df, 'Relations_x')
    merged_df['Relations_y_triplet_count'] = count_triplets_in_column(merged_df, 'Relations_y')


    truth_triples = merged_df['Relations_x_triplet_count'].sum()
    extracted_triples = merged_df['Relations_y_triplet_count'].sum()


    final_list = []
    for index, row in merged_df.iterrows():
        if pd.notna(row['Relations_x']) and pd.notna(row['Relations_y']):
            triplets_x = parse_triplets(row['Relations_x'].lower())
            triplets_y = parse_triplets(row['Relations_y'].lower())
            output = match_triplets(triplets_x, triplets_y)

            output_split = output.split(';')

            final_list.extend([item.strip() for item in output_split if item.strip()])

    return final_list, truth_triples, extracted_triples





def initialize_bert_model(model_file='../bert_model'):
    tokenizer = BertTokenizer.from_pretrained(model_file, clean_up_tokenization_spaces=True)
    model = BertModel.from_pretrained(model_file)


    if torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")


    model.to(device)
    return tokenizer, model, device



def get_embedding(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # 将输入数据移动到 GPU 或 CPU
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]



def cosine_similarity(vec1, vec2):
    return F.cosine_similarity(vec1, vec2)



def calculate_similarity_for_triplets(triples_list, tokenizer, model, device, threshold):
    similarities = []
    num = 0

    for idx, triple_pair in enumerate(triples_list):
        try:

            triple_pair = triple_pair.strip('{}').split('), (')
            if len(triple_pair) != 2:
                raise ValueError(f"Malformed triple pair at index {idx}: {triple_pair}")


            triple1 = triple_pair[0].lower().replace('(', '').replace(')', '').split(', ')
            triple2 = triple_pair[1].lower().replace('(', '').replace(')', '').split(', ')


            triple1_text = triple1[1].lower()
            triple2_text = triple2[1].lower()


            triple1_embedding = get_embedding(triple1_text, tokenizer, model, device)
            triple2_embedding = get_embedding(triple2_text, tokenizer, model, device)


            similarity = cosine_similarity(triple1_embedding, triple2_embedding).item()

            if similarity >= threshold:
                num += 1

            similarities.append((triple1_text, triple2_text, similarity))

        except Exception as e:

            print(f"Error processing triple pair at index {idx}: {e}")
            continue

    return similarities, num


# ======================== 整合和调用 ========================

def main(file1_path, file2_path, threshold):
    final_list, truth_num, extract_num = process_and_match_triplets(file1_path, file2_path)
    final_list = [item for item in final_list if item != '']


    tokenizer, model, device = initialize_bert_model()


    similarities, num = calculate_similarity_for_triplets(final_list, tokenizer, model, device, threshold)

    precision = round(num / extract_num, 2)
    recall = round(num / truth_num, 2)
    f1 = round((2 * precision * recall) / (precision + recall), 2)

    print(num)
    print(extract_num)
    print(truth_num)

    print(f"Threshold {threshold}: Precision {precision} Recall {recall} F1 {f1}")
if __name__ == '__main__':
    print("=====================================")
    file1_path = 'The groundtruth.csv'
    file2_path = 'new.csv'
    main(file1_path, file2_path, 0.90)
    main(file1_path, file2_path, 0.92)
    main(file1_path, file2_path, 0.94)



