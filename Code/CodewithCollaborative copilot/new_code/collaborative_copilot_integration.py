import csv
import io
import os
import re


class CollaborativeCopilotIntegration:
    def __init__(
        self,
        entity_type_path,
        relation_type_path,
        schema_path,
        suggestions_path,
        table_output_path,
        report_output_path,
    ):
        self.entity_type_path = entity_type_path
        self.relation_type_path = relation_type_path
        self.schema_path = schema_path
        self.suggestions_path = suggestions_path
        self.table_output_path = table_output_path
        self.report_output_path = report_output_path

    def build_prompt_content(self, prompt_template):
        prompt_content = prompt_template.replace(
            "${entity_types}$",
            self.read_text(self.entity_type_path, "No entity type file found."),
        )
        prompt_content = prompt_content.replace(
            "${relation_types}$",
            self.read_text(self.relation_type_path, "No relation type file found."),
        )
        prompt_content = prompt_content.replace(
            "${kg_schema}$",
            self.read_text(self.schema_path, "No schema file found."),
        )
        prompt_content = prompt_content.replace(
            "${previous_suggestions}$",
            self.read_text(self.suggestions_path, "No previous suggestions found."),
        )
        return prompt_content

    def save_outputs(self, llm_result):
        summary_text, table_text = self.parse_result(llm_result)
        table_text = self.filter_table_rows(table_text)
        self.write_text(self.report_output_path, summary_text or llm_result)
        self.write_text(self.table_output_path, table_text)

    def parse_result(self, llm_result):
        normalized = llm_result.replace("\r\n", "\n").strip()
        normalized = normalized.replace("```csv", "").replace("```", "").strip()

        summary_match = re.search(
            r"###\s*summary\s*###\s*(.*?)\s*###\s*table\s*###",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
        table_match = re.search(
            r"###\s*table\s*###\s*(.*)",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )

        summary_text = summary_match.group(1).strip() if summary_match else normalized
        table_text = table_match.group(1).strip() if table_match else ""
        return summary_text, table_text

    def filter_table_rows(self, table_text):
        if not table_text.strip():
            return table_text

        valid_types = self.read_entity_type_names()
        if not valid_types:
            return table_text

        try:
            reader = csv.reader(io.StringIO(table_text))
            rows = list(reader)
            if not rows:
                return table_text

            header = rows[0]
            filtered_rows = [header]
            for row in rows[1:]:
                if len(row) < 6:
                    continue
                type_a = row[0].strip()
                type_b = row[1].strip()
                if type_a in valid_types and type_b in valid_types and type_a != type_b:
                    filtered_rows.append(row)

            if len(filtered_rows) <= 1:
                return table_text

            output = io.StringIO()
            writer = csv.writer(output, lineterminator="\n")
            writer.writerows(filtered_rows)
            return output.getvalue().strip()
        except Exception:
            return table_text

    def read_entity_type_names(self):
        if not os.path.exists(self.entity_type_path):
            return set()
        valid_types = set()
        try:
            with open(self.entity_type_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row and row[0].strip():
                        valid_types.add(row[0].strip())
        except Exception:
            return set()
        return valid_types

    @staticmethod
    def read_text(path, fallback_text):
        if not os.path.exists(path):
            return fallback_text
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip() or fallback_text
        except Exception as e:
            return f"{fallback_text}\nRead error: {e}"

    @staticmethod
    def write_text(path, content):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.write(content.strip() if content else "")
