from src.output_parsers.AnswerParser import AnswerParser
from src.output_parsers.YesNoParser import YesNoParser
from src.preprocessing_utils import reduce_context, replace_empty_label
from src.filters import UnanswerableFilter
config = {
    "check_entity_type": {
        "groups": [
            "validators_main",
            "check_entity_type",
            "main",
            "main_w_fact_check_tr",
             "main_w_o_predict_answer",
             "main_w_o_predict_answer_reduced_context",
             "main_w_o_check_evidence",
             "main_w_o_check_evidence_w_fact_check_tr"
        ],
        "prompt_files": [
            {
                "file": "check_entity_type/example_1_user.txt",
                "role": "user"
            },
            {
                "file": "check_entity_type/example_1_assistant.txt",
                "role": "assistant"
            },
            {
                "file": "check_entity_type/example_2_user.txt",
                "role": "user"
            },
            {
                "file": "check_entity_type/example_2_assistant.txt",
                "role": "assistant"
            },
            {
                "file": "check_entity_type/example_3_user.txt",
                "role": "user"
            },
            {
                "file": "check_entity_type/example_3_assistant.txt",
                "role": "assistant"
            },
            {
                "file": "check_entity_type/prompt_template.txt",
                "role": "user",
                "variables": {
                    "context": "context",
                    "question": "question",
                    "label": "label"
                }
            }
        ],
        "filters": [
                UnanswerableFilter
        ],
        "validation_function": YesNoParser().validate,
        "parsing_function": YesNoParser().parse,
        "evaluation_function": lambda parsed, label : None if parsed is None else  parsed.lower() == "yes"
        },
    "check_evidence": {
        "groups": [
            "main",
            "validators_main",
            "check_evidence",
            "main_w_fact_check_tr",
            "main_w_o_predict_answer",
            "main_w_o_check_entity_type",
            "main_w_o_predict_answer_reduced_context"
        ],
        "prompt_files": [
            {
                "file": "check_evidence/example_1_user.txt",
                "role": "user"
            },
            {
                "file": "check_evidence/example_1_assistant.txt",
                "role": "assistant"
            },
            {
                "file": "check_evidence/example_2_user.txt",
                "role": "user"
            },
            {
                "file": "check_evidence/example_2_assistant.txt",
                "role": "assistant"
            },
            {
                "file": "check_evidence/prompt_template.txt",
                "role": "user",
                "variables": {
                    "context": "context",
                    "question": "question",
                    "label": "label"
                }
            }
        ],
        "preprocessing_steps": [
     
        ],
        "validation_function": YesNoParser().validate,
        "parsing_function": YesNoParser().parse,
        "evaluation_function": lambda parsed, label : None if parsed is None else  parsed.lower() == "yes"
        },  
    "predict_answer": {
        "groups": [
            "predictors",
            "main",
            "predict_answer",
            "main_w_fact_check_tr",
            "main_w_o_predict_answer_reduced_context",
            "main_w_o_check_entity_type",
            "main_w_o_check_evidence",
            "main_w_o_check_evidence_w_fact_check_tr"
        ],
        "prompt_files": [
            {
                "file": "predict_answer/example_1_user.txt",
                "role": "user"
            },
            {
                "file": "predict_answer/example_1_assistant.txt",
                "role": "assistant"
            },
            {
                "file": "predict_answer/example_2_user.txt",
                "role": "user"
            },
            {
                "file": "predict_answer/example_2_assistant.txt",
                "role": "assistant"
            },
            {
                "file": "predict_answer/prompt_template.txt",
                "role": "user",
                "variables": {
                    "context": "context",
                    "question": "question",
                    "label": "label"
                }
            }
        ],
        "validation_function": AnswerParser().validate,
        "parsing_function": AnswerParser().parse,
        "evaluation_function": AnswerParser().evaluate
    },
     "predict_answer_reduced_context": {
        "groups": [
            "predictors",
            "predict_answer_reduced_context",
            "reduced_context",
            "main",
            "main_w_fact_check_tr",
            "main_w_o_predict_answer",
            "main_w_o_check_entity_type",
            "main_w_o_check_evidence",
            "main_w_o_check_evidence_w_fact_check_tr"
        ],
        "prompt_files": [
            {
                "file": "predict_answer/example_1_user.txt",
                "role": "user"
            },
            {
                "file": "predict_answer/example_1_assistant.txt",
                "role": "assistant"
            },
            {
                "file": "predict_answer/example_2_user.txt",
                "role": "user"
            },
            {
                "file": "predict_answer/example_2_assistant.txt",
                "role": "assistant"
            },
            {
                "file": "predict_answer/prompt_template.txt",
                "role": "user",
                "variables": {
                    "context": "context",
                    "question": "question",
                    "label": "label"
                }
            }
        ],
        "preprocessing_steps": [
            reduce_context
        ],
        "validation_function": AnswerParser().validate,
        "parsing_function": AnswerParser().parse,
        "evaluation_function": AnswerParser().evaluate
    },
    "fact_check_tr": {
        "groups": [
            "validators",
            "fact_check_tr",
            "main_w_fact_check_tr",
            "main_val_fact_check_tr",
            "main_w_o_check_evidence_w_fact_check_tr"  
        ],
        "prompt_files": [
            {
                "file": "fact_check_tr/example_1_user.txt",
                "role": "user"
            },
            {
                "file": "fact_check_tr/example_1_assistant.txt",
                "role": "assistant"
            },
            {
                "file": "fact_check_tr/example_2_user.txt",
                "role": "user"
            },
            {
                "file": "fact_check_tr/example_2_assistant.txt",
                "role": "assistant"
            },
            {
                "file": "fact_check_tr/example_3_user.txt",
                "role": "user"
            },
            {
                "file": "fact_check_tr/example_3_assistant.txt",
                "role": "assistant"
            },
            {
                "file": "fact_check_tr/example_4_user.txt",
                "role": "user"
            },
            {
                "file": "fact_check_tr/example_4_assistant.txt",
                "role": "assistant"
            },
            {
                "file": "fact_check_tr/example_5_user.txt",
                "role": "user"
            },
            {
                "file": "fact_check_tr/example_5_assistant.txt",
                "role": "assistant"
            },
            {
                "file": "fact_check_tr/example_6_user.txt",
                "role": "user"
            },
            {
                "file": "fact_check_tr/example_6_assistant.txt",
                "role": "assistant"
            },
            {
                "file": "fact_check_tr/prompt_template.txt",
                "role": "user",
                "variables": {
                    "context": "context",
                    "question": "question",
                    "label": "label"
                }
            }
        ],
        "preprocessing_steps": [
            replace_empty_label
        ],
        "validation_function": YesNoParser().validate,
        "parsing_function": YesNoParser().parse,
        "evaluation_function": lambda parsed, label : None if parsed is None else  parsed.lower() == "yes"
        }
    }

