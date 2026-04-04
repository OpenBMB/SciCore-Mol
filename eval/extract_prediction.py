#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 raw_output extractpredictionresultfunction
supportsformat<|im_start|>assistant, <think>, <think> 
"""

import re
from typing import Optional


# tasktypedefineneedsscoringscriptdefineconsistent
SMILES_TOKEN_RE = re.compile(r"([A-Za-z0-9@+\-\[\]\(\)=#\\/%.]+)")
FORMULA_TOKEN_RE = re.compile(r"([A-Za-z0-9\(\)\.\+\-]+)")
NUMBER_TOKEN_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
BOOL_TOKEN_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _canonical_bool(text: str) -> str:
 """ Yes/Nosupportsformat"""
 if not isinstance(text, str):
 text = str(text)
 text = text.strip().lower()
 
 # remove
 text = text.rstrip('.,;:!?')
 
 # mapping - valueYes/True/Positive/Toxic
 yes_values = {"yes", "y", "true", "t", "1", "positive", "toxic", "unsafe", "harmful"}
 # mapping - valueNo/False/Negative/Non-toxic
 no_values = {"no", "n", "false", "f", "0", "negative", "non-toxic", "non toxic", "nontoxic", "safe", "non-harmful"}
 
 # match
 if text in yes_values:
 return "Yes"
 elif text in no_values:
 return "No"
 
 # matchprocesscontains
 m = BOOL_TOKEN_RE.search(text)
 if m:
 v = m.group(1).lower()
 if v in ("yes", "true", "toxic", "unsafe"):
 return "Yes"
 elif v in ("no", "false", "non-toxic", "safe"):
 return "No"
 
 # matchtoxic
 if "toxic" in text and "non" not in text and "not" not in text:
 return "Yes"
 elif "non-toxic" in text or "nontoxic" in text or ("non" in text and "toxic" in text):
 return "No"
 
 return ""


def _extract_core_answer(text: str, task_name: str, text_tasks: set, smiles_tasks: set, 
 formula_element_tasks: set, formula_split_tasks: set,
 number_tasks: set, boolean_tasks: set) -> str:
 """extract"""
 if text is None:
 return ""
 text = str(text)

 if task_name in text_tasks:
 return text.strip()

 if task_name in smiles_tasks:
 for line in text.splitlines():
 line = line.strip()
 if not line:
 continue
 m = SMILES_TOKEN_RE.search(line)
 if m:
 return m.group(1)
 return text.strip()

 if task_name in formula_element_tasks or task_name in formula_split_tasks:
 for line in text.splitlines():
 line = line.strip()
 if not line:
 continue
 m = FORMULA_TOKEN_RE.search(line)
 if m:
 return m.group(1)
 return text.strip()

 if task_name in number_tasks:
 m = NUMBER_TOKEN_RE.search(text)
 if m:
 return m.group(0)
 return text.strip()

 if task_name in boolean_tasks:
 return _canonical_bool(text)

 # defaultreturnsnon-empty
 for line in text.splitlines():
 line = line.strip()
 if line:
 return line
 return text.strip()


def get_special_tokens_from_tokenizer(tokenizer) -> list:
 """
 tokenizergetallspecial tokens
 returns: list of special token strings
 """
 special_tokens = []
 
 # getspecial tokens
 for attr in ['eos_token', 'bos_token', 'pad_token', 'unk_token', 'sep_token', 'cls_token']:
 token = getattr(tokenizer, attr, None)
 if token and isinstance(token, str):
 special_tokens.append(token)
 
 # getspecial tokensif
 if hasattr(tokenizer, 'additional_special_tokens') and tokenizer.additional_special_tokens:
 special_tokens.extend(tokenizer.additional_special_tokens)
 
 # removeNonevalue
 special_tokens = [t for t in special_tokens if t]
 special_tokens = list(set(special_tokens))
 
 return special_tokens


def get_assistant_marker_from_tokenizer(tokenizer) -> list:
 """
 tokenizerchat_templategetassistantmark
 returns: list of possible assistant markers (e.g., ["<|im_start|>assistant", "[/INST]"])
 """
 markers = []
 
 # chat_templateinferassistantmark
 if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
 try:
 # testchat_templateassistantmark
 test_messages = [
 {"role": "system", "content": "test"},
 {"role": "user", "content": "test"},
 {"role": "assistant", "content": ""}
 ]
 template_result = tokenizer.apply_chat_template(
 test_messages,
 tokenize=False,
 add_generation_prompt=True
 )
 
 # assistantmark
 if "<|im_start|>assistant" in template_result:
 markers.append("<|im_start|>assistant")
 if "[/INST]" in template_result:
 markers.append("[/INST]")
 if "<|start_header_id|>assistant<|end_header_id|>" in template_result:
 markers.append("<|start_header_id|>assistant<|end_header_id|>")
 except Exception:
 pass
 
 # ifusedefaultvalue
 if not markers:
 markers = ["<|im_start|>assistant", "[/INST]"]
 
 return markers


def remove_special_tokens(text: str, special_tokens: list = None) -> str:
 """
 removeallspecial tokensreturns
 
 Args:
 text: input
 special_tokens: special tokenslistifNoneusedefaulttokens
 """
 if not text:
 return ""
 
 text = str(text)
 
 if special_tokens:
 # usespecial tokens
 for token in special_tokens:
 if token:
 # forregex
 escaped_token = re.escape(token)
 text = re.sub(escaped_token, "", text)
 else:
 # ifusedefaulttokens
 text = re.sub(r"<\|endoftext\|>", "", text)
 text = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", text, flags=re.DOTALL)
 text = re.sub(r"<\|im_start\|>", "", text)
 text = re.sub(r"<\|im_end\|>", "", text)
 text = re.sub(r"<\|eot_id\|>", "", text)
 text = re.sub(r"</s>", "", text)
 text = re.sub(r"<s>", "", text)
 text = re.sub(r"\[/INST\]", "", text)
 text = re.sub(r"\[INST\]", "", text)
 
 # empty
 text = re.sub(r"\n\s*\n", "\n", text)
 text = text.strip()
 
 return text


def extract_answer_only(raw_output: str, assistant_markers: list = None, special_tokens: list = None) -> str:
 """
 raw_outputextractanswerversionassistantextractremovethinklabel
 
 Args:
 raw_output: originaloutput
 assistant_markers: assistantmarklistifNoneusedefaultmark
 special_tokens: special tokenslistfor
 
 returns: removethinklabelspecial tokens
 """
 if not raw_output:
 return ""
 
 text = str(raw_output)
 
 # ifassistant_markersusedefaultvalue
 if not assistant_markers:
 assistant_markers = ["<|im_start|>assistant", "[/INST]", "<|start_header_id|>assistant<|end_header_id|>", "\nassistant\n", "assistant\n"]
 
 # 1. extractassistant
 assistant_text = ""
 found_marker = False
 
 for marker in assistant_markers:
 if marker in text:
 if marker == "[/INST]":
 assistant_text = text.split("[/INST]")[-1]
 elif marker == "<|start_header_id|>assistant<|end_header_id|>":
 # mark
 marker_start = text.find(marker)
 if marker_start >= 0:
 after_marker = text[marker_start + len(marker):]
 # nextheaderendmark
 next_header = after_marker.find("<|start_header_id|>")
 if next_header >= 0:
 assistant_text = after_marker[:next_header]
 else:
 assistant_text = after_marker
 elif marker in ("\nassistant\n", "assistant\n"):
 # process "assistant" markIntern-S1modeluse
 # supports "\nassistant\n" "assistant\n" format
 marker_start = text.find(marker)
 if marker_start >= 0:
 assistant_text = text[marker_start + len(marker):]
 else:
 # processclass "<|im_start|>assistant" mark
 marker_start = text.find(marker)
 if marker_start >= 0:
 after_marker = text[marker_start + len(marker):]
 after_marker = after_marker.lstrip()
 # endmark
 if "<|im_end|>" in after_marker:
 assistant_text = after_marker.split("<|im_end|>")[0]
 elif "<|im_start|>" in after_marker:
 assistant_text = after_marker.split("<|im_start|>")[0]
 else:
 assistant_text = after_marker
 found_marker = True
 break
 
 if not found_marker:
 assistant_text = text
 
 assistant_text = assistant_text.lstrip()
 
 # 2. removethinklabelincludinglabel
 # supportsthinklabelformattokenizerget
 think_tags = [
 ("<think>", "</think>"), # thinklabel
 ("<thinking>", "</thinking>"), # thinkinglabel
 ]
 
 answer_text = assistant_text
 for open_tag, close_tag in think_tags:
 # removeallthinklabelincluding
 pattern = re.escape(open_tag) + r"(.*?)" + re.escape(close_tag)
 answer_text = re.sub(pattern, "", answer_text, flags=re.DOTALL)
 # processlabelremoveopen_tagallifclose_tag
 # open_tagclose_tagexecute
 if open_tag in answer_text and close_tag not in answer_text:
 # lastopen_tag
 last_open = answer_text.rfind(open_tag)
 if last_open >= 0:
 # removeopen_tagstartall
 answer_text = answer_text[:last_open]
 # removethinklabel
 answer_text = re.sub(re.escape(open_tag), "", answer_text)
 answer_text = re.sub(re.escape(close_tag), "", answer_text)
 
 # 3. removespecial tokens
 answer_text = remove_special_tokens(answer_text, special_tokens)
 
 return answer_text.strip()


def extract_prediction_from_raw(
 raw_output: str,
 task_name: str,
 text_tasks: set = None,
 smiles_tasks: set = None,
 formula_element_tasks: set = None,
 formula_split_tasks: set = None,
 number_tasks: set = None,
 boolean_tasks: set = None,
 answer_only: str = None, # ifanswer_onlyuse
) -> str:
 """
 raw_output answer_only extractpredictionresult
 
 rule:
 1. ifanswer_onlyuse
 2. otherwiseraw_outputextractextractassistantusethink
 3. removeallinvalidlabelspecial tokens
 
 Args:
 raw_output: originaloutput
 task_name: task
 text_tasks: taskset
 smiles_tasks: SMILEStaskset
 formula_element_tasks: matchtaskset
 formula_split_tasks: splitmatchtaskset
 number_tasks: valuetaskset
 boolean_tasks: taskset
 answer_only: containsansweroptionalifuse
 """
 # ifanswer_onlyuseno need toprocess
 if answer_only is not None and answer_only.strip():
 text = answer_only.strip()
 # defaulttasksetif
 if text_tasks is None:
 text_tasks = {"molecule_captioning"}
 if smiles_tasks is None:
 smiles_tasks = {"forward_synthesis", "retrosynthesis", "molecule_generation", "name_conversion-i2s"}
 if formula_element_tasks is None:
 formula_element_tasks = {"name_conversion-i2f", "name_conversion-s2f"}
 if formula_split_tasks is None:
 formula_split_tasks = {"name_conversion-s2i"}
 if number_tasks is None:
 number_tasks = {"property_prediction-esol", "property_prediction-lipo"}
 if boolean_tasks is None:
 boolean_tasks = {"property_prediction-bbbp", "property_prediction-clintox", 
 "property_prediction-hiv", "property_prediction-sider"}
 # answer_onlytasktypeextract
 return _extract_core_answer(
 text, task_name, text_tasks, smiles_tasks,
 formula_element_tasks, formula_split_tasks,
 number_tasks, boolean_tasks
 )
 
 # ifanswer_onlyraw_outputextract
 if not raw_output:
 return ""
 
 # defaulttasksetif
 if text_tasks is None:
 text_tasks = {"molecule_captioning"}
 if smiles_tasks is None:
 smiles_tasks = {"forward_synthesis", "retrosynthesis", "molecule_generation", "name_conversion-i2s"}
 if formula_element_tasks is None:
 formula_element_tasks = {"name_conversion-i2f", "name_conversion-s2f"}
 if formula_split_tasks is None:
 formula_split_tasks = {"name_conversion-s2i"}
 if number_tasks is None:
 number_tasks = {"property_prediction-esol", "property_prediction-lipo"}
 if boolean_tasks is None:
 boolean_tasks = {"property_prediction-bbbp", "property_prediction-clintox", 
 "property_prediction-hiv", "property_prediction-sider"}
 
 text = str(raw_output)
 
 # 1. extract <|im_start|>assistant [/INST] 
 assistant_text = ""
 if "[/INST]" in text:
 assistant_text = text.split("[/INST]")[-1]
 elif "<|im_start|>assistant" in text:
 # assistantstart
 assistant_start = text.find("<|im_start|>assistant")
 if assistant_start >= 0:
 # extractassistantlabel
 after_assistant = text[assistant_start + len("<|im_start|>assistant"):]
 # ifemptyskip
 after_assistant = after_assistant.lstrip()
 # if <|im_end|> <|im_start|>user
 if "<|im_end|>" in after_assistant:
 assistant_text = after_assistant.split("<|im_end|>")[0]
 elif "<|im_start|>" in after_assistant:
 assistant_text = after_assistant.split("<|im_start|>")[0]
 else:
 assistant_text = after_assistant
 else:
 # ifassistantlabeluse
 assistant_text = text
 
 # removeheadempty
 assistant_text = assistant_text.lstrip()
 
 # 2. process think label<think> <think>
 # strategy
 # a) removeallthinklabelextractthink
 # b) ifthinkextractthinklabel
 think_tags = [
 ("<think>", "</think>"), # thinklabel
 ("<think>", "</think>"), # redacted_reasoninglabel
 ]
 
 # extractthink
 text_without_think = assistant_text
 think_contents = [] # savethinklabel
 
 for open_tag, close_tag in think_tags:
 if open_tag not in text_without_think:
 continue
 
 # allthinklabel
 pattern = re.escape(open_tag) + r"(.*?)" + re.escape(close_tag)
 matches = list(re.finditer(pattern, text_without_think, re.DOTALL))
 
 if matches:
 # allthinklabel
 for match in matches:
 think_content = match.group(1).strip()
 if think_content:
 think_contents.append(think_content)
 
 # removeallthinklabelincludinglabelthink
 text_without_think = re.sub(pattern, "", text_without_think, flags=re.DOTALL)
 else:
 # iflabelremovelabel
 text_without_think = re.sub(re.escape(open_tag), "", text_without_think)
 
 # think
 text_without_think = text_without_think.strip()
 # removelabel
 if "<|im_end|>" in text_without_think:
 text_without_think = text_without_think.split("<|im_end|>")[0].strip()
 
 # 3. chooseusethinkifusethink
 if text_without_think and len(text_without_think) > 0:
 # usethink
 text = text_without_think
 elif think_contents:
 # ifthinkuselastthinklabel
 text = think_contents[-1]
 else:
 # ifuseoriginalassistantalreadyremovethinklabel
 text = assistant_text
 
 # 3. remove <|im_end|> 
 if "<|im_end|>" in text:
 text = text.split("<|im_end|>")[0]
 
 # 4. removeallinvalidlabel
 text = re.sub(r"<\|endoftext\|>", "", text)
 text = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", text) # removeall im_start...im_end 
 text = re.sub(r"<\|im_start\|>", "", text)
 text = re.sub(r"<\|im_end\|>", "", text)
 # remove think/redacted_reasoning labelifextract
 text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
 text = re.sub(r"<think>", "", text)
 text = re.sub(r"</think>", "", text)
 text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
 text = re.sub(r"<think>", "", text)
 text = re.sub(r"</think>", "", text)
 text = re.sub(r"</s>", "", text)
 text = re.sub(r"<s>", "", text)
 
 # 5. empty
 text = re.sub(r"\n\s*\n", "\n", text) # merge
 text = text.strip()
 
 # 6. according totasktypeextract
 return _extract_core_answer(
 text, task_name, text_tasks, smiles_tasks,
 formula_element_tasks, formula_split_tasks,
 number_tasks, boolean_tasks
 )

