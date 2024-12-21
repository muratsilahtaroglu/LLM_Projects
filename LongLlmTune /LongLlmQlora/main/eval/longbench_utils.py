import re
import string
import difflib
import numpy as np
from fuzzywuzzy import fuzz
from typing import List
from collections import Counter
from rouge import Rouge

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def normalize_tr_answer(s):
    """Turkish normalization. Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def count_score(prediction, ground_truth, **kwargs):
    """
    Count the number of times a number appears in the prediction and divide by the total
    number of numbers in the prediction. This is used to score the "number" task in the
    Longbench dataset.
    """
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_score(prediction, ground_truth, **kwargs):
    """
    Scores the prediction for the retrieval task in the Longbench dataset.

    The ground_truth string should contain a substring of the form "Paragraph X", where
    X is the correct paragraph number. The prediction is scored by finding all numbers
    in the prediction string and counting the proportion of them that match the
    correct paragraph number.

    Parameters
    ----------
    prediction : str
        The model's prediction.
    ground_truth : str
        The correct answer.

    Returns
    -------
    score : float
        The proportion of numbers in the prediction that match the correct paragraph
        number.
    """
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def code_sim_score(prediction, ground_truth, **kwargs):
    """
    Scores the prediction for the code similarity task in the Longbench dataset.

    The prediction is scored by finding the first line that is not a code block and
    does not contain a comment, and then computing the fuzzy similarity between
    this line and the ground truth.

    Parameters
    ----------
    prediction : str
        The model's prediction.
    ground_truth : str
        The correct answer.

    Returns
    -------
    score : float
        The fuzzy similarity between the prediction and the ground truth.
    """
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return (fuzz.ratio(prediction, ground_truth) / 100)

def classification_score(prediction, ground_truth, **kwargs):
    """
    Scores the prediction for the classification task in the Longbench dataset.

    The prediction is scored by finding all exact matches of class names in the
    prediction, and then computing the proportion of exact matches that match the
    ground truth. If there are no exact matches, the prediction is scored by finding
    the best match using the SequenceMatcher, and then computing the proportion of
    the best match that matches the ground truth.

    Parameters
    ----------
    prediction : str
        The model's prediction.
    ground_truth : str
        The correct answer.
    all_classes : list of str
        The list of all class names.

    Returns
    -------
    score : float
        The proportion of the prediction that matches the ground truth.
    """
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if len(em_match_list) != 0:
        if ground_truth in em_match_list:
            score = (1.0 / len(em_match_list))
        else:
            score = 0.0
    else:
        best_match = None
        highest_similarity = 0
        for string in all_classes:
            similarity = difflib.SequenceMatcher(None, string, prediction).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = string
        score = float(best_match == ground_truth)
    return score

def rouge_score(prediction, ground_truth, **kwargs):
    """
    Computes the ROUGE score for a given prediction and ground truth.

    Parameters
    ----------
    prediction : str
        The model's prediction.
    ground_truth : str
        The correct answer.

    Returns
    -------
    score : float
        The ROUGE score.

    Notes
    -----
    If the ROUGE score computation fails, this function returns 0.0.
    """
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

def f1_score(prediction, ground_truth, **kwargs):
    """
    Computes the F1 score for a given prediction and ground truth.

    Parameters
    ----------
    prediction : list of str
        The model's prediction.
    ground_truth : list of str
        The correct answer.

    Returns
    -------
    score : float
        The F1 score.
    """
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    """
    Computes the F1 score for a given prediction and ground truth.

    Parameters
    ----------
    prediction : str
        The model's prediction.
    ground_truth : str
        The correct answer.

    Returns
    -------
    score : float
        The F1 score.

    Notes
    -----
    This function normalizes the prediction and ground truth strings using
    `normalize_answer` before computing the F1 score.
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

def qa_f1_score_tr(prediction, ground_truth, **kwargs):
    """
    Computes the F1 score for Turkish QA tasks.

    Parameters
    ----------
    prediction : str
        The model's prediction.
    ground_truth : str
        The correct answer.

    Returns
    -------
    score : float
        The F1 score.
    """
    normalized_prediction = normalize_tr_answer(prediction)
    normalized_ground_truth = normalize_tr_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

def scorer(dataset, predictions, answers, all_classes):
    """
    Computes the score for the given predictions and answers based on the dataset
    name.

    Parameters
    ----------
    dataset : str
        The name of the dataset.
    predictions : list of str
        The model's predictions.
    answers : list of list of str
        The correct answers.
    all_classes : list of str
        The list of all class names.

    Returns
    -------
    score : float
        The score for the given predictions and answers.
    """
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, DATASET2METRIC[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

DATASET2PROMPT = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_tr": "Aşağıdaki metni okuyun ve kısaca cevaplayın.\n\n{context}\n\nŞimdi, yukarıdaki metne dayanarak aşağıdaki soruya cevap verin, sadece cevabı yazın ve başka bir şey söylemeyin.\n\nSoru: {input}\nCevap:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "Lütfen verilen haberi bir kategoriye yerleştirin. Aşağıda bazı örnekler verilmiştir.\n\n{context}\n{input}",
    "passage_count": "Aşağıda Wikipedia'dan alınmış bazı paragraflar var. Bunlardan bazıları tekrarlanan paragraflar olabilir. Lütfen bu paragrafları dikkatlice okuyun ve tekrar edenleri çıkardıktan sonra kaç tane eşsiz paragraf olduğunu belirleyin. Başka bir şey yazmadan sadece sayıyı belirtin.\n\n{context}\n\nSonuç:",
    "passage_retrieval_tr": "Aşağıda 30 Wikipedia paragrafı ile bir özet verilmiştir. Lütfen özetin hangi paragraftan alındığını belirleyin.\n\n{context}\n\nAşağıda verilen bir özet:\n\n{input}\n\nLütfen özetin hangi paragraftan alındığını belirtin. Cevap formatı şu şekilde olmalıdır: \"Paragraf 1\", \"Paragraf 2\" vb.\n\nCevap:",
    "lcc": "Lütfen aşağıdaki kodu tamamlayın. \n{context}Sonraki kod satırı:\n",
    "repobench-p": "Lütfen aşağıdaki kodu tamamlayın. \n{context}{input}Sonraki kod satırı:\n"
}

DATASET2MAXNEWTOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_tr": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_tr": 32,
    "lcc": 64,
    "repobench-p": 64
}

DATASET2METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_tr": qa_f1_score_tr,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_tr": retrieval_score,
    "passage_count": count_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

DATASET2CATEGORY = {
    "narrativeqa": "EN Single-Doc QA",
    "qasper": "EN Single-Doc QA",
    "multifieldqa_en": "EN Single-Doc QA",
    "multifieldqa_tr": "TR Single-Doc QA",
    "hotpotqa": "EN Multi-Doc QA",
    "2wikimqa": "EN Multi-Doc QA",
    "musique": "EN Multi-Doc QA",
    "gov_report": "EN Summarization",
    "qmsum": "EN Summarization",
    "multi_news": "EN Summarization",
    "trec": "EN Few-Shot Learning",
    "triviaqa": "EN Few-Shot Learning",
    "samsum": "EN Few-Shot Learning",
    "lsht": "TR Few-Shot Learning",
    "passage_retrieval_tr": "TR Synthetic Task",
    "passage_count": "TR Synthetic Task",
    "lcc": "Code Completion",
    "repobench-p": "Code Completion",
}
