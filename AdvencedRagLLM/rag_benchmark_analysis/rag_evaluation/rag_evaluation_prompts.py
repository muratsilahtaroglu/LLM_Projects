evaluation_system_prompt = """You are a fair evaluator language model."""

evaluation_base_prompt = """### Task Description:
An instruction (including a topic and the retrieved data from the semantic search system) and a score rubric representing the evaluation criteria are given.
1. Write a detailed feedback that assesses the **relevance of the retrieved data to the topic** strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 10. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 10}}"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

### The topic to evaluate:
{topic}

### Retrieved Data to evaluate:
{content}

### Reference (Score 10):
The retrieved data is completely relevant to the topic, perfectly aligned, and exhaustive in addressing the topic. This includes cases where the retrieved data mirrors the topic phrasing perfectly, unless additional detail is explicitly required.

### Score Rubrics:
[How relevant is the retrieved data to the topic?]
Score 1: The retrieved data is completely irrelevant to the topic.
Score 2: The retrieved data is mostly irrelevant to the topic.
Score 3: The retrieved data is barely relevant to the topic.
Score 4: The retrieved data is somewhat relevant to the topic, but lacks significant alignment.
Score 5: The retrieved data is moderately relevant to the topic, addressing parts of it.
Score 6: The retrieved data is fairly relevant to the topic, providing reasonable alignment.
Score 7: The retrieved data is mostly relevant to the topic, covering most of its aspects.
Score 8: The retrieved data is highly relevant to the topic, addressing it thoroughly with minor gaps.
Score 9: The retrieved data is very highly relevant to the topic, with only negligible gaps.
Score 10: The retrieved data is completely relevant to the topic, perfectly aligned, and exhaustive (including cases of verbatim or identical matches unless additional detail is explicitly required).

### Feedback:"""


evaluation_base_prompt2 = """### Task Description:
An instruction (including a topic and the retrieved data from the semantic search part of the RAG system) and a score rubric representing the evaluation criteria are given.
1. Write a detailed feedback that assesses the **relevance of the retrieved data to the topic** strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

### The topic to evaluate:
{topic}

### Retrieved Data:
{content}

### Score Rubrics:
[How relevant is the retrieved data to the topic?]
Score 1: The retrieved data is completely incorrect, inaccurate, and/or not factual to the topic.
Score 2: The retrieved data is mostly incorrect, inaccurate, and/or not factual to the topic.
Score 3: The retrieved data is somewhat correct, accurate, and/or factual to the topic.
Score 4: The retrieved data is mostly correct, accurate, and factual to the topic.
Score 5: The retrieved data is completely correct, accurate, and factual to the topic.

### Feedback:"""


