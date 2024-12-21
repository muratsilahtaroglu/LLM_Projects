# Overview

This directory contains the official implementation of the training data generation process for ***"Extending Llama-3's Context Ten-Fold Overnight"***. The current version utilizes OpenAI's API to generate two main categories of data: *Book* and *Paper*. These categories are further subdivided into:

- **"One-detail QA"**: Focused on generating questions and answers centered around a single detail or fact from the source material, such as a specific event or concept.
- **"Multi-detail QA"**: Involves generating questions and answers that encompass multiple details, providing a broader understanding of the context or topic.
- **"Biography Summary"**: Summarizes biographical information from books or papers, capturing key events and achievements of individuals in a concise format.

In the future, this pipeline will be enhanced to:

- Integrate with local models for data generation.
- Include automatic Turkish translation for processed datasets.
- Provide further support for long-context data such as multi-chapter books, academic papers, and longitudinal datasets.

# File Structure

```bash
data_pipeline/
├── data                    # Directory to store intermediate and final processed data.
├── raw_data                # Directory to store raw datasets.
├── prepare_bio_book.ipynb  # Processes book data for biography summaries.
├── prepare_multi_details_book.ipynb  # Processes book data for multi-detail QA.
├── prepare_multi_details_paper_long.ipynb  # Processes paper data for multi-detail QA.
├── prepare_one_detail_book.ipynb  # Processes book data for one-detail QA.
├── prepare_one_detail_paper_long.ipynb  # Processes paper data for one-detail QA.
├── _openai.py              # Handles data generation via OpenAI's API (future integration with local models).
└── README.md               # Documentation for the data pipeline.
```

# Requirements

```bash
datasets==2.20.0
langdetect==1.0.9
semantic-text-splitter==0.13.3
tiktoken==0.7.0
```

# Usage

1. **Download Raw Data**: Place your raw dataset files into the `raw_data` directory.

2. **Process Raw Data**: Execute the corresponding notebook(s) for your dataset type. The notebooks will create temporary request files in the `data` directory. Use the following mapping:

   - `prepare_one_detail_book.ipynb`: One-detail QA for books.
   - `prepare_bio_book.ipynb`: Biography summaries for books.
   - `prepare_multi_details_book.ipynb`: Multi-detail QA for books.
   - `prepare_one_detail_paper_long.ipynb`: One-detail QA for papers.
   - `prepare_multi_details_paper_long.ipynb`: Multi-detail QA for papers.

3. **Generate Data via API**: Update your API key in `_openai.py` and execute the script:

   ```bash
   python _openai.py --request <file in `data`>
   ```

   The processed results will be saved in `data` as `{file_name}.result.jsonl`.

# Future Enhancements

- **Local Model Integration**: Replace reliance on OpenAI’s API with local models for data generation.
- **Turkish Translation**: Automatically translate processed datasets to Turkish and include them in the output pipeline.
- **Enhanced File Structure**: Introduce separate folders for localized and translated data.
- **Long-Context Data Handling**: Further optimize for processing datasets with extended contexts, such as:
  - Multi-chapter novels.
  - Long-form academic papers.
  - Historical timelines or longitudinal datasets.

# Notes

- To avoid overlapping training data, execute the notebooks in the following order:
  1. `prepare_one_detail_book`
  2. `prepare_bio_book`
  3. `prepare_multi_details_book`
  4. `prepare_one_detail_paper_long`
  5. `prepare_multi_details_paper_long`

