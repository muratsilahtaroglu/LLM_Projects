from itertools import combinations
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from rouge_score import rouge_scorer

# 1. Kombinasyonel içerik oluşturma fonksiyonu
def generate_combinations(content_list):
    combined_contents = []
    for r in range(1, len(content_list) + 1):
        for combo in combinations(content_list, r):
            combined_contents.append(" ".join(combo))  # Combining content parts into one string
    return combined_contents

# 2. Cosine Similarity hesaplama fonksiyonu
def compute_cosine_similarity(contents, response):

    model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')  # You can change the model

    # Encode the response and contents using the SentenceTransformer
    response_embedding = model.encode(response, convert_to_tensor=True)
    content_embeddings = model.encode(contents, convert_to_tensor=True)
    
    # Compute cosine similarity between response and each content combination
    cosine_scores = util.pytorch_cos_sim(response_embedding, content_embeddings)[0]
    
    scores = []
    for i, content in enumerate(contents):
        scores.append({
            "Combination": content,
            "Cosine Similarity": cosine_scores[i].item()  # Use .item() to get a Python float value
        })
    
    return scores

def compute_rouge_scores_cosine_simularity(contents, response):


    model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')  # You can change the model

    # Encode the response and contents using the SentenceTransformer
    response_embedding = model.encode(response, convert_to_tensor=True)
    content_embeddings = model.encode(contents, convert_to_tensor=True)
    
    # Compute cosine similarity between response and each content combination
    cosine_scores = util.pytorch_cos_sim(response_embedding, content_embeddings)[0]

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = []
    
    for i, content in enumerate(contents):
        rouge_score = scorer.score(content, response)
        scores.append({
            "Combination": content,
            "ROUGE-1 (F1)": rouge_score['rouge1'].fmeasure,
            "ROUGE-L (F1)": rouge_score['rougeL'].fmeasure,
            "Cosine Similarity": cosine_scores[i].item()
        })
    
    return scores

# 3. Örnek içerik ve yanıt
content_list = ["Kan kanseri, kan hücrelerinde meydana gelen bir kanser türüdür.",
                "Genellikle beyaz kan hücrelerini etkiler.",
                "Kemik iliği bozukluklarına yol açar."]

generated_response = "Kan kanseri, beyaz kan hücrelerinin anormal çoğalmasıyla ortaya çıkar ve kemik iliğinde sorunlara neden olabilir."

# Initialize the SentenceTransformer model

# Kombinasyonları oluştur
combinations_result = generate_combinations(content_list)

# Cosine Similarity hesapla
cosine_scores = compute_rouge_scores_cosine_simularity(combinations_result, generated_response)

# 4. Sonuçları sıralı olarak yazdır
cosine_df = pd.DataFrame(cosine_scores)
sorted_df = cosine_df.sort_values(by=["ROUGE-L (F1)", "Cosine Similarity"], ascending=[False, False])

print(cosine_df)
