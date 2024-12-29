from itertools import combinations
from rouge_score import rouge_scorer

# 1. Kombinasyonel içerik oluşturma fonksiyonu
def generate_combinations(content_list):
    combined_contents = []
    for r in range(1, len(content_list) + 1):
        for combo in combinations(content_list, r):
            combined_contents.append("\n".join(combo))
    return combined_contents

# 2. ROUGE skorlarını hesaplama fonksiyonu
def compute_rouge_scores(contents, response):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = []
    
    for i, content in enumerate(contents):
        rouge_score = scorer.score(content, response)
        scores.append({
            "Combination": content,
            "ROUGE-1 (F1)": rouge_score['rouge1'].fmeasure,
            "ROUGE-L (F1)": rouge_score['rougeL'].fmeasure
        })
    
    return scores

# 3. Örnek içerik ve yanıt
content_list = ["Kan kanseri, kan hücrelerinde meydana gelen bir kanser türüdür.",
                "Genellikle beyaz kan hücrelerini etkiler.",
                "Kemik iliği bozukluklarına yol açar."]

generated_response = "Kan kanseri, beyaz kan hücrelerinin anormal çoğalmasıyla ortaya çıkar ve kemik iliğinde sorunlara neden olabilir."

# Kombinasyonları oluştur
combinations_result = generate_combinations(content_list)

# ROUGE skorlarını hesapla
rouge_scores = compute_rouge_scores(combinations_result, generated_response)

# 4. Sonuçları sıralı olarak yazdır
import pandas as pd
rouge_df = pd.DataFrame(rouge_scores)
rouge_df = rouge_df.sort_values(by="ROUGE-L (F1)", ascending=False)

print(rouge_df)