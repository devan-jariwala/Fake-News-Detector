from transformers import pipeline

# Load a pre-trained natural language inference model
nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")

def logic_check(article_text):
    """Checks if an article contains logically inconsistent statements."""
    
    hypothesis = "This article contains factual, coherent, and logically consistent information."

    result = nli_model(f"{article_text} </s> {hypothesis}")[0]

    label = result["label"]
    score = result["score"]

    if label == "ENTAILMENT":
        return "LOGICALLY SOUND", score
    else:
        return "LOGICALLY QUESTIONABLE", score

if __name__ == "__main__":
    text = input("Enter article text:\n")
    verdict, score = logic_check(text)
    print(f"Logic Check Result: {verdict} (confidence {score:.2f})")
