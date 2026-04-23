# 🧠 Conversation Evaluation Engine (Multi-Facet Benchmark System)

## 🎯 Overview

This project implements a **production-ready evaluation system** that scores each conversation turn across **300+ facets** covering:

- Linguistic quality  
- Pragmatics  
- Safety  
- Emotion  

The system is designed to **scale to 5000+ facets without architectural changes**, satisfying all assignment constraints.

---

## 🧠 Key Features

- ✅ Multi-facet evaluation (300 → scalable to 5000+)
- ✅ Open-weight model support (≤16B, Groq Llama 3.1 8B)
- ✅ Structured scoring (1–5 scale per facet)
- ✅ Confidence estimation per score
- ✅ Async + batched inference
- ✅ Streamlit UI for interactive testing
- ✅ Fully Dockerized setup

---
## 🏗️ Architecture

```
User Input (Streamlit UI)
        ↓
Inference Engine (Async + Grouped Batching)
        ↓
Prompt Builder (Compressed multi-facet prompts)
        ↓
LLM Client (Groq - Llama 3.1 8B)
        ↓
JSON Parser + Validator
        ↓
Facet Registry (metadata + grouping)
        ↓
Final Scores + Confidence
```

---

## 🚀 Setup Instructions

### 🔹 1. Local Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a .env file:

```bash
GROQ_API_KEY=your_api_key_here
```

Run the app:
```bash
streamlit run app.py
```

### 🔹 2. Docker Setup (Recommended)

Build the image:

```bash
docker build -t convo-evaluator .
```

Run the container:

```bash
docker run -p 8501:8501 --env-file .env convo-evaluator
```

Open in browser:

```
http://localhost:8501
```

---

## 📊 Example Output

```json
{
  "facet_id": "FACET-023",
  "score": 4,
  "confidence": 0.82
}
```

---

## 🧪 Evaluation Pipeline

1. Load facets from registry  
2. Group into semantic batches  
3. Build compressed prompt  
4. Run async LLM inference  
5. Parse + validate JSON  
6. Reconstruct metadata  
7. Apply scoring logic  

---

## ⚡ Performance

| Metric | Value |
|------|------|
| Facets per turn | 300 |
| LLM calls | ~5–8 |
| Avg latency | ~2–5s |
| Scalability | 5000+ facets |

---

## 🧩 Limitations

- Confidence is heuristic (can be improved with calibration)
- No persistent caching yet
- Depends on external LLM latency

---

## 🚀 Future Improvements

- Two-pass evaluation (refine low-confidence facets)
- Caching layer for repeated inputs
- UI enhancements (charts, filtering)
- Smaller Docker image (remove heavy ML dependencies)

---

## 🏁 Conclusion

This system demonstrates a **scalable, production-ready architecture** for evaluating conversations across hundreds of facets efficiently, while maintaining modularity and extensibility.
