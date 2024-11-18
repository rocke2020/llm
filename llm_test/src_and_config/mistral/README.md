---
license: apache-2.0
tags:
- generated_from_trainer
- mistral
- 7b
- calme
model-index:
- name: Calme-7B-Instruct-v0.2
  results: []
model_name: Calme-7B-Instruct-v0.2
inference: false
model_creator: MaziyarPanahi
pipeline_tag: text-generation
quantized_by: MaziyarPanahi
---

<img src="https://cdn-uploads.huggingface.co/production/uploads/5fd5e18a90b6dc4633f6d292/LzEf6vvq2qIiys-q7l9Hq.webp" width="550" />

# MaziyarPanahi/Calme-7B-Instruct-v0.2

## Model Description

Calme-7B is a state-of-the-art language model with 7 billion parameters, fine-tuned over high-quality datasets on top of Mistral-7B. The Calme-7B models excel in generating text that resonates with clarity, calmness, and coherence.

### How to Use

```python
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="MaziyarPanahi/Calme-7B-Instruct-v0.2")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("MaziyarPanahi/Calme-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("MaziyarPanahi/Calme-7B-Instruct-v0.2")
```

### Eval


| Metric    | [Mistral-7B Instruct v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) | [Calme-7B v0.1](https://huggingface.co/MaziyarPanahi/Calme-7B-Instruct-v0.1)  | [Calme-7B v0.2](https://huggingface.co/MaziyarPanahi/Calme-7B-Instruct-v0.2)  | [Calme-7B v0.3](https://huggingface.co/MaziyarPanahi/Calme-7B-Instruct-v0.3) | [Calme-7B v0.4](https://huggingface.co/MaziyarPanahi/Calme-7B-Instruct-v0.4)  | [Calme-7B v0.5](https://huggingface.co/MaziyarPanahi/Calme-7B-Instruct-v0.5)  | [Calme-4x7B v0.1](https://huggingface.co/MaziyarPanahi/Calme-4x7B-MoE-v0.1) | [Calme-4x7B v0.2](https://huggingface.co/MaziyarPanahi/Calme-4x7B-MoE-v0.2) |
|-----------|--------------------------|-------|-------|-------|-------|-------|------------|------------|
| ARC       | 63.14                    | 67.24 | 67.75 | 67.49 | 64.85 | 67.58 | 67.15      | 76.66      |
| HellaSwag | 84.88                    | 85.57 | 87.52 | 87.57 | 86.00 | 87.26 | 86.89      | 86.84      |
| TruthfulQA| 68.26                    | 59.38 | 78.41 | 78.31 | 70.52 | 74.03 | 73.30      | 73.06      |
| MMLU      | 60.78                    | 64.97 | 61.83 | 61.93 | 62.01 | 62.04 | 62.16      | 62.16      |
| Winogrande| 77.19                    | 83.35 | 82.08 | 82.32 | 79.48 | 81.85 | 80.82      | 81.06      |
| GSM8k     | 40.03                    | 69.29 | 73.09 | 73.09 | 77.79 | 73.54 | 74.53      | 75.66      |

Some extra information to help you pick the right `Calme-7B` model:

| Use Case Category                               | Recommended Calme-7B Model | Reason                                                                                   |
|-------------------------------------------------|-----------------------------|------------------------------------------------------------------------------------------|
| Educational Tools and Academic Research         | [Calme-7B v0.5](https://huggingface.co/MaziyarPanahi/Calme-7B-Instruct-v0.5)               | Balanced performance, especially strong in TruthfulQA for accuracy and broad knowledge.  |
| Commonsense Reasoning and Natural Language Apps | [Calme-7B v0.2](https://huggingface.co/MaziyarPanahi/Calme-7B-Instruct-v0.2) or [Calme-7B v0.3](https://huggingface.co/MaziyarPanahi/Calme-7B-Instruct-v0.3) | High performance in HellaSwag for understanding nuanced scenarios.                      |
| Trustworthy Information Retrieval Systems       | [Calme-7B v0.5](https://huggingface.co/MaziyarPanahi/Calme-7B-Instruct-v0.5)               | Highest score in TruthfulQA, indicating reliable factual information provision.          |
| Math Educational Software                       | [Calme-7B v0.4](https://huggingface.co/MaziyarPanahi/Calme-7B-Instruct-v0.4)               | Best performance in GSM8k, suitable for numerical reasoning and math problem-solving.    |
| Context Understanding and Disambiguation        | [Calme-7B v0.5](https://huggingface.co/MaziyarPanahi/Calme-7B-Instruct-v0.5)               | Solid performance in Winogrande, ideal for text with context and pronoun disambiguation. |


### Quantized Models

> I love how GGUF democratizes the use of Large Language Models (LLMs) on commodity hardware, more specifically, personal computers without any accelerated hardware. Because of this, I am committed to converting and quantizing any models I fine-tune to make them accessible to everyone!

- GGUF (2/3/4/5/6/8 bits): [MaziyarPanahi/Calme-7B-Instruct-v0.2-GGUF](https://huggingface.co/MaziyarPanahi/Calme-7B-Instruct-v0.2-GGUF)

## Examples

```
<s>[INST] You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

describe about pros and cons of docker system. [/INST]
```

<details>
  <summary>Show me the response</summary>
  
  ```

  ```
  
</details>


```

```

<details>
  <summary>Show me the response</summary>
  
  ```

  ```
  
</details>


```
<s> [INST] Mark is faster than Mary, Mary is faster than Joe. Is Joe faster than Mark? Let's think step by step [/INST]
```

<details>
  <summary>Show me the response</summary>
  
  ```

  ```
  
</details>


```

```

<details>
  <summary>Show me the response</summary>
  
  ``` 

  ```
  
</details>


```
<s> [INST] explain step by step 25-4*2+3=? [/INST]
```

<details>
  <summary>Show me the response</summary>
  
  ```

  ```  
</details>


**Multilingual:**

```
<s> [INST] Vous êtes un assistant utile, respectueux et honnête. Répondez toujours de la manière la plus utile possible, tout en étant sûr. Vos réponses ne doivent inclure aucun contenu nuisible, contraire à l'éthique, raciste, sexiste, toxique, dangereux ou illégal. Assurez-vous que vos réponses sont socialement impartiales et de nature positive.

Si une question n'a pas de sens ou n'est pas cohérente d'un point de vue factuel, expliquez pourquoi au lieu de répondre quelque chose d'incorrect. Si vous ne connaissez pas la réponse à une question, veuillez ne pas partager de fausses informations.

Décrivez les avantages et les inconvénients du système Docker.[/INST]
```

<details>
  <summary>Show me the response</summary>
  
  ```

  ```

<details>
  <summary>Show me the response</summary>
  
  ```

  ```
  
</details>


```
<s>[INST] Ви - корисний, поважний та чесний помічник. Завжди відповідайте максимально корисно, будучи безпечним. Ваші відповіді не повинні містити шкідливого, неетичного, расистського, сексистського, токсичного, небезпечного або нелегального контенту. Будь ласка, переконайтеся, що ваші відповіді соціально неупереджені та мають позитивний характер.

Якщо питання не має сенсу або не є фактично послідовним, поясніть чому, замість того, щоб відповідати щось некоректне. Якщо ви не знаєте відповіді на питання, будь ласка, не діліться неправдивою інформацією.

Опис про переваги та недоліки системи Docker.[/INST] 
```

<details>
  <summary>Show me the response</summary>
  
  ```

  ```
  
</details>