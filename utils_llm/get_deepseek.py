from openai import OpenAI


def get_llm(query):
    #调用deepseek v2
    client = OpenAI(api_key="sk-eba54da4994d4c9bab70408237163fab", base_url="https://api.deepseek.com")

    prompt = "你将扮演一个问题改写小助手，你仅需根据我给你的问题给出相应的同义改写即可，注意一定要与原句子语义完全一致，且意图清晰"

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
        stream=False
    )
    return response.choices[0].message.content

