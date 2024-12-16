import pandas as pd
from summac.model_summac import SummaCConv
from evaluate import load
import textstat
from factsumm import FactSumm
# from alignscore import AlignScore
from secrets_file import hf_token
from huggingface_hub import InferenceClient
from transformers import LlamaTokenizer, AutoTokenizer
import tiktoken
from collections import defaultdict
import random
import time
from tqdm import tqdm
import csv
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# Evaluation metrics
def evaluate_summaries(original_text, reference_summary, generated_summary):
    performance_metrics = dict()
    factsumm = FactSumm()

    ## Relevance - ROUGE-1-2-L and BERTScore
    rouges = factsumm.calculate_rouge(reference_summary, generated_summary)
    performance_metrics['rouge-1'] = rouges[0]
    performance_metrics['rouge-2'] = rouges[1]
    performance_metrics['rouge-L'] = rouges[2]

    bertscores = factsumm.calculate_bert_score(reference_summary, generated_summary)
    performance_metrics['bertscore-p'] = bertscores[0]
    performance_metrics['bertscore-r'] = bertscores[1]
    performance_metrics['bertscore-f1'] = bertscores[2]

    ## Factuality - AlignScore and SummaC - need GPU and separate environments?
    DEVICE = "cpu" # "cuda:0"

    # alignscorer = AlignScore(model='roberta-base', batch_size=32, device=DEVICE, ckpt_path='AlignScore-base.ckpt', evaluation_mode='nli_sp')
    # alignscore = alignscorer.score(contexts=[original_text], claims=[generated_summary])
    # performance_metrics['AlignScore'] = alignscore

    model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device=DEVICE, start_file="default", agg="mean")
    SummaC = model_conv.score([original_text], [generated_summary])
    performance_metrics['SummaC'] = SummaC

    ## Readability - SARI, Flesch-Kincaid Grade Level (FKGL), and Dale-Chall Readability Score (DCRS)
    sari = load("sari")
    performance_metrics['SARI'] = sari.compute(sources=[original_text], predictions=[generated_summary], references=[[reference_summary]])['sari']
    performance_metrics['Flesch-Kincaid Grade Level (FKGL)'] = textstat.flesch_kincaid_grade(generated_summary)
    performance_metrics['Dale-Chall Readability Score (DCRS)'] = textstat.dale_chall_readability_score(generated_summary)

    return performance_metrics


if __name__ == "__main__":
    # Hyperparameters - 
    persona = True
    reflection = True
    evaluate_performance = False
    base_model = 'meta-llama/Llama-3.1-8B-Instruct' #'meta-llama/Llama-3.1-70B-Instruct'
    max_content_length = 7600 #31000
    temperature = 0.0
    max_tokens = 2500 # based on what we saw with average length of reference summaries
    
    # ['pmcid', 'pub_year', 'abstract', 'plain_language_summary', 'implications_for_practice', 'implications_for_research', 'rest_of_the_text', 'split']
    data = pd.read_csv('../data/split_processed_articles.csv')

    enc = tiktoken.encoding_for_model("gpt-4o")
    def truncate_context(enc, context, max_length):
        encodings = enc.encode(context)
        return enc.decode(encodings[:max_length])
    client = InferenceClient(api_key=hf_token)
    prompt_abstract = "Generate a detailed, formal summary of the following text, focusing on technical background, objectives, methodologies, key findings, and future research directions, assuming a high level of domain knowledge:\n"
    prompt_pls = "Provide a very basic summary of the following text, focusing only on the main ideas and avoiding any technical language, making it accessible for someone without any background in the subject:\n"
    prompt_i4p = "Summarize the following text, emphasizing key findings and practical implications, while assuming familiarity with the domain's core concepts and terminology:\n"
    prompt_i4r = "Summarize the following text in a simple, easy-to-understand manner, focusing on the main points and significance of the findings from a researchers perspective:\n"
    all_prompts = [prompt_abstract, prompt_pls, prompt_i4p, prompt_i4r]

    pa1 = defaultdict(list)
    pa2 = defaultdict(list)
    pa3 = defaultdict(list)
    pa4 = defaultdict(list)
    performance_averages = [pa1, pa2, pa3, pa4] # abstract, pls, i4p, i4r
    llm_outputs = 'reflective_persona_llm_outputs.csv'
    with open(llm_outputs, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ['pmcid', 'persona', 'intermediate_summaries', 'criticisms', 'abstract', 'generated_abstract', 'plain_language_summary', 'generated_plain_language_summary', 'implications_for_practice', 'generated_implications_for_practice', 'implications_for_research', 'generated_implications_for_research', 'rest_of_the_text']
        csvwriter.writerow(header)

        for index, row in tqdm(data.iterrows(), total=data.shape[0]):
            generated_personas = []
            intermediate_summaries = []
            generated_suggestions = []
            generated_summaries = []
            reference_summaries = [row['abstract'], row['plain_language_summary'], row['implications_for_practice'], row['implications_for_research']]
            for i, prompt in enumerate(all_prompts):
                time.sleep(random.randint(1, 3))
                content = prompt + row['rest_of_the_text']
                content = truncate_context(enc, content, max_content_length)

                if persona:
                    if reflection:
                        persona_prompt = "Given the following task and biomedical article, create a user persona for a person who would benefit from reading the task output relating to the article if accomplished. The persona should include details about their educational background, current role, interests, key skills, motivations, and common challenges they face.\n"
                        persona_content = persona_prompt + content
                        generate_persona_message = [{"role": "user", "content": persona_content}]
                        persona = client.chat.completions.create(
                            model=base_model, 
                            messages=generate_persona_message,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        persona_user = persona.choices[0].message['content'].replace('\n', ' ')
                        generated_personas.append(persona_user)
                        messages = [{"role": persona_user, "content": content}]
                        # user-persona generated summary
                        completion = client.chat.completions.create(
                            model=base_model, 
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )

                        summary = completion.choices[0].message['content'].replace('\n', ' ')
                        intermediate_summaries.append(summary)

                        content = "Given a summary, write a critique of what information you would want added. Also, write a critique of what is superfluous or unnecessary information.\nSummary:\n"
                        content = content + summary
                        content = truncate_context(enc, content, max_content_length)
                        messages = [{"role": persona_user, "content": content}]
                        # user-persona generated summary
                        critic = client.chat.completions.create(
                            model=base_model, 
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )

                        suggested_edits = critic.choices[0].message['content'].replace('\n', ' ')
                        generated_suggestions.append(suggested_edits)

                        content = f"Edit the summary to address the criticisms.\nCriticisms:\n{suggested_edits}\nSummary:\n{summary}"
                        content = truncate_context(enc, content, max_content_length)
                        messages = [{"role": persona_user, "content": content}]
                        # user-persona generated summary
                        editor = client.chat.completions.create(
                            model=base_model, 
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )

                        # final summary after edits
                        summary = editor.choices[0].message['content'].replace('\n', ' ')

                    else:
                        persona_prompt = "Given the following task and biomedical article, create a user persona for a person who would benefit from reading the task output relating to the article if accomplished. The persona should include details about their educational background, current role, interests, key skills, motivations, and common challenges they face.\n"
                        persona_content = persona_prompt + content
                        generate_persona_message = [{"role": "user", "content": persona_content}]
                        persona = client.chat.completions.create(
                            model=base_model, 
                            messages=generate_persona_message,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        generated_personas.append(persona.choices[0].message['content'].replace('\n', ' '))
                        messages = [{"role": persona.choices[0].message['content'].replace('\n', ' '), "content": content}]
                        # user-persona generated summary
                        completion = client.chat.completions.create(
                            model=base_model, 
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )

                        summary = completion.choices[0].message['content'].replace('\n', ' ')
                else:
                    if reflection:
                        pass
                    messages = [{"role": "user", "content": content}]
                
                    completion = client.chat.completions.create(
                        model=base_model, 
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    summary = completion.choices[0].message['content'].replace('\n', ' ')

                generated_summaries.append(summary)
                if evaluate_performance:
                    eval_metrics = evaluate_summaries(row['rest_of_the_text'], reference_summaries[i], generated_summaries[i]) # original_text, reference_summary, generated_summary
                    for metric in eval_metrics:
                        performance_averages[i][metric].append(eval_metrics[metric])

            row2write = [row['pmcid'], generated_personas, intermediate_summaries, generated_suggestions, row['abstract'], generated_summaries[0], row['plain_language_summary'], generated_summaries[1], row['implications_for_practice'], generated_summaries[2], row['implications_for_research'], generated_summaries[3], row['rest_of_the_text']]
            csvwriter.writerow(row2write)
    
    if evaluate_performance:
        for i, pa in enumerate(performance_averages):
            if i == 0:
                filename = 'abstract_performance.pkl'
            elif i == 1:
                filename = 'pls_performance.pkl'
            elif i == 2:
                filename = 'i4p_performance.pkl'
            elif i == 3:
                filename = 'i4r_performance.pkl'
            with open(filename, 'wb') as handle:
                    pickle.dump(pa, handle, protocol=pickle.HIGHEST_PROTOCOL)
            for p in pa:
                print_dict = {0: 'abstract', 1: 'pls', 2: 'i4p', 3: 'i4r'}
                print(print_dict[i], p)
                print(sum(pa[p]) / float(len(pa[p])))
