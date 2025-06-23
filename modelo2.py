from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import torch

model_name = "google/pegasus-xsum"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

input_text = """
Transformar lixo plástico em substância para remédio parece coisa de ficção científica, mas é uma conquista anunciada por cientistas em estudo publicado na revista científica "Nature Chemistry". Em testes de pequena escala em laboratório, pesquisadores da Universidade de Edimburgo, no Reino Unido, usaram uma versão geneticamente modificada da bactéria Escherichia coli para converter plástico PET no princípio ativo do paracetamol, um dos analgésicos mais usados no mundo.

O PET, usado em garrafas e embalagens, serviu de matéria-prima para produzir uma substância essencial na medicina e de maneira surpreendentemente limpa: o processo foi realizado em temperatura ambiente, com praticamente zero emissão de carbono e em menos de 24 horas.
Um dos pontos curiosos do estudo é a chamada “reação de Lossen”, até então usada só em laboratório e sob condições rigorosas. Os cientistas descobriram que essa reação pode acontecer dentro da E. coli, em ambiente aquoso e com a ajuda apenas do fosfato — um componente já presente no meio de crescimento da bactéria.

Essa reação química permite transformar um tipo especial de composto — chamado hidroxamato — em uma amina, que é uma estrutura básica presente em muitas moléculas, inclusive medicamentos.

No caso do estudo, esse processo foi essencial para gerar o para-aminobenzoato (PABA), uma substância intermediária usada depois pela própria bactéria para fabricar o paracetamol. O mais surpreendente é que tudo isso ocorreu sem necessidade de metais pesados, calor extremo ou catalisadores artificiais — o fosfato, sozinho, foi suficiente para fazer a reação acontecer dentro das células vivas.
Para que a bactéria pudesse fazer o trabalho completo, os pesquisadores inseriram nela dois genes: um vindo de um cogumelo (Agaricus bisporus) e outro de uma bactéria do solo (Pseudomonas aeruginosa). Esses genes permitem que a E. coli transforme o produto derivado do plástico em paracetamol.

Todo o processo foi realizado em um único recipiente — o chamado método “one-pot” — e funciona em duas etapas: primeiro, a reação química transforma o PET em uma molécula intermediária; depois, a bactéria finaliza o trabalho, gerando o remédio.
Por enquanto, a transformação do plástico em paracetamol foi realizada apenas em pequena escala, dentro do laboratório. Os cientistas reconhecem que será necessário muito desenvolvimento até que o processo possa ser usado em indústrias.

Embora os resultados sejam promissores, a equipe enfatiza que "desenvolvimento adicional é necessário" antes que a produção em larga escala possa ser alcançada. Alguns desafios ainda precisam ser superados, como aumentar a concentração do material sem prejudicar as bactérias e garantir que o sistema funcione em biorreatores maiores. Também será necessário comparar os custos e benefícios ambientais em relação aos métodos tradicionais.

Mesmo assim, o avanço já aponta para um futuro em que o plástico descartável pode virar matéria-prima para medicamentos — reduzindo tanto o lixo quanto a dependência de combustíveis fósseis na indústria farmacêutica.
"""

input_text = preprocess_text(input_text)

prompt = "Summarize in 2-3 sentences: "
input_ids = tokenizer.encode(prompt + input_text, return_tensors="pt", max_length=512, truncation=True)

try:
    summary_ids = model.generate(
        input_ids,
        max_length=150,
        min_length=50,
        num_beams=6,
        length_penalty=0.8,
        no_repeat_ngram_size=2,
        early_stopping=True,
        do_sample=False
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
except Exception as e:
    print(f"Erro na geração: {e}")
    summary = "Falha na geração do resumo. Verifique os recursos computacionais ou parâmetros."

print("Texto original (primeiros 500 caracteres):\n")
print(input_text[:500] + "...")
print("\nResumo:\n")
print(summary)