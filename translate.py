from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# parameters
input_length = 500
max_seq_length = 1024

# load scripts or text
tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M", token=True, src_lang="bod-Tibt"
)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

# Load txt
with open('C:/Users/16168/Documents/SideProjects/tib_bud/digital_tengyur/derge-tengyur/text/001_བསྟོད་ཚོགས།_ཀ.txt', 'r', encoding='utf-8') as f:
    lines = []
    for i in range(input_length):
        line = f.readline()
        if not line:
            break
        lines.append(line)

tibetan_text = ''.join(lines)

# Tokenize the entire Tibetan text
inputs = tokenizer(tibetan_text, return_tensors="pt")
input_ids = inputs["input_ids"].tolist()[0]

translated_chunks = []

while input_ids:
    chunk = input_ids[:max_seq_length]
    input_ids = input_ids[max_seq_length:]

    chunk_tokens = tokenizer.convert_ids_to_tokens(chunk)
    chunk_inputs = tokenizer(' '.join(chunk_tokens), return_tensors="pt", truncation=True, max_length=max_seq_length)
    translated_chunk = model.generate(
        **chunk_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"],
        max_length=30
    )

    translated_chunks.extend(translated_chunk.tolist())

full_translation = tokenizer.batch_decode(translated_chunks, skip_special_tokens=True)
full_translation = "".join(full_translation)

print(full_translation)
print('Done.')