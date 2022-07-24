from transformers import T5Tokenizer, TFMT5ForConditionalGeneration
from torch import cuda

t5_tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')
t5_model = TFMT5ForConditionalGeneration.from_pretrained('/content/drive/MyDrive/models/model_files/', from_pt=True)

def sentence_generate(text, model=t5_model, tokenizer=t5_tokenizer):
    device = 'cuda' if cuda.is_available() else 'cpu'
    inputs = tokenizer.prepare_seq2seq_batch(src_texts=text,
                                            return_tensors='tf',
                                            max_length=128).input_ids
    output = model.generate(inputs,
                        max_length=200,
                        repetition_penalty=20.0,
                        early_stopping=True,
                        num_beams=10)
    generated_text = tokenizer.decode(output[0],skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return generated_text