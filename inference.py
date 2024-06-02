from transformers import GPT2LMHeadModel, AutoTokenizer


def load_checkpoint_and_generate_text(checkpoint_dir, tokenizer, prompt, max_length=512, num_beams=5, top_k=50, top_p=0.95):
    # Load tokenizer and model from checkpoint
    model = GPT2LMHeadModel.from_pretrained(checkpoint_dir)

    # Encode the input prompt

    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    input_ids = input_ids[:, :-1]
    attention_mask = attention_mask[:, :-1]

    # Generate text using beam search and sampling
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=num_beams,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = "".join(generated_text.split())
    return generated_text


if __name__ == "__main__":
    pretrained_path = './gpt2-distil-chinese-cluecorpussmall'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, cache_dir=pretrained_path)

    # Generate text using the latest checkpoint
    prompt = "我们的坦克偷到了级，都是升一星"
    generated_text = load_checkpoint_and_generate_text('./models', tokenizer, prompt)
    print(generated_text)
