def de_embed_text(sentence_embedding_seq)-> str:
    """ 
    Convert the text embedding back to text
    Finds the closest text to the embedding by comparing it with the BERT vocabulary.
    Args:
        sentence_embedding_seq (tensor): The text embedding to convert back to text.
    Returns:
        str: The approximate text.
    """
    from transformers import BertTokenizer, BertModel
    import torch
    import torch.nn.functional as F

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(sentence_embedding_seq.device)

    embedding_matrix = model.embeddings.word_embeddings.weight  # Shape: [Vocab_size, Embedding_dim]

    # Normalize for cosine similarity
    normalized_matrix = F.normalize(embedding_matrix, dim=1)
    normalized_input = F.normalize(sentence_embedding_seq, dim=1)

    # Compute cosine similarity
    similarities = torch.matmul(normalized_input, normalized_matrix.T)  # Shape: [seq_len, vocab_size]

    # For each embedding vector, find the most similar token (top-1)
    top_token_ids = similarities.argmax(dim=1)  # Shape: [seq_len]

    # Convert token IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(top_token_ids)

    # Convert list of tokens to sentence
    sentence = tokenizer.convert_tokens_to_string(tokens)

    print("Tokens:", tokens)
    print("Reconstructed Sentence:", sentence)
    
    return sentence

