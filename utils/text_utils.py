import torch

def de_embed_text(sentence_embedding_seq, bert_embedding=None, tokenizer=None)-> str:
    """ 
    Convert the text embedding back to text
    Finds the closest text to the embedding by comparing it with the BERT vocabulary.
    Args:
        sentence_embedding_seq (tensor): The text embedding to convert back to text.
    Returns:
        str: The approximate text.
    """
    import torch
    import torch.nn.functional as F

    embedding_matrix = bert_embedding.word_embeddings.weight.to(sentence_embedding_seq.device)  # Shape: [Vocab_size, Embedding_dim]


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

    # print("Tokens:", tokens)
    # print("Reconstructed Sentence:", sentence)
    
    return sentence


def get_text_from_tokens(token_ids, tokenizer) -> str:
    """
    Convert a list of token IDs to a string using the provided tokenizer.
    
    Args:
        token_ids (list or tensor): List or tensor of token IDs.
        tokenizer: Tokenizer with a method `convert_ids_to_tokens`.
    
    Returns:
        str: The reconstructed text.
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    sentence = tokenizer.convert_tokens_to_string(tokens)
    
    return sentence

