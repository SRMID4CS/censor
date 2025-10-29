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
    
    return sentence, tokens


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


def get_true_length(ground_truth_text_token_ids) -> int:
    """
    Get the true length of the ground truth text in terms of token count.
    Args:
        ground_truth_text (str): The ground truth text.
    Returns:
        int: The true length in terms of token count.
    """

    if ground_truth_text_token_ids.ndim == 1:
        ground_truth_text_token_ids = ground_truth_text_token_ids.unsqueeze(0)

    # Find length by [cls] and [sep] token positions
    true_length = 0
    if 101 in ground_truth_text_token_ids[0].tolist():
        cls_index = ground_truth_text_token_ids[0].tolist().index(101)
        if 102 in ground_truth_text_token_ids[0].tolist():
            sep_index = ground_truth_text_token_ids[0].tolist().index(102)
            true_length = sep_index - cls_index + 1  # +1 to include [SEP] token

    return true_length


def infer_label_length_convergence(sentence_embedding_seq, bert_embedding=None, tokenizer=None, true_length=None) -> tuple:
    """
    Infer the label length convergence from the caption optimization output.
    Args:
        cap_opt: The caption optimization output.
    Returns:
        tuple: A tuple containing the inferred length and a boolean indicating if the length is correct.
    """
    sentence, tokens = de_embed_text(sentence_embedding_seq, bert_embedding, tokenizer)
    # check the length by [cls] token and [sep] token positions
    cls_token_pos = tokens.index("[CLS]") if "[CLS]" in tokens else -1
    sep_token_pos = tokens.index("[SEP]") if "[SEP]" in tokens else -1
    if cls_token_pos != -1 and sep_token_pos != -1:
        inferred_length = sep_token_pos - cls_token_pos + 1  # +1 to include [SEP] token
    else:
        inferred_length = -1  # Unable to infer length

    is_length_correct = (inferred_length == true_length)  # Replace with actual correctness check
    return inferred_length, is_length_correct

def infer_label_perfect_match(sentence_embedding_seq, bert_embedding=None, tokenizer=None, ground_truth_text_token_ids=None) -> bool:
    """
    Infer if the reconstructed text perfectly matches the ground truth text.
    Args:
        cap_opt: The caption optimization output.
    Returns:
        bool: True if the reconstructed text matches the ground truth text, False otherwise.
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")
    
    if ground_truth_text_token_ids.ndim == 1:
        ground_truth_text_token_ids = ground_truth_text_token_ids.unsqueeze(0)

    # match only between [cls] and [sep] tokens
    reconstructed_text, tokens = de_embed_text(sentence_embedding_seq, bert_embedding, tokenizer)
    

    cls_token_pos = tokens.index("[CLS]") if "[CLS]" in tokens else -1
    sep_token_pos = tokens.index("[SEP]") if "[SEP]" in tokens else -1
    if cls_token_pos != -1 and sep_token_pos != -1:
        reconstructed_text_trim = tokenizer.convert_tokens_to_string(tokens[cls_token_pos:sep_token_pos + 1])
    else:
        reconstructed_text_trim = ""  # Unable to extract valid text for comparison

    gt_cls_token_pos = ground_truth_text_token_ids[0].tolist().index(101) if 101 in ground_truth_text_token_ids[0].tolist() else -1
    gt_sep_token_pos = ground_truth_text_token_ids[0].tolist().index(102) if 102 in ground_truth_text_token_ids[0].tolist() else -1
    if gt_cls_token_pos != -1 and gt_sep_token_pos != -1:
        gt_tokens = tokenizer.convert_ids_to_tokens(ground_truth_text_token_ids[0][gt_cls_token_pos:gt_sep_token_pos + 1])
        ground_truth_text_trim = tokenizer.convert_tokens_to_string(gt_tokens)
    else:
        ground_truth_text_trim = ""  # Unable to extract valid text for comparison

    # Check for perfect match
    return reconstructed_text_trim == ground_truth_text_trim and reconstructed_text_trim != ""
    