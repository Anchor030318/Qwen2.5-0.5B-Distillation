import torch

def calculate_efficiency_loss_reward(
    unweighted_token_loss_batch, # Shape: [batch_size, seq_len-1]
    shifted_labels_batch,      # Shape: [batch_size, seq_len-1]
    shifted_answer_mask_batch, # Shape: [batch_size, seq_len-1], bool, 1 for answer tokens
    shift_binary_target_mask_batch, # Shape: [batch_size, seq_len-1], bool, 1 for target tokens (non-pad)
    eff_loss_threshold,
    eff_reward_coeff,
    eff_penalty_coeff,
    device
):
    """
    Calculates the efficiency loss/reward component.
    The "correctness" is based on the model's mean loss on the ground truth answer tokens.
    """
    L_eff_total = torch.tensor(0.0, device=device)
    num_samples_in_eff_calc = 0

    for i in range(shifted_labels_batch.size(0)): # Iterate over batch items
        unweighted_token_loss = unweighted_token_loss_batch[i]
        shifted_labels = shifted_labels_batch[i]
        shifted_answer_mask = shifted_answer_mask_batch[i]
        shift_binary_target_mask = shift_binary_target_mask_batch[i]

        # Consider only tokens that are part of the answer AND are actual targets (not padding/-100)
        ans_tokens_ground_truth_mask_i = shifted_answer_mask & shift_binary_target_mask
        
        if ans_tokens_ground_truth_mask_i.sum() > 0:
            num_samples_in_eff_calc += 1
            answer_length_i = ans_tokens_ground_truth_mask_i.sum().float()
            
            # Calculate mean loss ONLY on the ground truth answer tokens
            ans_token_losses_i = unweighted_token_loss[ans_tokens_ground_truth_mask_i]
            mean_ans_token_loss_i = ans_token_losses_i.mean()
            
            # norm_mean_ans_loss_i: close to 0 for "correct-like" (low loss), close to 1 for "incorrect-like" (high loss)
            norm_mean_ans_loss_i = torch.sigmoid(mean_ans_token_loss_i - eff_loss_threshold) 
            
            reward_i = -eff_reward_coeff * (1.0 / (answer_length_i + 1e-6)) * (1.0 - norm_mean_ans_loss_i)
            penalty_i = eff_penalty_coeff * answer_length_i * norm_mean_ans_loss_i
            
            L_eff_total += (reward_i + penalty_i)
    
    if num_samples_in_eff_calc > 0:
        return L_eff_total / num_samples_in_eff_calc
    return torch.tensor(0.0, device=device)


def calculate_repetition_penalty(
    shifted_labels_batch,      # Shape: [batch_size, seq_len-1]
    shifted_think_mask_batch, # Shape: [batch_size, seq_len-1], bool, 1 for THINKING tokens
    shift_binary_target_mask_batch, # Shape: [batch_size, seq_len-1], bool, 1 for target tokens (non-pad)
    rep_ngram_n,
    rep_penalty_coeff,
    device
):
    """
    Calculates the repetition penalty for n-grams within the THINKING section of the labels.
    """
    if rep_penalty_coeff <= 0: # No penalty if coefficient is zero or negative
        return torch.tensor(0.0, device=device)

    L_rep_total = torch.tensor(0.0, device=device)
    num_samples_in_rep_calc = 0

    for i in range(shifted_labels_batch.size(0)): # Iterate over batch items
        shifted_labels = shifted_labels_batch[i]
        # Use the think mask here
        shifted_section_mask_for_rep = shifted_think_mask_batch[i]
        shift_binary_target_mask = shift_binary_target_mask_batch[i]

        # Consider only tokens that are part of the thinking section AND are actual targets
        rep_tokens_ground_truth_mask_i = shifted_section_mask_for_rep & shift_binary_target_mask

        # Check if the number of tokens in the thinking section is greater than n-gram size
        if rep_tokens_ground_truth_mask_i.sum() > rep_ngram_n:
            num_samples_in_rep_calc += 1
            # Get actual ground truth thinking token IDs for n-gram analysis
            actual_section_token_ids_i = shifted_labels[rep_tokens_ground_truth_mask_i]
            
            # Ensure enough tokens for at least one n-gram
            if len(actual_section_token_ids_i) >= rep_ngram_n: 
                ngrams = {}
                num_repeated_ngrams = 0
                # Iterate through tokens to form n-grams
                for k in range(len(actual_section_token_ids_i) - rep_ngram_n + 1):
                    ngram_tokens = actual_section_token_ids_i[k : k + rep_ngram_n]
                    # Ensure tokens are integers for tuple conversion
                    ngram = tuple(ngram_tokens.int().tolist()) if ngram_tokens.is_floating_point() else tuple(ngram_tokens.tolist())

                    if ngram in ngrams:
                        ngrams[ngram] += 1
                        num_repeated_ngrams += 1 # Count all repetitions beyond the first occurrence
                    else:
                        ngrams[ngram] = 1
                
                num_possible_ngrams = len(actual_section_token_ids_i) - rep_ngram_n + 1
                if num_possible_ngrams > 0:
                    L_rep_total += rep_penalty_coeff * (num_repeated_ngrams / num_possible_ngrams)
    
    if num_samples_in_rep_calc > 0:
        return L_rep_total / num_samples_in_rep_calc
    return torch.tensor(0.0, device=device) 