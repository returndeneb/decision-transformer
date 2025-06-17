# Conceptual PyTorch Code for the Decision Transformer Model
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

class DecisionTransformer(nn.Module):
    """
    This model uses a GPT-2 architecture to model trajectories for autoregressive action prediction.
    """
    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_length = max_length

        # Utilize a pre-configured GPT-2 model from the Hugging Face transformers library
        # This serves as the powerful sequence modeling backbone
        config = transformers.GPT2Config(
            vocab_size=1,  # Not used for continuous inputs
            n_embd=hidden_size,
            **kwargs
        )
        self.transformer = transformers.GPT2Model(config)

        # Embedding layers for each input modality
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        
        # Action embeddings can be learned or fixed
        # Here we use a learnable embedding for continuous actions
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        # Layer normalization for the combined embeddings
        self.embed_ln = nn.LayerNorm(hidden_size)

        # Prediction head to map transformer output back to action space
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ( if action_tanh else))
        )

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape, states.shape

        # Create attention mask for padding if not provided
        if attention_mask is None:
            # Assume all tokens are valid (no padding)
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # 1. Embed each input modality
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # 2. Add positional (time) embeddings to each token
        # This broadcast-adds the time embedding to the modality embedding
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # 3. Interleave embeddings to form the input sequence: (R_1, s_1, a_1, R_2, s_2, a_2,...)
        # This is a key step to structure the data for the transformer
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # 4. Create the causal attention mask for the interleaved sequence
        # The mask needs to be 3x the original sequence length
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        # 5. Pass the sequence through the GPT-2 transformer
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # 6. Reshape and extract state representations to predict actions
        # We only use the output from state token positions to predict the next action
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        state_reps = x[:, 1]  # (batch_size, seq_length, hidden_size)

        # 7. Predict actions from the state representations
        action_preds = self.predict_action(state_reps)

        return action_preds
