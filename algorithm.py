import torch
import torch.nn as nn

# H = embeds[starting_state]
def get_start_embeds(embed, embed_dict, start_state):
    start_state = torch.Tensor([word_to_emb[x] for x in start_state]).long()
    H = embed(start_state)
    X = embed(start_state)
    return H, X



if __name__=='__main__':
    embed = nn.Embedding(10, 3)
    word_to_emb = {i: i for i in range(10)}
    start_state = torch.randint(-1, 10, (81,)).tolist()
    print(get_start_embeds(embed, word_to_emb, start_state))
