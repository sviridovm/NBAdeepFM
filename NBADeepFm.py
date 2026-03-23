    
from typing import List, Optional, TypeAlias
from huggingface_hub import PyTorchModelHubMixin
from pydantic import BaseModel

PlayerID: TypeAlias = int

class ShotCycle(BaseModel):
    offensive_players: List[PlayerID]
    defensive_players: List[PlayerID]
    shot_distance: int | float | None = None # none on free throws or turnovers
    shot_points_scored: int
    shooting_player: PlayerID
    defending_player: Optional[PlayerID] = None
    assisting_player: Optional[PlayerID] = None
    rebounding_player: Optional[PlayerID] = None
    is_and1: bool = False
    is_putback: bool = False
    is_freethrow: bool = False
    is_turnover: bool = False
    is_steal: bool = False


import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionLayer(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttentionLayer, self).__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

    def forward(self, x, mask=None):
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        return output, attention_weights
   
class NBADeepFM(nn.Module):
    def __init__(self, num_players, embed_dim=32):
        super(NBADeepFM, self).__init__()
        
        # 1. Latent Space (Shared for both components)
        self.embedding = nn.Embedding(num_players, embed_dim)
        
        # EmbeddingBag handles sum pooling for lineups automatically
        self.lineup_pooling = nn.EmbeddingBag(num_players, embed_dim, mode='sum')
        
        # 2. Linear "Skill" Layer (Equivalent to RAPM/Individual impact)
        self.linear = nn.Embedding(num_players, 1)
        
        # 3. Deep Component (Higher-order logic)
        # We have 10 players on court, so input is 10 * embed_dim
        self.deep_layers = nn.Sequential(
            nn.Linear(5 * embed_dim + 6, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, shot_cycle: ShotCycle):
        """
        shot_cycle: An instance of ShotCycle containing all relevant info for a single shot attempt
        """
        off_pooled = self.lineup_pooling(shot_cycle.offensive_players) # Shape: [Batch, embed_dim]
        def_pooled = self.lineup_pooling(shot_cycle.defensive_players)
        action_roles = torch.stack([shot_cycle.shooting_player, shot_cycle.assisting_player, shot_cycle.defending_player], dim=1) # Shape: [Batch, 3]

        # action_roles shape: [Batch, 3] -> (Shooter, Assister, Defender)
        role_embeds = self.embedding(action_roles) # Shape: [Batch, 3, embed_dim]
        role_flattened = role_embeds.view(role_embeds.size(0), -1)
        
        is_putback_embed = torch.tensor(shot_cycle.is_putback, dtype=torch.float).unsqueeze(1) # Shape: [Batch, 1]
        is_and1_embed = torch.tensor(shot_cycle.is_and1, dtype=torch.float).unsqueeze(1) # Shape: [Batch, 1]
        is_freethrow_embed = torch.tensor(shot_cycle.is_freethrow, dtype=torch.float).unsqueeze(1) # Shape: [Batch, 1]
        is_turnover_embed = torch.tensor(shot_cycle.is_turnover, dtype=torch.float).unsqueeze(1) # Shape: [Batch, 1]
        is_steal_embed = torch.tensor(shot_cycle.is_steal, dtype=torch.float).unsqueeze(1) # Shape: [Batch, 1]
        shot_distance_embed = torch.tensor(shot_cycle.shot_distance if shot_cycle.shot_distance is not None else 0, dtype=torch.float).unsqueeze(1) # Shape: [Batch, 1]
        

        info_embeds = torch.cat([is_putback_embed, is_and1_embed, is_freethrow_embed, is_turnover_embed, is_steal_embed, shot_distance_embed], dim=1) # Shape: [Batch, 6]
        
        combined = torch.cat([off_pooled, def_pooled, role_flattened, info_embeds], dim=1)
        
        return self.deep_layers(combined)
    


class NBATransformer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_players, embed_dim=32, num_heads=8, num_layers=3):
        super(NBATransformer, self).__init__()
        self.embedding = nn.Embedding(num_players + 1, embed_dim)
        self.layernorm = nn.LayerNorm(embed_dim)        

        # Self-Attention: Lineup interactions (The "Environment")
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.lineup_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        
        # cross attention for shooter, assister, defender to attend to the lineup context   
        self.role_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        
        # Output Heads
        self.id_classifier = nn.Linear(embed_dim, num_players) # For Masked Identity

        # Output: 4 classes (0, 1, 2, 3, or 4 points)
        self.point_classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5) 
        )
        

    def forward(self, lineup_ids, role_ids):
        lineup_nodes = self.lineup_encoder(self.layernorm(self.embedding(lineup_ids)))
        role_query = self.layernorm(self.embedding(role_ids))
        
        attn_out, _ = self.role_attn(query=role_query, key=lineup_nodes, value=lineup_nodes)
        
        points_logits = self.point_classifier(attn_out.mean(dim=1))
        
        id_logits = self.id_classifier(attn_out[:, 0, :])
        
        return points_logits, id_logits
    
    
    
    
class NBATransformerLearner(nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_players, embed_dim=32, num_heads=4, num_layers=2):
        super(NBATransformerLearner, self).__init__()

        """
        Same architecture just without priviliged info cross attention
        """
        # 1. Latent Space (Shared for both components)
        self.embedding = nn.Embedding(num_players, embed_dim)
        
        
        # Player-to-Player Interaction (Self-Attention)
        self.self_attn = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        
        # Output: 4 classes (0, 1, 2, 3, or 4 points)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5) 
        )
        

    def forward(self, shot_cycle: ShotCycle):
        lineup = torch.tensor(shot_cycle.offensive_players + shot_cycle.defensive_players, dtype=torch.long) # Shape: [10]
        lineup_embeds = self.embedding(lineup) # Shape: [10, embed_dim]
        lineup_feat = self.self_attn(lineup_embeds)
        
        return self.classifier(lineup_feat.mean(dim=1))
    
    
    
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, true_labels, T=2.0, alpha=0.5):
    # 1. KL Divergence (Matching the Teacher's Distribution)
    # We use "Temperature" (T) to soften the distribution for better learning
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    distill_loss = F.kl_div(student_log_probs, soft_targets, reduction='batchmean') * (T**2)
    
    # 2. Cross-Entropy (Matching the actual Points Scored)
    label_loss = F.cross_entropy(student_logits, true_labels)
    
    return alpha * distill_loss + (1 - alpha) * label_loss