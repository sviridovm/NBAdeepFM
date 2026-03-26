    
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
        self.player_embedding = nn.Embedding(num_players, embed_dim)
        
        
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
        role_embeds = self.player_embedding(action_roles) # Shape: [Batch, 3, embed_dim]
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
    def __init__(self, num_players, embed_dim=64, num_heads=8, num_layers=3):
        super(NBATransformer, self).__init__()
        self.player_embedding = nn.Embedding(num_players + 1, embed_dim)
        self.putback_embedding = nn.Embedding(2, embedding_dim=embed_dim)
        self.freethrow_embedding = nn.Embedding(2, embedding_dim=embed_dim)
        
        self.hwp_proj_dim = 8
        
        self.distance_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),         # GELU is great here as it allows a smooth, non-linear curve
            nn.Linear(16, embed_dim)
        )
        
        self.hwp_projection = nn.Linear(3, self.hwp_proj_dim)
        
        self.hwp_mlp = nn.Sequential(
            nn.Linear(self.hwp_proj_dim, 16),
            nn.GELU(),
            nn.Linear(16, embed_dim)
        )
        
        
        self.layernorm = nn.LayerNorm(embed_dim)        

        # Self-Attention: Lineup interactions (The "Environment")
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.lineup_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        # cross attention for shooter, assister, defender to attend to the lineup context   
        self.role_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        
        # Output Heads
        self.id_classifier = nn.Linear(embed_dim, num_players) # For Masked Identity

    def forward(self, 
                lineup_ids,
                lineup_hwps,
                role_ids,
                role_hwps,
                is_putback,
                is_freethrow,
                shot_distance,
                mask_indices
                ):
        lineup_base_embeds = self.player_embedding(lineup_ids)
        
        lineup_projs = self.hwp_projection(lineup_hwps)
        lineup_hwp_embeds = self.hwp_mlp(lineup_projs)

        # concat hwp embeds to individual players
        lineup_embeds = torch.stack([lineup_hwp_embeds, lineup_base_embeds], dim = 0)
        lineup_embeds = torch.sum(lineup_embeds, dim=0)


        lineup_nodes = self.lineup_encoder(self.layernorm(lineup_embeds))
        # ----------
        
        shot_distance = shot_distance.unsqueeze(-1).float()
        shot_distance_embed = self.distance_mlp(shot_distance)
        free_throw_embed = self.freethrow_embedding(is_freethrow.long())
        putback_embed = self.putback_embedding(is_putback.long())
        
        play_context = torch.stack([shot_distance_embed, free_throw_embed, putback_embed], dim=0)
        play_context = torch.sum(play_context, dim=0)

        role_context = self.player_embedding(role_ids)
        role_hwp_projs = self.hwp_projection(role_hwps)
        role_hwp_embeds = self.hwp_mlp(role_hwp_projs)
        
        role_embeds = torch.stack([role_context, role_hwp_embeds,], dim=0)
        role_embeds = torch.sum(role_embeds, dim=0)
        role_embeds = role_embeds + play_context.unsqueeze(1)
        
        role_query = self.layernorm(role_embeds)
        
        attn_out, _ = self.role_attn(query=role_query, key=lineup_nodes, value=lineup_nodes)
        batch_size = attn_out.size(0)
        batch_indices = torch.arange(batch_size, device=attn_out.device)

        masked_tokens = attn_out[batch_indices, mask_indices, :]        
        id_logits = self.id_classifier(masked_tokens)        
        
        return id_logits
    
    
    

class NBATransformerPointsPredictor(nn.Module, PyTorchModelHubMixin):
    def __init__(self, pretrained_model, hidden_dim=64):
        super(NBATransformerPointsPredictor, self).__init__()
        
        self.player_embedding = pretrained_model.player_embedding
        self.hwp_mlp = pretrained_model.hwp_mlp
        self.hwp_projection = pretrained_model.hwp_projection
        self.layernorm = pretrained_model.layernorm


        # predict prob of [0, 1 , 2, 3, 4]
        self.point_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, lineup_ids, lineup_hwps,):
        lineup_base_embeds = self.player_embedding(lineup_ids)
        
        lineup_projs = self.hwp_projection(lineup_hwps)
        lineup_hwp_embeds = self.hwp_mlp(lineup_projs)

        # concat hwp embeds to individual players
        lineup_embeds = torch.stack([lineup_hwp_embeds, lineup_base_embeds], dim = 0)
        lineup_embeds = torch.sum(lineup_embeds, dim=0)
        lineup_nodes = self.lineup_encoder(self.layernorm(lineup_embeds))

        offense_nodes = lineup_nodes[:, :5, :]  # First 5 players [Batch, 5, hidden_dim]
        defense_nodes = lineup_nodes[:, 5:, :]  # Last 5 players [Batch, 5, hidden_dim]
        
        offense_pooled = offense_nodes.mean(dim=1) # [Batch, hidden_dim]
        defense_pooled = defense_nodes.mean(dim=1) # [Batch, hidden_dim]

        matchup_vector = torch.cat([offense_pooled, defense_pooled], dim=-1) 

        points_logits = self.points_head(matchup_vector)
        
        return points_logits

    


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