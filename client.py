# client.py
# client.py provides the Client class for federated learning clients, including benign and attacker clients.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from models import VGAE

# Client class for federated learning
class Client:

    def __init__(self, client_id: int, model: nn.Module, data_loader, lr, local_epochs, alpha):
        """
        Initialize a federated learning client.
        
        Args:
            client_id: Unique identifier for the client
            model: The neural network model (will be deep copied)
            data_loader: DataLoader for local training data
            lr: Learning rate for local training (must be provided, no default)
            local_epochs: Number of local training epochs per round (must be provided, no default)
            alpha: Proximal regularization coefficient α ∈ [0,1] from paper formula (1) (must be provided, no default)
        
        Note: All parameters must be explicitly provided. Default values are removed to prevent
        inconsistencies with config settings. See main.py for proper usage.
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.data_loader = data_loader
        self.lr = lr
        self.local_epochs = local_epochs
        self.alpha = alpha  # Regularization coefficient α ∈ [0,1] from paper formula (1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.current_round = 0
        self.is_attacker = False

    def reset_optimizer(self):
        """Reset the optimizer."""
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def set_round(self, round_num: int):
        """Set the current training round."""
        self.current_round = round_num

    def get_model_update(self, initial_params: torch.Tensor) -> torch.Tensor:
        """Calculate the model update (Current - Initial)."""
        current_params = self.model.get_flat_params()
        return current_params - initial_params

    def local_train(self, epochs=None) -> torch.Tensor:
        """Base local training method (to be overridden)."""
        raise NotImplementedError


# BenignClient class for benign clients
class BenignClient(Client):

    def prepare_for_round(self, round_num: int):
        """Benign clients do not require special preparation."""
        self.set_round(round_num)

    def local_train(self, epochs=None) -> torch.Tensor:
        """Perform local training - includes proximal regularization."""
        if epochs is None:
            epochs = self.local_epochs
            
        self.model.train()
        initial_params = self.model.get_flat_params().clone()
        
        # Proximal regularization coefficient (paper formula (1): α ∈ [0,1])
        mu = self.alpha

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            pbar = tqdm(self.data_loader,
                    desc=f'Client {self.client_id} - Epoch {epoch + 1}/{epochs}',
                    leave=False)
            
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                # NewsClassifierModel returns logits directly
                logits = outputs
                
                ce_loss = nn.CrossEntropyLoss()(logits, labels)
                
                # Add proximal regularization term
                current_params = self.model.get_flat_params()
                proximal_term = mu * torch.norm(current_params - initial_params) ** 2
                
                loss = ce_loss + proximal_term
                
                self.optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': loss.item()})
        
        return self.get_model_update(initial_params)

    def receive_benign_updates(self, updates: List[torch.Tensor]):
        # Benign clients do not use this method
        pass


# AttackerClient class for clients that perform attacks
class AttackerClient(Client):

    def __init__(self, client_id: int, model: nn.Module, data_manager,
                 data_indices, lr, local_epochs, alpha,
                 dim_reduction_size=10000, vgae_lambda=0.5,
                 vgae_epochs=20, vgae_lr=0.01, camouflage_steps=30, camouflage_lr=0.1,
                 lambda_proximity=1.0, lambda_aggregation=0.5, graph_threshold=0.5,
                 attack_start_round=10, lambda_attack=2.0, lambda_camouflage=0.3):
        """
        Initialize an attacker client with VGAE-based camouflage capabilities.
        
        Args:
            client_id: Unique identifier for the client
            model: The neural network model (will be deep copied)
            data_manager: DataManager instance for managing attacker data
            data_indices: List of data indices assigned to this client
            lr: Learning rate for local training (must be provided, no default)
            local_epochs: Number of local training epochs per round (must be provided, no default)
            alpha: Proximal regularization coefficient α ∈ [0,1] (must be provided, no default)
            dim_reduction_size: Dimensionality for feature reduction (default: 10000)
            vgae_lambda: Weight for preservation loss in camouflage (default: 0.5)
            vgae_epochs: Number of epochs for VGAE training (default: 20)
            vgae_lr: Learning rate for VGAE optimizer (default: 0.01)
            camouflage_steps: Number of optimization steps for camouflage (default: 30)
            camouflage_lr: Learning rate for camouflage optimization (default: 0.1)
            lambda_proximity: Weight for constraint (4b) proximity loss (default: 1.0)
            lambda_aggregation: Weight for constraint (4c) aggregation loss (default: 0.5)
            graph_threshold: Threshold for graph adjacency matrix binarization (default: 0.5)
            attack_start_round: Round when attack phase starts (default: 10)
            lambda_attack: Weight for attack objective loss (default: 2.0) - CRITICAL for ASR
            lambda_camouflage: Weight for camouflage loss (default: 0.3) - Lower to preserve attack
        
        Note: lr, local_epochs, and alpha must be explicitly provided to ensure consistency
        with config settings. Other parameters have defaults but should be set via config in main.py.
        """
        self.data_manager = data_manager
        self.data_indices = data_indices
        
        # Store parameters first (before using them)
        self.attack_start_round = attack_start_round
        self.vgae_lambda = vgae_lambda
        self.dim_reduction_size = dim_reduction_size
        self.vgae_epochs = vgae_epochs
        self.vgae_lr = vgae_lr
        self.camouflage_steps = camouflage_steps
        self.camouflage_lr = camouflage_lr
        self.lambda_proximity = lambda_proximity
        self.lambda_aggregation = lambda_aggregation
        self.graph_threshold = graph_threshold
        self.lambda_attack = lambda_attack  # Weight for attack objective (Formula 4a)
        self.lambda_camouflage = lambda_camouflage  # Weight for camouflage (reduced to preserve attack)

        dummy_loader = data_manager.get_attacker_data_loader(client_id, data_indices, 0, self.attack_start_round)
        super().__init__(client_id, model, dummy_loader, lr, local_epochs, alpha)
        self.is_attacker = True

        # VGAE components
        self.vgae = None
        self.vgae_optimizer = None
        self.benign_updates = []
        self.feature_indices = None
        
        # Store original business data loader (for attack loss computation)
        # This contains Business samples with ORIGINAL labels (not flipped)
        self.original_business_loader = data_manager.get_attacker_original_business_loader(
            client_id, data_indices
        )
        
        # Formula 4 constraints parameters
        self.d_T = None  # Distance threshold for constraint (4b): d(w'_j(t), w_g(t)) ≤ d_T
        self.gamma = None  # Upper bound for constraint (4c): Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ
        self.global_model_params = None  # Store global model params for constraint (4b)

    def prepare_for_round(self, round_num: int):
        """Prepare for a new training round."""
        self.set_round(round_num)
        # Update dataloader with progressive poisoning logic
        self.data_loader = self.data_manager.get_attacker_data_loader(
            self.client_id, self.data_indices, round_num, self.attack_start_round
        )

    def receive_benign_updates(self, updates: List[torch.Tensor]):
        """Receive updates from benign clients."""
        # Store detached copies to avoid graph retention issues
        self.benign_updates = [u.detach().clone() for u in updates]

    def local_train(self, epochs=None) -> torch.Tensor:
        """
        Perform local training to get the initial malicious update.
        According to paper formula (1): F(w_i(t)) = (1/D_i(t)) * Σ f(...) + α ζ(w_i(t))
        """
        if epochs is None:
            epochs = self.local_epochs

        self.model.train()
        initial_params = self.model.get_flat_params().clone()
        
        # Proximal regularization coefficient (paper formula (1): α ∈ [0,1])
        mu = self.alpha
        
        # 1. Standard training on poisoned data with regularization
        for epoch in range(epochs):
            for batch in self.data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                ce_loss = nn.CrossEntropyLoss()(outputs, labels)
                
                # Add proximal regularization term (paper formula (1))
                current_params = self.model.get_flat_params()
                proximal_term = mu * torch.norm(current_params - initial_params) ** 2
                loss = ce_loss + proximal_term

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
        
        # Get raw malicious update
        poisoned_update = self.get_model_update(initial_params)
        
        # NOTE: Do NOT apply camouflage here!
        # Camouflage is applied in server.run_round() after collecting benign updates.
        # Applying it here would cause double camouflage.
        
        return poisoned_update

    def _get_reduced_features(self, updates: List[torch.Tensor], fix_indices=True) -> torch.Tensor:
        """
        Helper function to reduce dimensionality of updates.
        Randomly selects indices to slice the high-dimensional vector.
        """
        stacked_updates = torch.stack(updates)
        total_dim = stacked_updates.shape[1]
        
        # 如果更新维度小于降维目标，则不降维
        if total_dim <= self.dim_reduction_size:
            return stacked_updates
            
        # 每一轮攻击开始时，固定一组特征索引，保证这一轮内的训练一致性
        if self.feature_indices is None or not fix_indices:
            # Randomly select indices
            self.feature_indices = torch.randperm(total_dim)[:self.dim_reduction_size].to(self.device)
            
        # Select features
        reduced_features = torch.index_select(stacked_updates, 1, self.feature_indices)
        return reduced_features

    def _construct_graph(self, reduced_features: torch.Tensor):
        """
        Construct graph according to the paper (Section III).
        
        Paper formulation:
        - Feature matrix F(t) = [w_1(t), ..., w_i(t)]^T ∈ R^{I×M}
        - Adjacency matrix A(t) ∈ R^{M×M} (NOT I×I!)
        - δ_{m,m'} = cosine similarity between w_m(t) and w_{m'}(t)
        - w_m(t) ∈ R^{I×1} is the m-th COLUMN of F(t)
        
        So we need to compute similarity between COLUMNS (parameter dimensions),
        not ROWS (clients).
        """
        # reduced_features shape: (I, M) where I=num_clients, M=feature_dim
        # We need to compute similarity between columns (parameter dimensions)
        # Transpose to get (M, I), then compute similarity
        
        # F^T shape: (M, I) - each row is a parameter dimension across all clients
        features_transposed = reduced_features.t()  # (M, I)
        
        # Normalize for cosine similarity (along dim=1, i.e., across clients)
        norm_features = F.normalize(features_transposed, p=2, dim=1)  # (M, I)
        
        # Compute adjacency matrix A ∈ R^{M×M}
        # A[m, m'] = cosine_sim(w_m, w_m') where w_m is m-th column of F
        similarity_matrix = torch.mm(norm_features, norm_features.t())  # (M, M)
        
        # Remove self-loops
        adj_matrix = similarity_matrix.clone()
        adj_matrix.fill_diagonal_(0)
        
        # Threshold for binarization (paper doesn't specify, but common practice)
        adj_matrix = (adj_matrix > self.graph_threshold).float()
        
        return adj_matrix

    def _train_vgae(self, adj_matrix: torch.Tensor, feature_matrix: torch.Tensor, epochs=None):
        """
        Train the VGAE model according to the paper.
        
        Paper formulation:
        - Input: A ∈ R^{M×M} (adjacency), F ∈ R^{I×M} (features)
        - For VGAE, we use F^T ∈ R^{M×I} as node features
        - Each node represents a parameter dimension
        - VGAE learns to reconstruct A
        """
        if epochs is None:
            epochs = self.vgae_epochs
        
        # adj_matrix shape: (M, M) - from _construct_graph
        # feature_matrix shape: (I, M) - original features
        # For VGAE input, we use F^T as node features: (M, I)
        node_features = feature_matrix.t()  # (M, I)
        
        input_dim = node_features.shape[1]  # I (number of clients)
        num_nodes = node_features.shape[0]  # M (feature dimension)
        
        # Initialize VGAE if needed
        # Paper: input_dim = I (number of clients/benign models)
        if self.vgae is None or self.vgae.gc1.weight.shape[0] != input_dim:
            # Following paper: hidden1_dim=32, hidden2_dim=16
            hidden_dim = 32
            latent_dim = 16
            self.vgae = VGAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, dropout=0.0).to(self.device)
            self.vgae_optimizer = optim.Adam(self.vgae.parameters(), lr=self.vgae_lr)

        self.vgae.train()
        
        for _ in range(epochs):
            self.vgae_optimizer.zero_grad()
            
            # Forward pass: VGAE takes (node_features, adj_matrix)
            # node_features: (M, I), adj_matrix: (M, M)
            adj_recon, mu, logvar = self.vgae(node_features, adj_matrix)
            
            # Loss calculation
            loss = self.vgae.loss_function(adj_recon, adj_matrix, mu, logvar)
            
            loss.backward()
            self.vgae_optimizer.step()
        
        return adj_recon.detach()  # Return reconstructed adjacency for GSP

    def set_global_model_params(self, global_params: torch.Tensor):
        """Set global model parameters for constraint (4b) calculation."""
        self.global_model_params = global_params.clone().detach().to(self.device)
    
    def set_constraint_params(self, d_T: float = None, gamma: float = None):
        """Set constraint parameters for Formula 4."""
        self.d_T = d_T  # Constraint (4b): d(w'_j(t), w_g(t)) ≤ d_T
        self.gamma = gamma  # Constraint (4c): Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ

    def _compute_attack_loss(self, malicious_update: torch.Tensor) -> torch.Tensor:
        """
        Compute attack loss using a DIRECT and EFFECTIVE approach.
        
        NEW STRATEGY: Instead of trying to compute model outputs (which has gradient issues),
        we use a direction-based attack loss that encourages the malicious update to:
        1. Be different from benign updates (attack direction)
        2. But not too different (avoid detection)
        
        The key insight: The REAL attack comes from training on poisoned data (in local_train).
        This function just helps PRESERVE that attack while adding camouflage.
        
        Returns a loss that should be MAXIMIZED (caller will negate it).
        """
        if not self.benign_updates:
            # No benign updates to compare, return zero
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Compute benign update statistics
        benign_stack = torch.stack(self.benign_updates)
        benign_mean = benign_stack.mean(dim=0)
        
        # ATTACK STRATEGY: Encourage update to be in a specific "attack direction"
        # The poisoned_update from local_train already contains attack information
        # We want to preserve this attack direction while camouflaging
        
        # Method 1: Maximize distance from benign mean (encourages distinct attack)
        # But this conflicts with camouflage, so we use a softer version
        distance_from_benign = torch.norm(malicious_update - benign_mean)
        
        # Method 2: Maximize update magnitude (stronger attack)
        # Larger updates have more impact on the global model
        update_magnitude = torch.norm(malicious_update)
        
        # Method 3: Encourage alignment with the original poisoned direction
        # This is implicitly handled by loss_preservation in camouflage_update
        
        # Combined attack loss: We want to MAXIMIZE this
        # Higher distance = more distinct from benign (stronger attack)
        # Higher magnitude = larger impact on global model
        attack_loss = 0.5 * distance_from_benign + 0.5 * update_magnitude
        
        return attack_loss

    def _gsp_generate_malicious(self, feature_matrix: torch.Tensor, 
                                  adj_orig: torch.Tensor, adj_recon: torch.Tensor,
                                  poisoned_update: torch.Tensor) -> torch.Tensor:
        """
        Graph Signal Processing (GSP) module according to the paper (Section III).
        
        Paper formulation:
        1. L = diag(A·1) - A                 (Laplacian of original graph)
        2. L = B Λ B^T                       (SVD decomposition)
        3. S = F · B                         (GFT coefficient matrix)
        4. L̂ = diag(Â·1) - Â                 (Laplacian of reconstructed graph)
        5. L̂ = B̂ Λ̂ B̂^T                       (SVD decomposition)
        6. F̂ = S · B̂^T                       (Reconstructed feature matrix)
        7. w'_j(t) selected from F̂           (Malicious model)
        
        Args:
            feature_matrix: F ∈ R^{I×M} - benign model features
            adj_orig: A ∈ R^{M×M} - original adjacency matrix
            adj_recon: Â ∈ R^{M×M} - reconstructed adjacency matrix from VGAE
            poisoned_update: The poisoned update from local training
            
        Returns:
            Malicious update generated using GSP
        """
        # Step 1: Compute Laplacian of original graph
        # L = diag(A·1) - A
        degree_orig = adj_orig.sum(dim=1)
        L_orig = torch.diag(degree_orig) - adj_orig  # (M, M)
        
        # Step 2: SVD of original Laplacian
        # L = B Λ B^T
        try:
            U_orig, S_orig, Vh_orig = torch.linalg.svd(L_orig, full_matrices=True)
            B_orig = U_orig  # GFT basis (M, M)
        except:
            # Fallback if SVD fails
            print(f"    [Attacker {self.client_id}] SVD failed, using fallback")
            return poisoned_update
        
        # Step 3: Compute GFT coefficient matrix
        # S = F · B where F ∈ R^{I×M}, B ∈ R^{M×M}
        S = torch.mm(feature_matrix, B_orig)  # (I, M)
        
        # Step 4: Compute Laplacian of reconstructed graph
        # L̂ = diag(Â·1) - Â
        degree_recon = adj_recon.sum(dim=1)
        L_recon = torch.diag(degree_recon) - adj_recon  # (M, M)
        
        # Step 5: SVD of reconstructed Laplacian
        try:
            U_recon, S_recon, Vh_recon = torch.linalg.svd(L_recon, full_matrices=True)
            B_recon = U_recon  # New GFT basis (M, M)
        except:
            print(f"    [Attacker {self.client_id}] SVD of recon failed, using fallback")
            return poisoned_update
        
        # Step 6: Generate reconstructed feature matrix
        # F̂ = S · B̂^T where S ∈ R^{I×M}, B̂ ∈ R^{M×M}
        F_recon = torch.mm(S, B_recon.t())  # (I, M)
        
        # Step 7: Generate malicious update
        # Paper: "vectors w'_j(t) in F̂ are selected as malicious local models"
        # We combine the reconstructed features with the poisoned update direction
        
        # Method: Use weighted sum of reconstructed features, biased towards attack
        # Following reference code: w_attack = sum(new_features) / n * random_noise
        malicious_direction = F_recon.mean(dim=0)  # (M,)
        
        # Scale by random factor (similar to reference code)
        random_scale = torch.empty(1, device=self.device).uniform_(-0.5, 0.1).item()
        gsp_attack = malicious_direction * random_scale
        
        return gsp_attack

    def camouflage_update(self, poisoned_update: torch.Tensor) -> torch.Tensor:
        """
        GRMP Attack using VGAE + GSP according to the paper (Section III).
        
        Paper Algorithm 1:
        1. Calculate A according to cosine similarity (eq. 8)
        2. Train VGAE to maximize L_loss (eq. 12), obtain optimal Â
        3. Use GSP module to obtain F̂, determine w'_j(t) based on F̂
        """
        if not self.benign_updates:
            print(f"    [Attacker {self.client_id}] No benign updates, using raw poisoned update")
            return poisoned_update

        # Reset feature indices for this session
        self.feature_indices = None
        
        # ============================================================
        # STEP 1: Prepare feature matrix F ∈ R^{I×M}
        # ============================================================
        benign_stack = torch.stack([u.detach() for u in self.benign_updates])  # (I, full_dim)
        
        # Reduce dimensionality for computational efficiency
        reduced_benign = self._get_reduced_features(self.benign_updates, fix_indices=False)  # (I, M)
        M = reduced_benign.shape[1]
        I = reduced_benign.shape[0]
        
        # ============================================================
        # STEP 2: Construct adjacency matrix A ∈ R^{M×M}
        # According to paper eq. (8): δ_{m,m'} = cosine_sim(w_m, w_m')
        # ============================================================
        adj_matrix = self._construct_graph(reduced_benign)  # (M, M)
        
        # ============================================================
        # STEP 3: Train VGAE to learn graph structure
        # Paper: "Train VGAE to maximize L_loss"
        # ============================================================
        adj_recon = self._train_vgae(adj_matrix, reduced_benign)  # Returns Â
        
        # ============================================================
        # STEP 4: GSP module to generate malicious update
        # Paper: "Use GSP module to obtain F̂, determine w'_j(t)"
        # ============================================================
        gsp_attack_reduced = self._gsp_generate_malicious(
            reduced_benign, adj_matrix, adj_recon, poisoned_update
        )
        
        # ============================================================
        # STEP 5: Combine GSP attack with poisoned update
        # The GSP attack provides the camouflage direction
        # The poisoned update provides the attack direction
        # ============================================================
        
        # Expand GSP attack back to full dimension
        # Only modify the reduced indices, keep rest from poisoned_update
        malicious_update = poisoned_update.clone()
        
        if self.feature_indices is not None and gsp_attack_reduced is not None:
            # Blend: combine poisoned update with GSP-generated attack
            # Paper: attack vector replaces part of the benign weights
            blend_ratio = 0.7  # 70% poisoned (attack), 30% GSP (camouflage)
            
            # Get the reduced portion of poisoned update
            poisoned_reduced = poisoned_update[self.feature_indices]
            
            # Blend poisoned with GSP attack
            blended_reduced = blend_ratio * poisoned_reduced + (1 - blend_ratio) * gsp_attack_reduced
            
            # Put back into full update
            malicious_update[self.feature_indices] = blended_reduced
        
        # ============================================================
        # STEP 6: Apply constraint (4b): d(w'_j, w'_g) ≤ d_T
        # ============================================================
        benign_norms = torch.stack([torch.norm(u) for u in self.benign_updates])
        target_norm = benign_norms.mean() + 0.3 * benign_norms.std()
        
        current_norm = torch.norm(malicious_update)
        if current_norm > 1e-8:
            scale_factor = target_norm / current_norm
            scale_factor = torch.clamp(scale_factor, 0.5, 1.5)
            malicious_update = malicious_update * scale_factor
        
        if self.d_T is not None:
            update_norm = torch.norm(malicious_update)
            if update_norm > self.d_T:
                malicious_update = malicious_update * (self.d_T / update_norm)
        
        print(f"    [Attacker {self.client_id}] GSP Attack: norm={torch.norm(malicious_update):.4f}")
        
        return malicious_update.detach()
        
        # ============================================================
        # STEP 4: Scale to match benign update statistics
        # This helps evade norm-based detection while preserving direction
        # ============================================================
        
        # Target norm: slightly above benign average (more impact but not suspicious)
        target_norm = benign_norms.mean() + 0.3 * benign_norms.std()
        
        current_norm = torch.norm(blended_update)
        if current_norm > 1e-8:
            # Scale to target norm
            scale_factor = target_norm / current_norm
            # Limit scaling to reasonable range
            scale_factor = torch.clamp(scale_factor, 0.5, 1.5)
            blended_update = blended_update * scale_factor
        
        # ============================================================
        # STEP 5: Apply constraint (4b) if needed
        # d(w'_j(t), w_g(t)) ≤ d_T
        # ============================================================
        if self.d_T is not None:
            update_norm = torch.norm(blended_update)
            if update_norm > self.d_T:
                blended_update = blended_update * (self.d_T / update_norm)
        
        print(f"    [Attacker {self.client_id}] Camouflage: blend={blend_ratio:.1%}, "
              f"norm={torch.norm(blended_update):.4f} (benign avg={benign_norms.mean():.4f})")
        
        return blended_update.detach()