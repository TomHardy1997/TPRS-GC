import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedSurvLoss(nn.Module):
    """
    Combined survival loss function:
    - NLLSurvLoss (negative log-likelihood loss for survival analysis)
    - Rank Loss (for ranking risk scores)
    """
    
    def __init__(self, alpha=0.0, lambda_rank=0.5, eps=1e-7, reduction='mean'):
        super(CombinedSurvLoss, self).__init__()
        self.alpha = alpha
        self.lambda_rank = lambda_rank
        self.eps = eps
        self.reduction = reduction

    def forward(self, outputs, y, t, c):
        """
        Forward pass of the combined loss function.
        
        Args:
            outputs: Model predictions/logits
            y: Survival intervals
            t: Survival times
            c: Censorship indicators (0=event, 1=censored)
        """
        device = outputs.device
        y, t, c = y.to(device), t.to(device), c.to(device)
        
        # Loss 1: NLLSurvLoss
        loss_nll = self.nll_loss(
            h=outputs, 
            y=y.unsqueeze(dim=1), 
            c=c.unsqueeze(dim=1)
        )

        # Generate mask: 1 for event occurred, 0 for censored
        event_mask = (c == 0).float()

        # Loss 2: Rank Loss
        loss_rank = self.rank_loss(outputs=outputs, t=t, event_mask=event_mask)

        # Combine the losses
        combined_loss = loss_nll + self.lambda_rank * loss_rank
        return combined_loss

    def nll_loss(self, h, y, c):
        """
        NLLSurvLoss: The negative log-likelihood loss function for survival analysis.
        """
        y = y.type(torch.int64).to(h.device)
        c = c.type(torch.int64).to(h.device)

        hazards = torch.sigmoid(h)
        S = torch.cumprod(1 - hazards, dim=1)

        S_padded = torch.cat([torch.ones_like(c, device=h.device), S], 1)
        s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=self.eps)
        h_this = torch.gather(hazards, dim=1, index=y).clamp(min=self.eps)
        s_this = torch.gather(S_padded, dim=1, index=y + 1).clamp(min=self.eps)

        uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
        censored_loss = -c * torch.log(s_this)

        total_loss = censored_loss + uncensored_loss
        
        if self.alpha > 0:
            total_loss = (1 - self.alpha) * total_loss + self.alpha * uncensored_loss

        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")

    def rank_loss(self, outputs, t, event_mask):
        """
        Pairwise Rank Loss implementation.
        
        Args:
            outputs: Model's logits (before converting to risk scores)
            t: Survival times
            event_mask: Event mask (1 for event occurred, 0 for censored)
        """
        # Convert logits to risk scores
        risk_scores = -torch.sum(torch.cumprod(1 - torch.sigmoid(outputs), dim=1), dim=1)
        
        loss = 0.0
        count = 0
        
        for i in range(len(t)):
            if event_mask[i] == 1:  # Only consider uncensored events
                # Find all samples with longer survival times
                relevant_indices = (t > t[i]).nonzero(as_tuple=True)[0]
                if len(relevant_indices) > 0:
                    loss += torch.logsumexp(risk_scores[relevant_indices], dim=0) - risk_scores[i]
                    count += 1
        
        # Return the mean loss
        if count > 0:
            return loss / count
        else:
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)


