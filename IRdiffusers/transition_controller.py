import torch


class TransitionController:
    def __init__(self, batch_size, device, total_steps, warmup=0, force_ratio=0.2):
        self.batch_size = batch_size
        self.device = device
        self.warmup = warmup
        self.force_step_limit = int(total_steps * force_ratio)
        
        self.is_stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.stop_steps = torch.full((batch_size,), -1, dtype=torch.long, device=device)

        self.prev_vals = None


    def update(self, current_vals, step_idx):
        if step_idx < self.warmup:
            self._shift_history(current_vals)
            return self.is_stopped

        if self.prev_vals is None:
            self._shift_history(current_vals)
            return self.is_stopped

        is_non_increasing = current_vals <= self.prev_vals
        
        just_stopped = is_non_increasing & (~self.is_stopped)
        if just_stopped.any():
            self.is_stopped[just_stopped] = True
            self.stop_steps[just_stopped] = step_idx - 1

        if step_idx >= self.force_step_limit:
            still_running = ~self.is_stopped
            if still_running.any():
                self.is_stopped[still_running] = True
                self.stop_steps[still_running] = step_idx

        self._shift_history(current_vals)
        
        return self.is_stopped

    def _shift_history(self, current_vals):
        self.prev_vals = current_vals.clone()
