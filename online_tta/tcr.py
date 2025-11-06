import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F

from online_tta.param import load_model_and_optimizer, copy_model_and_optimizer


from online_tta.ddp import *
from online_tta.losses import *


class TCR(nn.Module):
    """TCR adapts a model by entropy minimization during testing.

    Once TCRed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "TCR requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(
            self,
            text_input,
            image_feats_norm,
            device, args, metric_logger, queue_list, max_queue_size, update_signal
        ):
        if self.episodic:
            self.reset()

        for step in range(self.steps):
            queue_list, outputs, text_embed, text_feat = forward_and_adapt(
                text_input,
                image_feats_norm,
                device, args, metric_logger, queue_list, max_queue_size, update_signal, step, self.model, self.optimizer)

        return queue_list, outputs, text_embed, text_feat

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(
        text_input,
        image_feats_norm,
        device, args, metric_logger, queue_list, max_queue_size, update_signal, step, model, optimizer
    ):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # loss_REM, loss_UNI, loss_EMG, queue_list, outputs=model.module.forward_tta(
    #     modality_query,
    #     device,
    #     queue_list,
    #     max_queue_size,
    #     update_signal,
    #     step,
    #     args)

    # output = model.get_cross_embeds(
    #     encoder_output,
    #     encoder_att,
    #     text_embeds,
    #     text_atts,
    # )[:, 0, :]
    # print(output.shape)#torch.Size([128, 768])
    # outputs = model.itm_head(output)[:, 1]
    # print(outputs.shape)#torch.Size([128])

    text_embed = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
    text_feat = model.get_text_feat(text_embed)
    text_feats_norm = F.normalize(text_feat, dim=-1)
    sims = image_feats_norm @ text_feats_norm.t()
    outputs = sims.t()

    modality_gallery_feat_all = image_feats_norm
    modality_query_feat = text_feats_norm

    nearest_neighbors_indices = (outputs).argmax(dim=1)
    modality_gallery_feat = modality_gallery_feat_all[nearest_neighbors_indices]

    if (step==0 and update_signal):
        queue_list=update_queue(modality_query_feat, modality_gallery_feat, queue_list, args.con_ratio, max_queue_size, args)

    margin, entropy_queue=get_current_value(queue_list)
    sim_inter = (modality_query_feat @ modality_gallery_feat.t()) /args.temperature

    loss_REM=entropy_loss_against_noisy(sim_inter, entropy_queue)
    loss_UNI=center_uniform_loss(modality_query_feat, t=args.t)

    target_modality_gap=compute_modality_gap(modality_query_feat, modality_gallery_feat)
    loss_EMG=(target_modality_gap-margin)**2

    loss=loss_REM+loss_UNI+loss_EMG


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    metric_logger.update(loss_REM=loss_REM.item())
    metric_logger.update(loss_UNI=loss_UNI.item())
    metric_logger.update(loss_EMG=loss_EMG.item())
    metric_logger.update(loss_total=loss.item())
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return queue_list, outputs, text_embed, text_feat
