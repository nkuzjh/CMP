import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F

from tta.online_tta.param import load_model_and_optimizer, copy_model_and_optimizer


from tta.online_tta.ddp import *
from tta.online_tta.losses import *


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
            encoder_output,
            encoder_att,
            text_embeds,
            text_atts,
            config, device, args, metric_logger, queue_list, max_queue_size, update_signal
        ):
        if self.episodic:
            self.reset()

        for step in range(self.steps):
            queue_list, outputs = forward_and_adapt(
                encoder_output,
                encoder_att,
                text_embeds,
                text_atts,
                config, device, args, metric_logger, queue_list, max_queue_size, update_signal, step, self.model, self.optimizer)

        return queue_list, outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(
        encoder_output,
        encoder_att,
        text_embeds,
        text_atts,
        config, device, args, metric_logger, queue_list, max_queue_size, update_signal, step, model, optimizer
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
    output = model.get_cross_embeds(
        encoder_output,
        encoder_att,
        text_embeds,
        text_atts,
    )[:, 0, :]
    logits = model.itm_head(output) # (bs*tta, 2)
    logits = logits.reshape(-1, config['k_tta'], 2) # (bs, tta, 2)
    outputs = logits[..., 1] # (bs, tta)


    text_feats = model.get_text_feat(text_embeds)
    text_feats = F.normalize(text_feats, dim=-1) #torch.Size([128, 256])
    text_feats = text_feats.reshape(-1, config['k_tta'], text_feats.size(-1))
    image_feats = model.get_image_feat(encoder_output) #torch.Size([16, 8, 256])
    image_feats = F.normalize(image_feats, dim=-1) #torch.Size([128, 256])
    image_feats = image_feats.reshape(-1, config['k_tta'], image_feats.size(-1)) #torch.Size([16, 8, 256])


    modality_query_feat = text_feats[:, 0, :] # torch.Size([16, 256]) # 为了统一itm输入向量的维度,dataset中将一每个query feature复制了k_tta次
    modality_gallery_feat_all = image_feats # torch.Size([16, 8, 256])

    nearest_neighbors_indices = (outputs).argmax(dim=1) # 16
    modality_gallery_feat = modality_gallery_feat_all[torch.arange(modality_gallery_feat_all.shape[0], device=modality_gallery_feat_all.device), nearest_neighbors_indices, :] #16,256
    # modality_gallery_feat_list = []
    # for indice, gallery_feat in zip(nearest_neighbors_indices, modality_gallery_feat_all):
    #     modality_gallery_feat_list.append(gallery_feat[indice])
    # modality_gallery_feat = torch.stack(modality_gallery_feat_list)


    if (step==0 and update_signal):
        queue_list=update_queue(outputs, modality_query_feat, modality_gallery_feat, queue_list, args.con_ratio, max_queue_size, args)

    margin, entropy_queue=get_current_value(queue_list)
    sim_inter = (modality_query_feat @ modality_gallery_feat.t()) /args.temperature

    loss_REM=entropy_loss_against_noisy(outputs, entropy_queue)
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
    return queue_list, outputs
