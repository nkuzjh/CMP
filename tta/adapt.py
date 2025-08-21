import math

import torch
from torch.cuda.amp import autocast
from torch.nn import functional as F

import utils


from train import mlm



@torch.enable_grad()
def test_time_adapt_itm(model, optimizer, scaler, epoch, device, scheduler, config, dataloader):#sims_matrix, image_embeds, text_embeds, text_atts):
    # model.eval()
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('entropy', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # metric_logger.add_meter('uncertainty', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = '      TTA Epoch: [{}]'.format(epoch)
    print_freq = 100

    for iter, (encoder_output, encoder_att, text_embeds, text_atts, uncertainty, proba_top1_sim_list, proba_inversed_sim_list) in enumerate(metric_logger.log_every(dataloader, print_freq, header)):

        encoder_output = encoder_output.reshape(-1, encoder_output.size(-2), encoder_output.size(-1)).to(device)
        encoder_att = encoder_att.reshape(-1, encoder_att.size(-1)).to(device)
        text_embeds = text_embeds.reshape(-1, text_embeds.size(-2), text_embeds.size(-1)).to(device)
        text_atts = text_atts.reshape(-1, text_atts.size(-1)).to(device)
        uncertainty = uncertainty.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model.get_cross_embeds(
                encoder_output,#([24, 50, 1024])
                encoder_att,#([24, 50])
                text_embeds,#([24, 56, 768])
                text_atts#([24, 56])
            )[:, 0, :] # (bs*k_tta, sequence, last_hidden_states)[:, 0, :] -> (bs*tta, last_hidden_states)
            logits = model.itm_head(output) # (bs*tta, 2)
            logits = logits.reshape(-1, config['k_tta'], 2) # (bs, tta, 2)
            score = logits[..., 1] # (bs, tta)
            entropy = -(F.softmax(score * config['score_temper'], dim=-1) * F.log_softmax(score * config['score_temper'], dim=-1)).sum(-1)
            if config.get('uncertainty', None) is not None:
                loss = entropy / uncertainty + uncertainty
            else:
                loss = entropy
            loss = loss.mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale > scaler.get_scale())
        if not skip_lr_sched:
            scheduler.step()
        optimizer.zero_grad()

        metric_logger.update(entropy=entropy.mean().item())
        # metric_logger.update(uncertainty=uncertainty.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("     Averaged stats:", metric_logger.global_avg())
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}