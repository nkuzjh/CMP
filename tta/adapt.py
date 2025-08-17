import math

import torch
from torch.cuda.amp import autocast

import utils


from train import mlm



def test_time_adapt_itm(model, optimizer, scaler, epoch, device, scheduler, config, sims_matrix, image_embeds, text_embeds, text_atts):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('entropy', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # metric_logger.add_meter('uncertainty', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'TTA Epoch: [{}]'.format(epoch)
    print_freq = 10

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, print_freq, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_tta'], dim=0)
        encoder_output = image_embeds[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model.get_cross_embeds(
                encoder_output,
                encoder_att,
                text_embeds=text_embeds[i].repeat(config['k_test'], 1, 1),
                text_atts=text_atts[i].repeat(config['k_test'], 1),
            )[:, 0, :] # bs, sequence, last_hidden_states
            score = model.itm_head(output)[:, 1]
            entropy = -(F.softmax(score * score_temper, dim=-1) * F.log_softmax(score * score_temper, dim=-1)).sum(-1)
            loss = entropy #/ uncertainty + uncertainty

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale > scaler.get_scale())
        if not skip_lr_sched:
            scheduler.step()
        optimizer.zero_grad()

        metric_logger.update(entropy=entropy.item())
        # metric_logger.update(uncertainty=uncertainty.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("     Averaged stats:", metric_logger.global_avg())
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}