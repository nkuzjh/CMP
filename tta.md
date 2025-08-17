# official baseline

## command

1. train
    python3 run.py --task "cmp" --dist "f4" --output_dir "output/cmp"
2. evaluate
    python3 run.py --task "cmp" --evaluate --dist "f4" --output_dir "output/cmp_eval" --checkpoint "checkpoint/cmp.pth"
3. tta
    python3 run.py --task "tta" --tta --output_dir "output/tta" --checkpoint "checkpoint/cmp.pth"

## exp

1. tta_debug
    python3 run.py --task "tta_debug" --tta --output_dir "output/tta" --checkpoint "checkpoint/cmp.pth"

1. tta_exp0
    python3 run.py --task "tta_exp0" --tta --output_dir "output/tta" --checkpoint "checkpoint/cmp.pth"