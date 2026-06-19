import os, json
runs = 'runs'
for d in sorted(os.listdir(runs)):
    full = os.path.join(runs, d)
    if not os.path.isdir(full): continue
    subs = sorted(os.listdir(full))
    if not subs: continue
    printed_header = False
    for s in subs:
        cfg_path = os.path.join(full, s, 'config.json')
        if not os.path.exists(cfg_path): continue
        with open(cfg_path, encoding='utf-8') as f:
            c = json.load(f)
        has_model = os.path.exists(os.path.join(full, s, 'best_model.pt'))
        if not has_model: continue
        if not printed_header:
            print(f'\n[{d}]')
            printed_header = True
        K  = c.get('K','?')
        h  = c.get('hidden_sizes', [c.get('n_hidden','?')])
        rt = c.get('readout_type','linear')
        tm = c.get('train_mode','?')
        print(f'  {s:<58s}  K={K}  h={h}  rt={rt}  mode={tm}')
