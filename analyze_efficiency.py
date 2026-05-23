import json, gc, os, sys
from math import floor

BASE = os.path.dirname(os.path.abspath(__file__))
SIMPLE_DIR = os.path.join(BASE, "results_simple_baseline")
FULL_DIR = os.path.join(BASE, "results_full_baseline")
DATASETS = ['ActivityNet-QA', 'EMCompress', 'NExT-OE', 'EgoSchema', 'LVBench', 'MLVU', 'Video-MME']
CHARS_PER_TOKEN = 3.5


def estimate_output_tokens(resp_text):
    if not resp_text:
        return 0
    return len(str(resp_text)) / CHARS_PER_TOKEN


def analyze_simple():
    results = {}
    for ds in DATASETS:
        log_path = os.path.join(SIMPLE_DIR, f"simple_{ds}_log.json")
        res_path = os.path.join(SIMPLE_DIR, f"simple_{ds}.json")
        if not os.path.exists(log_path):
            continue
        with open(log_path) as f:
            logs = json.load(f)
        with open(res_path) as f:
            res_data = json.load(f)

        samples = []
        for key in logs:
            e = logs[key]
            if not isinstance(e, dict):
                continue

            result = e.get('result', {})
            if not isinstance(result, dict):
                continue
            status = result.get('status', 'unknown')
            if status == 'failed':
                continue

            screened_ts = result.get('screened_timestamps', [])
            vid_dur = e.get('vid_duration', 0)

            screened_dur = 0
            if isinstance(screened_ts, list):
                for seg in screened_ts:
                    if isinstance(seg, list) and len(seg) >= 2:
                        screened_dur += max(0, seg[1] - seg[0])

            is_screened = True
            if vid_dur > 0 and screened_dur >= vid_dur * 0.99:
                is_screened = False

            init_caps = len(e.get('initial_captions', []))
            extra_caps = len(e.get('tool_calls', []))

            n_llm = 0
            total_ct = 0
            total_pt = 0
            total_resp_chars = 0
            for c in e.get('llm_calls', []):
                if isinstance(c, dict) and 'usage' in c:
                    n_llm += 1
                    total_ct += c['usage'].get('completion_tokens', 0)
                    total_pt += c['usage'].get('prompt_tokens', 0)
                    resp = c.get('response', '')
                    total_resp_chars += len(str(resp)) if resp else 0

            real_out_tok = total_resp_chars / CHARS_PER_TOKEN

            samples.append({
                'status': status,
                'is_screened': is_screened,
                'vid_dur': vid_dur,
                'screened_dur': screened_dur,
                'init_caps': init_caps,
                'extra_caps': extra_caps,
                'n_llm': n_llm,
                'tool_calls': extra_caps,
                'real_out_tok': real_out_tok,
                'prompt_tok': total_pt,
            })

        results[ds] = samples
        del logs, res_data
        gc.collect()

    return results


def analyze_full():
    results = {}
    for ds in DATASETS:
        log_path = os.path.join(FULL_DIR, f"full_{ds}_log.json")
        res_path = os.path.join(FULL_DIR, f"full_{ds}.json")
        if not os.path.exists(log_path):
            continue
        with open(log_path) as f:
            logs = json.load(f)
        with open(res_path) as f:
            res_data = json.load(f)

        samples = []
        for key in logs:
            e = logs[key]
            if e is None or not isinstance(e, dict):
                continue
            traj = e.get('trajectory')
            if traj is None or not isinstance(traj, dict):
                continue

            result = e.get('result', {})
            if not isinstance(result, dict):
                continue
            status = result.get('status', 'unknown')
            if status == 'failed':
                continue

            vid_dur = e.get('vid_duration', 0)

            screened_ts = result.get('screened_timestamps', [])
            screened_dur = 0
            if isinstance(screened_ts, list):
                for seg in screened_ts:
                    if isinstance(seg, list) and len(seg) >= 2:
                        screened_dur += max(0, seg[1] - seg[0])

            is_screened = (status == 'succeeded')

            all_caption_calls = traj.get('caption_calls', [])
            total_captions = len(all_caption_calls)

            validator_init_caps_total = 0
            n_launcher_llm = 0
            n_validator_llm = 0
            n_viewer_llm = 0
            n_validator_viewer_spawns = 0
            n_viewer_tool_calls = 0
            n_scan = 0
            n_localize = 0
            n_get_image_cap = 0
            launcher_out_chars = 0
            launcher_pt = 0
            validator_out_chars = 0
            validator_pt = 0
            viewer_out_chars = 0
            viewer_pt = 0
            localize_out_chars = 0
            localize_pt = 0
            scan_synth_out_chars = 0
            scan_synth_pt = 0
            localize_init_caps_total = 0

            for rnd in traj.get('rounds', []):
                if rnd is None or not isinstance(rnd, dict):
                    continue
                for trial in rnd.get('trials', []):
                    if trial is None or not isinstance(trial, dict):
                        continue

                    launcher = trial.get('launcher')
                    if isinstance(launcher, dict):
                        for c in launcher.get('llm_calls', []):
                            if isinstance(c, dict) and 'usage' in c:
                                n_launcher_llm += 1
                                launcher_pt += c['usage'].get('prompt_tokens', 0)
                                resp = c.get('response', '')
                                launcher_out_chars += len(str(resp)) if resp else 0

                    validator = trial.get('validator')
                    if isinstance(validator, dict):
                        vic = validator.get('initial_captions', {})
                        if isinstance(vic, dict):
                            validator_init_caps_total += len(vic)
                        elif isinstance(vic, list):
                            validator_init_caps_total += len(vic)

                        for c in validator.get('llm_calls', []):
                            if isinstance(c, dict) and 'usage' in c:
                                n_validator_llm += 1
                                validator_pt += c['usage'].get('prompt_tokens', 0)
                                resp = c.get('response', '')
                                validator_out_chars += len(str(resp)) if resp else 0

                    viewer_sessions = trial.get('viewer_sessions', [])
                    n_validator_viewer_spawns += len(viewer_sessions) if viewer_sessions else 0

                    for vs in (viewer_sessions or []):
                        if vs is None or not isinstance(vs, dict):
                            continue
                        for c in vs.get('llm_calls', []):
                            if isinstance(c, dict) and 'usage' in c:
                                n_viewer_llm += 1
                                viewer_pt += c['usage'].get('prompt_tokens', 0)
                                resp = c.get('response', '')
                                viewer_out_chars += len(str(resp)) if resp else 0

                        for tc in vs.get('tool_calls', []):
                            if not isinstance(tc, dict):
                                continue
                            n_viewer_tool_calls += 1
                            parsed = tc.get('parsed', {})
                            func = parsed.get('function', '')
                            if func == 'scan':
                                n_scan += 1
                            elif func == 'localize':
                                n_localize += 1
                                localize_init_caps_total += 10
                                for lc in tc.get('localize_llm_calls', []):
                                    if isinstance(lc, dict) and 'usage' in lc:
                                        localize_pt += lc['usage'].get('prompt_tokens', 0)
                                        resp = lc.get('response', '')
                                        localize_out_chars += len(str(resp)) if resp else 0
                            elif func == 'get_image_cap':
                                n_get_image_cap += 1

            n_localize_llm = 0
            for rnd in traj.get('rounds', []):
                if rnd is None or not isinstance(rnd, dict): continue
                for trial in rnd.get('trials', []):
                    if trial is None or not isinstance(trial, dict): continue
                    for vs in (trial.get('viewer_sessions') or []):
                        if vs is None or not isinstance(vs, dict): continue
                        for tc in vs.get('tool_calls', []):
                            if isinstance(tc, dict) and tc.get('parsed', {}).get('function') == 'localize':
                                n_localize_llm += len(tc.get('localize_llm_calls', []))

            n_total_llm = n_launcher_llm + n_validator_llm + n_viewer_llm + n_localize_llm
            n_total_llm += n_scan

            total_out_chars = launcher_out_chars + validator_out_chars + viewer_out_chars + localize_out_chars
            total_pt = launcher_pt + validator_pt + viewer_pt + localize_pt
            total_real_out = total_out_chars / CHARS_PER_TOKEN

            passive_caps = validator_init_caps_total + localize_init_caps_total
            active_caps = max(0, total_captions - passive_caps)

            samples.append({
                'status': status,
                'is_screened': is_screened,
                'vid_dur': vid_dur,
                'screened_dur': screened_dur,
                'total_captions': total_captions,
                'passive_caps': passive_caps,
                'validator_init_caps': validator_init_caps_total,
                'localize_init_caps': localize_init_caps_total,
                'active_caps': active_caps,
                'n_launcher_llm': n_launcher_llm,
                'n_validator_llm': n_validator_llm,
                'n_viewer_llm': n_viewer_llm,
                'n_localize_llm': n_localize_llm,
                'n_scan_synth': n_scan,
                'n_total_llm': n_total_llm,
                'n_validator_viewer_spawns': n_validator_viewer_spawns,
                'n_viewer_tool_calls': n_viewer_tool_calls,
                'n_scan': n_scan,
                'n_localize': n_localize,
                'n_get_image_cap': n_get_image_cap,
                'real_out_tok': total_real_out,
                'prompt_tok': total_pt,
            })

        results[ds] = samples
        del logs, res_data
        gc.collect()

    return results


def avg(lst):
    return sum(lst) / len(lst) if lst else 0


def print_simple(simple_results):
    print()
    print("=" * 140)
    print("TABLE 1: SIMPLE BASELINE (Single-agent, gpt-4o)")
    print("=" * 140)
    print()
    print("Parameter definitions:")
    print("  N                 = number of valid samples (excluding API/video failures)")
    print("  Init Cap/s        = initial frame captions per sample (uniformly sampled at pipeline start, passively provided in prompt)")
    print("  Extra Cap/s       = additional caption requests per sample (actively requested by agent via tool calls)")
    print("  Total Cap/s       = Init + Extra captions per sample")
    print("  Tool Calls/s      = number of tool calls per sample (= Extra Cap/s, since the only tool is get_image_cap)")
    print("  LLM Calls/s       = number of LLM (gpt-4o) calls per sample (multi-turn conversation)")
    print("  Output Tok/s      = estimated real output tokens per sample (excluding hidden reasoning tokens)")
    print("  Dur Ratio (all)   = avg(screened_duration / original_duration) across ALL valid samples")
    print("  Dur Ratio (scrn)  = avg(screened_duration / original_duration) only for samples where screening was applied (ratio < 0.99)")
    print("  Screen%           = percentage of samples where screening was applied")
    print()

    header = f'{"Dataset":<18} {"N":>5} {"Init":>5} {"Extra":>6} {"Total":>6} {"Tool":>5} {"LLM":>5} {"OutTok":>8} {"DurAll":>7} {"DurScrn":>8} {"Scrn%":>6}'
    print(header)
    print("-" * len(header))

    for ds in DATASETS:
        if ds not in simple_results:
            continue
        samples = simple_results[ds]
        n = len(samples)
        if n == 0:
            continue

        init_caps = avg([s['init_caps'] for s in samples])
        extra_caps = avg([s['extra_caps'] for s in samples])
        total_caps = init_caps + extra_caps
        tool_calls = avg([s['tool_calls'] for s in samples])
        llm_calls = avg([s['n_llm'] for s in samples])
        out_tok = avg([s['real_out_tok'] for s in samples])

        dur_ratios_all = []
        dur_ratios_screened = []
        for s in samples:
            if s['vid_dur'] > 0:
                ratio = s['screened_dur'] / s['vid_dur']
                dur_ratios_all.append(ratio)
                if s['is_screened']:
                    dur_ratios_screened.append(ratio)

        dur_all = avg(dur_ratios_all)
        dur_scrn = avg(dur_ratios_screened) if dur_ratios_screened else 0
        scrn_pct = len(dur_ratios_screened) / n * 100 if n > 0 else 0

        ds_label = ds
        print(f'{ds_label:<18} {n:>5} {init_caps:>5.1f} {extra_caps:>6.1f} {total_caps:>6.1f} {tool_calls:>5.1f} {llm_calls:>5.1f} {out_tok:>8.0f} {dur_all:>7.1%} {dur_scrn:>8.1%} {scrn_pct:>5.1f}%')


def print_full(full_results):
    print()
    print("=" * 180)
    print("TABLE 2: FULL BASELINE (Multi-agent: Launcher→Validator→Viewer, gpt-4o)")
    print("=" * 180)
    print()
    print("Parameter definitions:")
    print("  N                   = number of valid samples (excluding API/video failures)")
    print()
    print("  --- Caption Breakdown ---")
    print("  Total Cap/s         = total caption calls per sample (all sources)")
    print("  Passive Cap/s       = captions passively provided in prompts = Val Init + Loc Init")
    print("    Val Init/s        = validator initial captions (uniformly sampled, embedded in validator prompt)")
    print("    Loc Init/s        = localize initial captions (k=10 uniform frames per localize() call, embedded in localize prompt)")
    print("  Active Cap/s        = captions actively fetched during tool execution = Total - Passive")
    print("                        (includes: scan frame sampling, localize extra tool calls, get_image_cap)")
    print()
    print("  --- Tool Call Breakdown ---")
    print("  Val→Viewer/s        = viewer sessions spawned per sample (= number of times validator invokes viewer as its tool)")
    print("  Viewer Tools/s      = tool calls made by viewer per sample, broken down as:")
    print("    scan/s            = scan(start,end) calls: uniformly samples up to 4 frames → LLM synthesizes summary")
    print("    localize/s        = localize(query) calls: 3-stage LLM sub-agent (Propose→Select→Spread)")
    print("    get_cap/s         = get_image_cap(ts) calls: fetches single frame caption")
    print()
    print("  --- LLM & Token Cost ---")
    print("  LLM Calls/s         = total LLM calls per sample (launcher + validator + viewer + localize sub-agent + scan synthesis)")
    print("  Output Tok/s        = real output tokens per sample (reasoning tokens excluded)")
    print()
    print("  --- Screening Effectiveness ---")
    print("  Dur Ratio (all)     = avg(screened_duration / original_duration) across ALL valid samples (incl. no_screening)")
    print("  Dur Ratio (scrn)    = same ratio only for successfully screened samples")
    print("  Screen%             = percentage of samples successfully screened")
    print()

    header = f'{"Dataset":<18} {"N":>5} | {"TotCap":>7} {"Pasv":>5} {"VaIn":>5} {"LoIn":>5} {"Actv":>5} | {"V→Vw":>5} {"VwTl":>5} {"scan":>5} {"loc":>4} {"gcap":>5} | {"LLM":>5} {"OutTok":>8} | {"DurAll":>7} {"DurScr":>7} {"Scr%":>5}'
    print(header)
    print("-" * len(header))

    for ds in DATASETS:
        if ds not in full_results:
            continue
        samples = full_results[ds]
        n = len(samples)
        if n == 0:
            continue

        total_caps = avg([s['total_captions'] for s in samples])
        passive_caps = avg([s['passive_caps'] for s in samples])
        val_init = avg([s['validator_init_caps'] for s in samples])
        loc_init = avg([s['localize_init_caps'] for s in samples])
        active_caps = avg([s['active_caps'] for s in samples])
        val_viewer = avg([s['n_validator_viewer_spawns'] for s in samples])
        viewer_tools = avg([s['n_viewer_tool_calls'] for s in samples])
        n_scan = avg([s['n_scan'] for s in samples])
        n_loc = avg([s['n_localize'] for s in samples])
        n_gcap = avg([s['n_get_image_cap'] for s in samples])
        llm_calls = avg([s['n_total_llm'] for s in samples])
        out_tok = avg([s['real_out_tok'] for s in samples])

        dur_ratios_all = []
        dur_ratios_screened = []
        for s in samples:
            if s['vid_dur'] > 0:
                ratio = s['screened_dur'] / s['vid_dur']
                dur_ratios_all.append(ratio)
                if s['is_screened']:
                    dur_ratios_screened.append(ratio)

        dur_all = avg(dur_ratios_all)
        dur_scrn = avg(dur_ratios_screened) if dur_ratios_screened else 0
        scrn_pct = len(dur_ratios_screened) / n * 100 if n > 0 else 0

        ds_label = ds
        print(f'{ds_label:<18} {n:>5} | {total_caps:>7.1f} {passive_caps:>5.1f} {val_init:>5.1f} {loc_init:>5.1f} {active_caps:>5.1f} | {val_viewer:>5.1f} {viewer_tools:>5.1f} {n_scan:>5.1f} {n_loc:>4.1f} {n_gcap:>5.1f} | {llm_calls:>5.1f} {out_tok:>8.0f} | {dur_all:>7.1%} {dur_scrn:>7.1%} {scrn_pct:>4.1f}%')


def print_comparison(simple_results, full_results):
    print()
    print("=" * 160)
    print("TABLE 3: FULL / SIMPLE RATIO (per-sample averages)")
    print("=" * 160)
    print()
    print("All values are Full_avg / Simple_avg. Values >1x mean full costs more; <1x means full costs less.")
    print()
    print("Parameter definitions:")
    print("  Passive Cap  = passively provided initial captions ratio")
    print("                 Simple: init_caps (20 uniform frames at start)")
    print("                 Full:   validator_init (10/trial) + localize_init (10/call)")
    print("  Active Cap   = actively requested extra captions ratio")
    print("                 Simple: extra tool calls (get_image_cap only)")
    print("                 Full:   scan frames + localize extra + get_image_cap")
    print("  Total Cap    = total captions ratio (passive + active)")
    print("  LLM Calls    = total LLM calls ratio")
    print("  Output Tok   = real output tokens ratio (reasoning tokens excluded)")
    print()

    header = f'{"Dataset":<18} {"PassCap":>8} {"ActCap":>8} {"TotCap":>8} {"LLM":>8} {"OutTok":>8}'
    print(header)
    print("-" * len(header))

    for ds in DATASETS:
        if ds not in simple_results or ds not in full_results:
            continue
        ss = simple_results[ds]
        fs = full_results[ds]
        if not ss or not fs:
            continue

        s_init = avg([s['init_caps'] for s in ss])
        s_extra = avg([s['extra_caps'] for s in ss])
        s_total_cap = s_init + s_extra
        s_llm = avg([s['n_llm'] for s in ss])
        s_out = avg([s['real_out_tok'] for s in ss])

        f_passive = avg([s['passive_caps'] for s in fs])
        f_active = avg([s['active_caps'] for s in fs])
        f_total_cap = avg([s['total_captions'] for s in fs])
        f_llm = avg([s['n_total_llm'] for s in fs])
        f_out = avg([s['real_out_tok'] for s in fs])

        r_pass = f_passive / s_init if s_init > 0 else float('inf')
        r_act = f_active / s_extra if s_extra > 0 else float('inf')
        r_cap = f_total_cap / s_total_cap if s_total_cap > 0 else float('inf')
        r_llm = f_llm / s_llm if s_llm > 0 else float('inf')
        r_out = f_out / s_out if s_out > 0 else float('inf')

        ds_label = ds
        print(f'{ds_label:<18} {r_pass:>7.1f}x {r_act:>7.1f}x {r_cap:>7.1f}x {r_llm:>7.1f}x {r_out:>7.1f}x')


def _compute_baseline_stats(datasets, results, baseline_type, downstream_k):
    rows = []
    for ds in datasets:
        if ds not in results:
            continue
        samples = results[ds]

        screened = [s for s in samples if s['is_screened'] and s['vid_dur'] > 0]
        n_scrn = len(screened)
        if n_scrn == 0:
            continue
        density_amps = [1.0 / (s['screened_dur'] / s['vid_dur']) for s in screened if s['screened_dur'] > 0]
        dens_amp = avg(density_amps) if density_amps else 0

        if baseline_type == 'simple':
            total_caps = avg([s['init_caps'] + s['extra_caps'] for s in screened])
        else:
            total_caps = avg([s['total_captions'] for s in screened])

        avg_out_tok = avg([s['real_out_tok'] for s in screened])
        dur_ratios = [s['screened_dur'] / s['vid_dur'] for s in screened if s['vid_dur'] > 0]
        avg_dur_ratio = avg(dur_ratios) if dur_ratios else 0

        reduction_pct = (1.0 - avg_dur_ratio) * 100
        avg_tok_per_pct = avg_out_tok / reduction_pct if reduction_pct > 0 else 0

        equiv_frames = downstream_k * dens_amp
        total_visual = downstream_k + total_caps
        cost_ratio = total_visual / equiv_frames if equiv_frames > 0 else float('inf')

        rows.append({
            'ds': ds,
            'n_scrn': n_scrn,
            'dens_amp': dens_amp,
            'equiv_frames': equiv_frames,
            'total_caps': total_caps,
            'total_visual': total_visual,
            'cost_ratio': cost_ratio,
            'avg_tok_per_pct': avg_tok_per_pct,
        })
    return rows


def print_cost_effectiveness(simple_results, full_results):
    K_VALUES = [8, 16, 32, 100]

    print()
    print("=" * 160)
    print("TABLE 4: COST-EFFECTIVENESS ANALYSIS (only screened samples)")
    print("=" * 160)
    print()
    print("Why K=8 matters:")
    print("  Most Video-LLMs (LLaVA-OneVision, VideoChat2, Video-LLaMA, etc.) uniformly sample 8 frames for inference.")
    print("  K=8 is therefore the de facto standard frame budget in current VideoQA benchmarks.")
    print("  We additionally report K ∈ {16, 32, 100} to show how cost-effectiveness scales with denser downstream sampling.")
    print()
    print("All metrics below are computed ONLY on successfully screened samples (dur_ratio < 0.99).")
    print()
    print("Parameter definitions:")
    print("  K                   = downstream frame budget (number of frames uniformly sampled by the VideoQA model)")
    print()
    print("  --- Metric 1: Density Amplification (K-independent) ---")
    print("  DensAmp             = avg(1 / dur_ratio) across screened samples")
    print("                        Interpretation: after screening, K frames cover the relevant segment at DensAmp× the")
    print("                        density of sampling the full video with K frames. Equivalently, to achieve the same")
    print("                        per-second frame density on the FULL video, one would need K × DensAmp frames.")
    print()
    print("  --- Metric 2: EMC Visual Cost Ratio (K-dependent) ---")
    print("  EquivFrames(K)      = K × DensAmp = frames needed on the full video to match post-screening density")
    print("  TotalVisual(K)      = K + TotalCap/s = downstream frames + EMC caption overhead")
    print("  CostRatio(K)        = TotalVisual(K) / EquivFrames(K)")
    print("                        Interpretation: fraction of visual cost EMC+downstream actually spends vs. dense sampling.")
    print("                        Lower = more cost-effective. E.g., 0.10 means EMC achieves the same density at 10% cost.")
    print("                        As K increases, the fixed EMC caption cost is amortized over more downstream frames,")
    print("                        so CostRatio converges toward 1/DensAmp (the theoretical minimum).")
    print()
    print("  --- Metric 3: Output Tokens per 1% Video Reduction (K-independent) ---")
    print("  OutTok/1%Red        = avg(OutTok) / ((1 - avg(DurRatio_scrn)) × 100)")
    print("                        Interpretation: LLM output token cost to achieve each 1% of video length reduction.")
    print("                        Lower = cheaper screening per unit of reduction.")
    print("                        Directly verifiable from TABLE 1/2: OutTok / ((1 - DurScrn) × 100).")
    print()

    for baseline_type, results, label in [('simple', simple_results, 'SIMPLE BASELINE'),
                                           ('full', full_results, 'FULL BASELINE')]:
        print("-" * 130)
        print(f"  {label}")
        print("-" * 130)

        rows = _compute_baseline_stats(DATASETS, results, baseline_type, 8)

        header_a = f'{"Dataset":<18} {"N_scrn":>6} {"DensAmp":>8} {"TotCap":>7} {"OutTok/1%":>10}'
        print(header_a)
        print("-" * len(header_a))
        for row in rows:
            ds_label = row['ds'] + '*' if row['ds'] == 'NExT-OE' else row['ds']
            print(f'{ds_label:<18} {row["n_scrn"]:>6} {row["dens_amp"]:>7.1f}x {row["total_caps"]:>7.1f} {row["avg_tok_per_pct"]:>10.1f}')
        print()

        for k in K_VALUES:
            print(f'  K={k}:')
            header_k = f'  {"Dataset":<18} {"EquivFr":>8} {"TotVis":>7} {"CostR":>7}'
            print(header_k)
            print(f'  {"-" * (len(header_k)-2)}')
            for row in rows:
                ds_label = row['ds'] + '*' if row['ds'] == 'NExT-OE' else row['ds']
                ef = k * row['dens_amp']
                tv = k + row['total_caps']
                cr = tv / ef if ef > 0 else float('inf')
                print(f'  {ds_label:<18} {ef:>8.0f} {tv:>7.1f} {cr:>7.2f}')
            print()

        print()


if __name__ == "__main__":
    print("Analyzing simple baseline...")
    simple_results = analyze_simple()
    print("Analyzing full baseline...")
    full_results = analyze_full()

    print_simple(simple_results)
    print_full(full_results)
    print_comparison(simple_results, full_results)
    print_cost_effectiveness(simple_results, full_results)
