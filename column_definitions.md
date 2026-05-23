# Table 2 Column Definitions

- **#TotalCap**: total caption calls per sample (all sources)
- **Pasv**: passive captions per sample = VaIn + LoIn, embedded in prompts without explicit tool calls
- **VaIn**: validator initial captions, uniformly sampled frames embedded in the Validator's prompt (~10 per trial)
- **LoIn**: localize initial captions, each `localize()` call samples k=10 uniform frames for its sub-agent prompt
- **Actv**: active captions per sample = TotalCap − Pasv, fetched during tool execution (scan sampling, localize queries, get_image_cap)
- **V→Vw**: viewer sessions spawned per sample = number of times the Validator invokes the Viewer as its tool
- **VwTl**: total tool calls made by the Viewer per sample = scan + localize + get_cap
- **#scan**: `scan(start, end)` calls, samples up to 4 frames from a time range and synthesizes a temporal summary
- **#localize**: `localize(query)` calls, a 3-stage sub-agent (Propose→Select→Spread) for fine-grained timestamp search
- **#get_cap**: `get_image_cap(ts)` calls, fetches a single frame caption at the specified timestamp
- **DurAll**: avg(screened_duration / original_duration) across all valid samples, including no_screening outcomes
- **DurScrn**: same ratio, only for successfully screened samples
- **Scrn%**: percentage of samples successfully screened
- **OutTok/1%**: output token cost per 1% of video reduction = avg(OutTok) / ((1 − avg(DurScrn)) × 100). For a 1-hour video at 30fps, 1% = 36 seconds = 1,080 frames. The simple baseline achieves each 1% reduction at a cost of only 3.8–15.1 output tokens.
