# Erben Blind Analysis Pipeline

## What This Project Is

An experimental pipeline to test whether the Erben Johansson phonosemantic framework has genuine predictive power or whether LLM readings of it are contaminated by latent semantic knowledge.

**The problem:** When an LLM does a phonosemantic "Erben reading" of a word, it already knows what the word means (from training data). So a reading that matches the word's meaning could be genuine phonosemantic signal OR the model unconsciously retrofitting from known semantics. We can't distinguish these.

**The solution:** A fully blinded pipeline where:
1. **Deterministic code** decomposes a word into phonemes and mechanically applies the Erben lookup table to produce an abstract activation profile
2. **A blinded LLM** receives ONLY the abstract activation profile — never the word, language, or phonemes — and predicts what semantic field it points to
3. **Human evaluation** compares the prediction against the actual meaning

The LLM literally cannot contaminate the result because it never sees the word.

## Architecture

```
erben-blind/
├── config/                              # ALREADY BUILT — do not modify without understanding
│   ├── erben_table.json                 # 20 macro-concepts ↔ sound groups (from the paper)
│   ├── phoneme_features.json            # IPA phonemes → sound group memberships
│   ├── hungarian_orthography.json       # Hungarian spelling → IPA mapping
│   ├── concept_codes.json               # MC-01..MC-20 ↔ human-readable labels
│   ├── semantic_domains.json            # Pool of semantic fields for forced-choice options
│   └── prompt_templates.json            # Exact prompts for blinded LLM steps
│
├── pipeline/
│   ├── phoneme_decomposer.py            # TO BUILD: Hungarian text → phoneme sequence
│   ├── activation_profiler.py           # TO BUILD: phonemes → layered Erben activation profile
│   ├── prompt_formatter.py              # TO BUILD: activation profile → sequential prompts
│   ├── option_selector.py              # TO BUILD: picks forced-choice options per trial
│   ├── random_profile_generator.py      # TO BUILD: creates random activation profiles for baseline
│   └── result_recorder.py              # TO BUILD: saves all artifacts to disk
│
├── run_blinded_trial.sh                 # TO BUILD: orchestrates one trial (one word, one run)
├── run_batch.sh                         # TO BUILD: batch runner for word lists + controls
│
├── results/                             # Trial outputs land here
│   └── {word_id}_{run_id}/
│       ├── profile.json
│       ├── step1_consonant_prompt.txt
│       ├── step1_consonant_response.txt
│       ├── step2_vowel_prompt.txt
│       ├── step2_vowel_response.txt
│       ├── step3_synthesis_prompt.txt
│       ├── step3_synthesis_response.txt
│       └── meta.json
│
├── controls/                            # Random-profile control data
│
└── evaluate.py                          # TO BUILD: comparison/summary across trials
```

## Pipeline Flow

### Step 1: Mechanical Analysis (no LLM, fully deterministic)

```
phoneme_decomposer.py:
  Input: Hungarian word string (e.g., "rút")
  Process: Apply hungarian_orthography.json longest-match-first
  Output: phoneme sequence ["r", "uː", "t"]
  
  IMPORTANT: Handle gemination (doubled consonants like "tt", "ssz") by
  deduplicating — same phoneme, same activation. Handle digraphs/trigraphs
  before single characters.

activation_profiler.py:
  Input: phoneme sequence ["r", "uː", "t"]
  Process:
    1. Look up each phoneme in phoneme_features.json → get its sound_groups
    2. For each macro-concept in erben_table.json, check if the phoneme's
       sound_groups intersect with the macro-concept's sound_groups
    3. Record which macro-concepts each phoneme activates, preserving position
    4. Separate into consonant layer and vowel layer
    5. Compute reinforcement (codes activated in BOTH layers)
    6. Compute activation density (which codes appear at multiple positions)
  Output: structured activation profile (JSON)
```

**Activation matching rule:** A phoneme activates a macro-concept if ANY of the phoneme's sound_groups appears in ANY of the macro-concept's sound_groups. This is an OR match, not AND.

Example for `r` (sound_groups: `["voiced", "alveolar", "vibrant", "alveolar_voiced"]`):
- UNEVEN has sound_group `"alveolar_voiced"` → r is in alveolar_voiced → **ACTIVATES**
- TURN has sound_group `"alveolar_voiced"` → r is in alveolar_voiced → **ACTIVATES**
- TONGUE has sound_groups `"voiced"` and `"alveolar_voiced"` → r matches both → **ACTIVATES**
- SMALLNESS has sound_groups `"voiceless"`, `"stop"`, `"voiceless_stop"` → r matches none → **does not activate**

### Step 2: Prompt Formatting (deterministic)

```
prompt_formatter.py:
  Input: activation profile from step 1, trial_type, step number
  Process:
    1. For steps 1-2: replace macro-concept names with MC-XX codes
    2. Format positional data: "Position 1: [MC-05, MC-10, MC-08], Position 3: [MC-11, MC-13]"
    3. Insert forced-choice options from option_selector.py
    4. For step 2+: insert quoted prior responses
  Output: formatted prompt string ready for LLM
```

### Step 3: Blinded LLM Calls (via `claude` CLI)

Each step is a **separate** `claude` invocation — no conversation memory carries over.
Prior commitments are injected as quoted text, not as conversation history.

```bash
# Step 1: consonant frame only, coded labels
response1=$(echo "$prompt1" | claude --print)

# Step 2: vowel fill + quoted step 1 commitment
response2=$(echo "$prompt2" | claude --print)

# Step 3: synthesis with decoded labels + both prior commitments
response3=$(echo "$prompt3" | claude --print)
```

**CRITICAL:** The `claude` CLI invocations must each be a fresh context. The model must not have access to any prior conversation about this word. Use `--print` flag for stdout capture.

### Step 4: Result Recording

```
result_recorder.py:
  Saves to results/{word_id}_{run_id}/:
    - profile.json: full mechanical analysis
    - step{N}_prompt.txt: exact prompt sent
    - step{N}_response.txt: full model response
    - meta.json: run metadata (model, timestamp, trial_type, ordering, etc.)
```

## Trial Types

Four types, controlled by `option_selector.py`:

1. **target_present**: Word's actual semantic domain IS one of the 3-4 forced-choice options
2. **target_absent**: Word's actual semantic domain is deliberately EXCLUDED from options
3. **random_profile**: Random activation profile (not from a real word), with real options
4. **null_trial**: Real word profile, but ALL options are semantically distant from actual meaning

The trial_type is recorded in meta.json but NEVER visible to the LLM.

### Option Selection Logic

```python
# option_selector.py pseudocode:

def select_options(target_domain_id, trial_type, num_options=4):
    all_domains = load("semantic_domains.json")
    
    if trial_type == "target_present":
        # Include target, fill rest randomly (excluding semantically adjacent)
        options = [target_domain] + random_sample(non_adjacent_domains, num_options - 1)
    
    elif trial_type == "target_absent":
        # Exclude target, include some adjacent and some distant
        options = random_sample(all_except_target, num_options)
    
    elif trial_type == "null_trial":
        # All options are semantically distant from target
        options = random_sample(distant_from_target, num_options)
    
    elif trial_type == "random_profile":
        # Any options — there's no real target
        options = random_sample(all_domains, num_options)
    
    shuffle(options)  # randomize position
    return options
```

**Semantic adjacency** needs to be defined — some domains are naturally close (WATER_LIQUID and AIR_WIND share fluidity) while others are distant (KINSHIP_SOCIAL and CUTTING_SHARP). This can start as a simple manual adjacency matrix in config/ and be refined over time.

## What the Config Files Contain (already built)

### erben_table.json
The 20 macro-concepts from Erben Johansson et al. (2020) Table 5. Each has:
- `id`: MC-01 through MC-20
- `sound_groups`: which phonetic feature groups activate this concept
- `primary_cardinal_sounds`: the main sounds that drive the association
- `mapping_type`: how the association works (onomatopoeia, vocal gesture, relative, circumstantial)
- `strength`: strong or weak evidence from the paper
- `contained_concepts`: the specific concepts that form the macro-concept

### phoneme_features.json
Every IPA phoneme mapped to its sound group memberships. **Critical Hungarian note:** Hungarian 'a' is /ɒ/ (low back ROUNDED) and 'á' is /aː/ (low front UNROUNDED). This flips many expectations.

**Missing phonemes to add:** Hungarian gy /ɟ/ (voiced palatal stop → voiced, palatal, stop) and ty /c/ (voiceless palatal stop → voiceless, palatal, stop). These need entries in the consonants section.

### hungarian_orthography.json
Spelling → IPA mapping. Longest-match-first rule for digraphs/trigraphs.

### concept_codes.json
Bidirectional MC-XX ↔ label mapping.

### semantic_domains.json
24 semantic domains for forced-choice trials. Each has an ID, a short label, and an evaluation description.

### prompt_templates.json
The exact prompt templates with {variable} placeholders. Three steps:
1. Consonant frame only (coded, forced-choice)
2. Vowel fill (coded, forced-choice, with quoted step 1)
3. Synthesis (decoded to human-readable, open-ended, with both priors quoted)

## Important Design Decisions

### Why separate `claude` calls per step
If we used a single conversation, the model could revise its earlier interpretations implicitly. Separate calls mean each step's context contains ONLY the current prompt and explicitly quoted prior commitments — no hidden state.

### Why coded labels in steps 1-2
The macro-concept names (ROUNDNESS, TONGUE, DEEP, etc.) are English words with semantic content. A model seeing "ROUNDNESS + DEEP + SOFTNESS" could pattern-match to meanings without any phonosemantic reasoning. Coded labels (MC-06 + MC-12 + MC-14) prevent this. Labels are decoded only at step 3 (synthesis), by which point the model is anchored to its earlier commitments.

### Why forced-choice in steps 1-2 but open-ended in step 3
Forced-choice prevents vague hedging that keeps options open for later correction. Open-ended at step 3 maximizes sensitivity for the final prediction. The combination gives us: constrained intermediate reasoning → specific final prediction.

### Why random-profile controls
The LLM is a meaning-making machine — it will find "coherent" interpretations in ANY activation profile. Random profiles establish the false-positive baseline. If real words don't beat random profiles, the framework isn't detecting signal.

## Running a Trial

```bash
# Single word, single run
./run_blinded_trial.sh --word "rút" --target-domain "SD-15" --trial-type "target_present"

# Batch with controls
./run_batch.sh --word-list words.txt --runs-per-word 5 --include-controls

# Resume a partially completed batch (skips trials with existing results/)
./run_batch.sh --word-list words.txt --runs-per-word 5 --include-controls --skip-completed

# Local Ollama model (sequential to avoid GPU contention, longer timeout for reasoning models)
./run_batch.sh --word-list words.txt --runs-per-word 10 --include-controls \
  --model "ollama:deepseek-r1:14b" --reasoning-prompt --skip-completed --parallel 1 --timeout 600
```

### Batch runner flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `sonnet` | `sonnet`, `opus`, or `ollama:MODEL_NAME` |
| `--reasoning-prompt` | off | Structured counting-based system prompts for weaker models |
| `--decode-all-steps` | off | Human-readable concept labels in steps 1-2 (instead of hex tokens) |
| `--skip-completed` | off | Skip trials that already have results on disk (resume support) |
| `--parallel` | 4 | Max concurrent trials; use 1 for local Ollama models |
| `--timeout` | 600 | Seconds per model call (Ollama only; Claude CLI uses 120s) |
| `--dry-run` | off | Run full pipeline logic without calling the model; does NOT write results to disk |

The `--target-domain` is the human-assigned correct semantic domain for the word. This requires a pre-annotated word list where each word has been manually tagged with its best-matching domain from semantic_domains.json.

### Word List Format

```tsv
word	target_domain	actual_meaning
rút	SD-15	ugly, foul, repulsive
lop	SD-24	steal, sneak, smuggle
víz	SD-01	water
tűz	SD-03	fire
kör	SD-20	circle
lyuk	SD-14	hole, opening
```

## Evaluation Metrics

After running batches, `evaluate.py` computes:

1. **Hit rate by trial type**: % correct picks across target_present trials (baseline = 25-33% depending on option count)
2. **False positive rate**: how often the model picks confidently on target_absent and null trials
3. **Random baseline**: hit rate on random_profile trials (should be at chance)
4. **Signal strength**: real-word hit rate minus random-profile hit rate
5. **Synthesis consistency**: how often step 3 prediction aligns with step 1-2 choices vs. introduces new information
6. **Confidence calibration**: does the model hedge more when it should?

Statistical significance: binomial test against chance (p < 0.05 requires roughly 32+ hits out of 100 with 4 options).

## Code Style

- Use descriptive function and variable names (e.g., `compute_macro_concept_activations`, not `get_mc`)
- Type hints on all functions
- Docstrings explaining the what and why, not just the how
- No external dependencies beyond Python stdlib + json
- All config loading goes through a single `load_config()` utility
- Every intermediate artifact is saved to disk for auditability

## CRITICAL: The Activation Density Problem

Testing the mechanical matching reveals that every phoneme activates 4-10 macro-concepts
out of 20. Front rounded vowels (ö, ő, ü, ű) hit 10 out of 20. A 3-phoneme word covers
12-15 unique macro-concepts. Everything connects to everything — the signal is drowned in noise.

**Root cause:** Broad sound groups like `[voiced]` and `[voiceless]` match too many phonemes.
Every voiced consonant activates MOTHER, INFANCY, DEIXIS, TONGUE. Every voiceless consonant
activates AIRFLOW, PHARYNGEAL, EXPULSION, UNEVEN, SMALLNESS, HARDNESS.

**Proposed solution: activation strength tiers.** The paper's `primary_cardinal_sounds` field
tells us which phonemes actually DROVE each association statistically. Use this for tiered matching:

- **Strong activation**: the phoneme IS one of the macro-concept's primary cardinal sounds
  (e.g., r → TURN because r is TURN's primary cardinal sound)
- **Medium activation**: the phoneme matches a SPECIFIC sound group (alveolar_voiced, nasal_voiced, etc.)
  but is NOT a primary cardinal sound
- **Weak activation**: the phoneme matches only a BROAD sound group (voiced, voiceless, rounded_vowel, etc.)

The activation profile sent to the LLM should include these tiers. For `rút`:

```
Position 1 (consonant): 
  STRONG: MC-10 (TURN), MC-05 (UNEVEN)    # r is primary cardinal for both
  MEDIUM: MC-08 (TONGUE)                    # r matches alveolar_voiced but l is primary
  WEAK:   MC-16, MC-19, MC-20              # r matches broad [voiced] only

Position 2 (vowel):
  STRONG: MC-06 (ROUNDNESS), MC-12 (DEEP), MC-14 (SOFTNESS)  # u is primary cardinal
  MEDIUM: MC-01 (AIRFLOW)                   # u matches [+round][back] but p is also primary
  WEAK:   MC-02 (PHARYNGEAL), MC-19         # broad matches only

Position 3 (consonant):
  STRONG: MC-11 (SMALLNESS), MC-13 (HARDNESS)  # t is primary for SMALLNESS
  MEDIUM: MC-03 (EXPULSION), MC-05 (UNEVEN)    # t matches via [–voice] specifically
  WEAK:   MC-01, MC-02, MC-17                   # broad matches
```

This gives the blinded LLM meaningful signal to work with. The strong activations for rút
are: TURN, UNEVEN, ROUNDNESS, DEEP, SOFTNESS, SMALLNESS, HARDNESS — which matches
our manual readings almost exactly.

**Implementation:** `activation_profiler.py` should classify each activation as strong/medium/weak.
The prompt templates should present tiers separately or use a notation like:
`Position 1: MC-10***, MC-05***, MC-08**, MC-16*, MC-19*, MC-20*`

## Model Support

The pipeline supports multiple backends via `--model`:

| Model spec | Backend | Notes |
|---|---|---|
| `sonnet`, `opus`, `claude` | Claude CLI (`claude --print`) | Requires Claude Code CLI installed |
| `ollama:MODEL_NAME` | Ollama HTTP API (`localhost:11434`) | Local inference; use `--parallel 1` |

### DeepSeek-R1 via Ollama

DeepSeek-R1 outputs `<think>...</think>` reasoning blocks before its final answer. The pipeline automatically strips these before parsing choices, but saves the full response (including thinking) to disk for auditing. Use `--timeout 600` and `--parallel 1`.

### Experimental results (rút/víz/tűz, 10 runs each, reasoning-prompt, hex tokens)

| Configuration | Hit rate (step 2) | p-value | Controls |
|---|---|---|---|
| Claude sonnet (hex tokens) | 50% (15/30) | 0.003 | 0% |
| DeepSeek-R1 14B via Ollama (hex tokens) | 50% (15/30) | 0.003 | 0% |
| Qwen 2.5 14B via Ollama (hex tokens) | 23% (7/30) | 0.65 | 0% |
| Qwen 2.5 14B via Ollama (decoded labels) | 14% (4/29) | 0.95 | 0% |
| Qwen 2.5 14B via Ollama (reasoning prompt) | 29% (9/31) | 0.37 | 0% |

Key finding: DeepSeek-R1 14B (local) matches Claude at 50%, both well above 25% chance.
Step 1→2 correction rate for R1: 11 corrections, 0 regressions — the vowel layer is doing real work.

## Known Gaps / TODO

- [ ] Add gy /ɟ/ and ty /c/ to phoneme_features.json (voiced/voiceless palatal stops)
- [ ] Semantic adjacency matrix for option_selector.py
- [ ] Word list with manual target_domain annotations
- [ ] Order-effect testing: run each word consonant-first AND vowel-first
- [ ] Specificity scoring rubric for evaluate.py
- [ ] **Refinement**: `t` classifies as WEAK for HARDNESS because `k` is the primary cardinal, even though `t` is also a voiceless stop. Consider adding `"voiceless_stop"` to HARDNESS's sound_groups so `t` gets MEDIUM. Same pattern may apply to other macro-concepts where the paper tested [–voice] + [stop] as separate groups but clearly meant voiceless stops. Review all entries for this.
- [ ] **Refinement**: INFANCY and FATHER activate strongly for common phonemes (u→INFANCY, t→FATHER) because these are circumstantial mappings driven by baby sounds and kinship. Consider whether circumstantial mappings should be weighted differently for non-kinship words, or whether this is fine as-is (the blinded LLM should learn to ignore kinship signals when the rest of the profile doesn't support it).
