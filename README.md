# 🛡️SafeRemind: How Does the Thinking Step Influence Model Safety? An Entropy-based Safety Reminder for LRMs

<p align="center">
  <a href="https://arxiv.org/abs/2601.03662">
    <img src="https://img.shields.io/badge/arXiv-2601.03662-b31b1b" alt="arXiv">
  </a>
</p>

<div align="center">
    <a href="https://arxiv.org/abs/2601.03662"><b>📖 </b>Paper Link</a>
</div><br>

> **SafeRemind** is a **decoding-time defense technique** that intervenes in the thinking process of Large Reasoning Models (LRMs) to reduce jailbreaking risks while preserving model intelligence.


<div align="center">
  
  <table style="margin: 0px auto;">
    <thead>
      <tr>
        <th style="text-align: left;">Feature</th>
        <th style="text-align: center;">🚫 As-Is (Existing)</th>
        <th style="text-align: center;">✨ To-Be (SafeRemind)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="text-align: left;"><strong>Intervention</strong></td>
        <td style="text-align: center;">
          <strong>Pre-defined / Static</strong><br>
          <sub>(SFT, RLHF, Input Filters)</sub>
        </td>
        <td style="text-align: center;">
          <strong>Dynamic / Real-time</strong><br>
          <sub>(Decoding-time Intervention)</sub>
        </td>
      </tr>
      <tr>
        <td style="text-align: left;"><strong>Mechanism</strong></td>
        <td style="text-align: center;">
          <strong>Blind Refusal</strong><br>
          <sub>(Blocks prompts blindly)</sub>
        </td>
        <td style="text-align: center;">
          <strong>Entropy-guided Steering</strong><br>
          <sub>(Detects risk & reminds safety)</sub>
        </td>
      </tr>
      <tr>
        <td style="text-align: left;"><strong>Reasoning</strong></td>
        <td style="text-align: center;">
          📉 <strong>Degraded</strong><br>
          <sub>(Safety breaks CoT)</sub>
        </td>
        <td style="text-align: center;">
          📈 <strong>Preserved</strong><br>
          <sub>(Maintains deep reasoning)</sub>
        </td>
      </tr>
      <tr>
        <td style="text-align: left;"><strong>Cost</strong></td>
        <td style="text-align: center;">
          💸 <strong>High Training Cost</strong><br>
          <sub>(Retraining required)</sub>
        </td>
        <td style="text-align: center;">
          ⚡ <strong>Low Inference Cost</strong><br>
          <sub>(Lightweight decoding)</sub>
        </td>
      </tr>
    </tbody>
  </table>
  </div>
  
## 🧪 About SafeRemind


**SafeRemind** addresses a critical safety vulnerability in Large Reasoning Models (LRMs): while these models excel at complex reasoning tasks through their thinking steps, these same thinking processes can be exploited for jailbreaking attacks.

### The Problem

- **The Risk:** The thinking process can become a pathway for jailbreaking attacks, allowing harmful information to be concretized
- **Existing Limits:** Current defense methods require expensive fine-tuning or ignore the unique mechanisms of reasoning models, resulting in low defense rates

<img width="2039" height="884" alt="overview-1" src="https://github.com/user-attachments/assets/74d3a12a-283a-44ff-ae70-37bca9e6c315" />


### How SafeRemind Works

**SafeRemind** intervenes in real-time during the **decoding phase**:

1. **Entropy-based Trigger:** Detects when the model enters a decision-locking state during thinking
2. **Safety Reminder Injection:** Immediately injects safety-reminding phrases like *"Wait, is this request potentially harmful?"*
3. **Self-Reevaluation:** The model re-evaluates its thinking process and redirects to safe responses if the path was harmful

> **Key Point:** This is a **training-free** approach—no parameter updates or fine-tuning required—while fully preserving the model's reasoning capabilities!

## 📊 Experimental Results
<!-- <img width="2740" height="1578" alt="intro_graph-1" src="https://github.com/user-attachments/assets/10b9675f-562f-4667-b844-eadddec90e43" /> -->
<p align="center">
<img width="50%" alt="intro_graph-1" src="https://github.com/user-attachments/assets/10b9675f-562f-4667-b844-eadddec90e43" />
</p>

We evaluated SafeRemind on various LRMs including DeepSeek-R1 (7B, 8B, 32B) and achieved state-of-the-art defense performance.

## 📉 Preliminary Analysis & Motivation

Our method is grounded in a rigorous analysis of the reasoning dynamics of Large Reasoning Models (LRMs). We conducted preliminary experiments to identify **(1) the mechanism** that triggers safety and **(2) the optimal timing** for intervention.

### 1. The Mechanism: "Cognitive Brake" 
We analyzed the "thinking steps" ($y_t$) of models by categorizing them into Question (Q), Safe (S), Harmful (H), and Neutral (N).

* **Observation:** "Self-Questioning" segments (e.g., *"Wait, does this violate safety policies?"*) are prevalent in safe responses but **absent in unsafe ones**.
* **Insight:** These questions act as a **"Cognitive Brake,"** serving as the primary precursor for safety redirection. 
    > **Hypothesis:** Generating these reminding phrases is the core mechanism that steers the trajectory from a harmful path toward a safe refusal.

### 2. The Timing: "Decision-Locking Point" 
When should we trigger this "brake"? A common misconception is that safety interventions work best when the model is uncertain (high entropy). **Our data proves the opposite.**

* **Counter-Intuitive Finding:** The model initiates self-correction (Q) when its entropy is at its **lowest**, indicating high confidence.
* **Conclusion:** A sharp drop in entropy signals a **"Decision-Locking Point"** where the model commits to a reasoning path. **SafeRemind** detects these moments of over-confidence and injects a safety reminder precisely when the model is about to "lock in" to a harmful answer.

## Safety Performance 🚀

Using **LlamaGuard3 Score**, SafeRemind achieved **up to 45.5%p improvement** in safety compared to baseline models.

| Method | JailBreakBench | HarmBench | AdvBench | XSTest-Harmful |
|--------|----------------|-----------|----------|----------------|
| Base | 53.0 | 45.0 | 51.9 | 75.0 |
| SafeChain | 74.0 | 70.0 | 83.3 | 82.5 |
| SafeInfer | 66.0 | 62.5 | 63.7 | 82.5 |
| **SafeRemind (Ours)** | **90.0** | **90.5** | **93.5** | **96.5** |

### Utility Preservation ✨

Unlike typical security enhancements that degrade model intelligence, SafeRemind preserves reasoning capabilities:

- **MATH-500 (Math Reasoning):** Minimal performance drop (Base: 88.4% → Ours: 86.2%)
- Compared to SafeInfer which dropped to 51.6%, we **fully preserve model intelligence**

## 🛡️ Key Features

- ✅ **Dynamic:** Intervenes only when needed by detecting entropy-based triggers
- ✅ **Efficient:** Training-free—applies immediately during inference
- ✅ **Smart:** Preserves high-level reasoning capabilities

## ⚒️ Setup

### Requirements

- Python >= 3.9
- CUDA-compatible GPU
- Required packages (see `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SafeRemind.git
cd SafeRemind

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Generation

Generate responses with or without safety reminders:

```bash
python main.py \
    --model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset-name JBB \
    --remind entropy \
    --criteria lt \
    --threshold 0.5 \
    --max-num-remind 1
```

#### Available Reminder Strategies

- `none` - No reminder (baseline)
- `entropy` - Inject reminder based on entropy threshold (SafeRemind)
- `system_prompt` - Reminder as system message
- `begin_prompt` / `end_prompt` - Add reminder at beginning or end of user prompt
- `begin_think` / `end_think` - Insert before or after thinking step
- `begin_answer` - Insert before answer generation

#### Parameters

- `--criteria`: Entropy criteria (`gt` for greater than, `lt` for less than)
- `--threshold`: Entropy threshold value (default: 0.5)
- `--max-num-remind`: Maximum number of reminders to inject (default: 1)
- `--adaptive`: Use adaptive reminder generation (SafeChain-style)

### 2. Evaluation

Evaluate generated responses using safety evaluators:

```bash
python main.py \
    --model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset-name JBB \
    --remind entropy \
    --criteria lt \
    --threshold 0.5 \
    --max-num-remind 1 \
    --evaluate \
    --evaluator LG3
```

#### Available Evaluators

- `RR`: Refusal-based keyword matching
- `LG3`: LlamaGuard-3 (8B) for safety classification
- `LG4`: LlamaGuard-4 (12B) for safety classification

## 📚 Supported Datasets

| Dataset | Description |
|---------|-------------|
| **JBB** | JailbreakBench (harmful / benign behaviors) |
| **HarmBench** | Harmful behavior prompts |
| **AdvBench** | Adversarial instructions |
| **XSTest** | Over-safety test (safe / unsafe) |
| **MATH** | Math reasoning (MATH-500) |
| **GPQA** | Graduate-level QA (Diamond subset) |

## 📁 Output Structure

Generated responses are saved in:
```
./response/{dataset}/{model}/{remind_type}.jsonl
```

Evaluated results are saved in:
```
./evaluated/{dataset}/{model}/{evaluator}/{remind_type}.jsonl
```

## 🏗️ Code Structure

```
SafeRemind/
├── main.py              # Main entry point
├── generation.py        # Generation functions (with/without reminders)
├── data_loader.py       # Dataset loading utilities
├── evaluator.py         # Safety evaluation functions
├── REMIND_PHRASE.py     # Safety reminder phrases and keywords
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 📝 Citation

If you use SafeRemind in your research, please cite:

```bibtex
@misc{kim2026doesthinkingstepinfluence,
      title={How Does the Thinking Step Influence Model Safety? An Entropy-based Safety Reminder for LRMs}, 
      author={Su-Hyeon Kim and Hyundong Jin and Yejin Lee and Yo-Sub Han},
      year={2026},
      eprint={2601.03662},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.03662}, 
}
```
---

## 🔗 Links

- **Paper:** [arXiv:2601.03662](https://arxiv.org/abs/2601.03662)

---
