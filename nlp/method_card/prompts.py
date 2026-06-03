"""Few-shot prompt templates for mechanism-level method card extraction.

The literal word "JSON" MUST stay in SYSTEM_PROMPT: DeepSeek's
response_format={"type": "json_object"} rejects prompts that don't mention JSON.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an expert at reading machine-learning papers, especially graph neural "
    "networks and recommender systems. Given a paper's title and abstract, extract a "
    "mechanism-level method card.\n\n"
    "Return ONLY a single STRICT JSON object (no prose, no markdown fences) with EXACTLY "
    "these keys:\n"
    '  "task"      (string): the problem the paper solves, e.g. "top-K recommendation".\n'
    '  "input"     (string): what the model consumes, e.g. "user-item interaction graph".\n'
    '  "output"    (string): what the model produces, e.g. "ranked list of items per user".\n'
    '  "backbone"  (string): the core architecture / encoder.\n'
    '  "loss"      (string): the training objective(s).\n'
    '  "key_idea"  (string): ONE sentence capturing the central mechanism. Most important field.\n'
    '  "datasets"  (list of strings): benchmark datasets used.\n'
    '  "metrics"   (list of strings): evaluation metrics used.\n\n'
    "Rules:\n"
    "- Be concise and factual. Extract only what the abstract supports.\n"
    "- If the abstract explicitly names an objective such as BPR, InfoNCE, "
    'cross-entropy, pairwise ranking, or regularization, put it in "loss".\n'
    "- If the abstract explicitly names benchmark datasets or metrics, extract them "
    'exactly into "datasets" and "metrics".\n'
    '- Do not leave "datasets" or "metrics" empty when they are explicitly mentioned.\n'
    '- Never write placeholder phrases such as "not specified", "not named", '
    '"not mentioned", "not explicitly named", "not explicitly named in abstract", '
    '"not explicitly stated", or "N/A". Use "" or [] instead.\n'
    "- Do not infer common losses, datasets, or metrics from the task type. Only extract "
    "them when explicitly stated in the abstract.\n"
    "- For datasets and metrics, include only concrete names explicitly stated in the "
    'abstract; if the abstract only says "real-world datasets" or '
    '"extensive experiments", "publicly accessible benchmarks", or gives only a dataset '
    'count without names such as "two public benchmark datasets" or '
    '"three real-world datasets", use [].\n'
    '- Never write hedged guesses such as "implicitly", "typical for", "based on context", '
    '"likely", or "e.g." in any field; leave the field empty instead.\n'
    "- Final self-check before returning JSON: if any string field would contain a banned "
    'placeholder or hedge, replace the whole field with ""; if any list item would contain '
    "a banned placeholder, hedge, or generic unnamed dataset/metric description, remove "
    "that item; if no concrete items remain, use [].\n"
    '- If a field cannot be determined from the abstract, use an empty string "" for '
    "string fields or an empty list [] for list fields. NEVER guess or hallucinate.\n"
    "- Output must be valid JSON parseable by a strict JSON parser."
)

# Hand-crafted exemplars (well-known GNN-recsys papers). paper_id values are placeholders;
# build_prompt strips them before showing the model — they document provenance only.
FEW_SHOT_EXAMPLES: list[dict] = [
    {
        "paper_id": "example-lightgcn",
        "title": "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation",
        "abstract": (
            "Graph Convolution Network (GCN) has become a new state-of-the-art for "
            "collaborative filtering, yet the reasons for its effectiveness are not well "
            "understood. We empirically show that two common GCN designs, feature "
            "transformation and nonlinear activation, contribute little to collaborative "
            "filtering and can even degrade performance. We propose LightGCN, which keeps "
            "only the most essential component, neighborhood aggregation, linearly "
            "propagating user and item embeddings over the interaction graph and using the "
            "weighted sum of all-layer embeddings as the final representation. Trained with "
            "the BPR loss, LightGCN substantially outperforms NGCF on Gowalla, Yelp2018, and "
            "Amazon-Book."
        ),
        "card": {
            "task": "top-K item recommendation (collaborative filtering)",
            "input": "user-item interaction graph (implicit feedback bipartite graph)",
            "output": "ranked list of items per user (predicted preference scores)",
            "backbone": "simplified GCN (no feature transformation, no nonlinearity)",
            "loss": "BPR",
            "key_idea": (
                "strip non-essential operations from GCN-based CF, keep only neighborhood "
                "aggregation"
            ),
            "datasets": ["Gowalla", "Yelp2018", "Amazon-Book"],
            "metrics": [],
        },
    },
    {
        "paper_id": "example-simgcl",
        "title": (
            "Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for "
            "Recommendation"
        ),
        "abstract": (
            "Contrastive learning (CL) recently boosts graph collaborative filtering, where "
            "graph augmentations such as edge or node dropout generate contrastive views. We "
            "reveal that the InfoNCE loss, rather than the graph augmentations, is the key to "
            "performance, and that augmentations can even be discarded. We propose SimGCL, "
            "which drops graph augmentations and instead adds uniform random noise to the "
            "embedding space to build contrastive views. Built on a LightGCN encoder and "
            "optimized with a joint BPR and InfoNCE objective, SimGCL improves both accuracy "
            "and training efficiency on Douban-Book, Yelp2018, and Amazon-Book."
        ),
        "card": {
            "task": "top-K recommendation with graph contrastive learning",
            "input": "user-item interaction graph (implicit feedback)",
            "output": "ranked list of items per user",
            "backbone": "LightGCN encoder",
            "loss": "BPR + InfoNCE",
            "key_idea": (
                "replace graph augmentations with uniform embedding noise for contrastive " "views"
            ),
            "datasets": ["Douban-Book", "Yelp2018", "Amazon-Book"],
            "metrics": [],
        },
    },
    {
        "paper_id": "example-bitgcf",
        "title": (
            "Cross Domain Recommendation via Bidirectional Transfer Graph Collaborative "
            "Filtering Network"
        ),
        "abstract": (
            "Cross-domain recommendation alleviates data sparsity by transferring knowledge "
            "across domains. We propose BiTGCF, a bidirectional transfer graph collaborative "
            "filtering network. It performs graph convolution on the user-item interaction "
            "graphs of two domains and transfers information bidirectionally through users "
            "shared between domains, fusing common and domain-specific features. Optimized "
            "with a BPR loss in each domain plus a transfer regularization term, BiTGCF "
            "improves recommendation in both source and target domains on several Amazon "
            "cross-domain pairs."
        ),
        "card": {
            "task": "cross-domain recommendation",
            "input": (
                "user-item interaction graphs from two domains with shared (overlapping) users"
            ),
            "output": "ranked item recommendations in both source and target domains",
            "backbone": "bidirectional transfer GCN",
            "loss": "BPR per domain + transfer regularization",
            "key_idea": (
                "bidirectional information transfer between source and target domains via "
                "shared user embeddings"
            ),
            "datasets": ["Amazon (cross-domain pairs)"],
            "metrics": [],
        },
    },
]


def _user_content(title: str, abstract: str) -> str:
    """Render the user-turn text shown to the model for one target paper."""
    return (
        f"Title: {title}\n\n"
        f"Abstract:\n{abstract or '(no abstract available)'}\n\n"
        "Extract the method card as a JSON object."
    )


def build_prompt(abstract: str, title: str) -> list[dict]:
    """Build OpenAI-format messages: system + 3 few-shot (user, assistant) pairs + target user."""
    import json

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": _user_content(ex["title"], ex["abstract"])})
        messages.append(
            {"role": "assistant", "content": json.dumps(ex["card"], ensure_ascii=False)}
        )
    messages.append({"role": "user", "content": _user_content(title, abstract)})
    return messages
