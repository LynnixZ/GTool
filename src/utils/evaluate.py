import pandas as pd
import re
import Levenshtein

def remove_tools(input_str):
    return re.sub(r'Tool\d+: ', '', input_str)

def cat_metric(pred_list, truth_list):
    TP = sum(i in truth_list for i in pred_list)
    FP = sum(i not in truth_list for i in pred_list)
    FN = sum(i not in pred_list for i in truth_list)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0

def _row_metrics(pred, label):
    """Per-sample (node_f1, edge_f1, ned_ratio, gold_tool_count)."""
    p = pred.split("</s>")[0]
    p = p.replace("<|endoftext|>", "")
    p = remove_tools(p)
    l = remove_tools(label)
    pred_list = p.lower().rstrip().split("\n")
    label_list = l.lower().rstrip().split("\n")

    pred_link_list = [(pred_list[i], pred_list[i+1]) for i in range((len(pred_list) - 1))]
    label_link_list = [(label_list[i], label_list[i+1]) for i in range((len(label_list) - 1))]

    node_f1 = cat_metric(pred_list, label_list)
    edge_f1 = cat_metric(pred_link_list, label_link_list)
    ned_ratio = Levenshtein.ratio(pred_list, label_list)
    n_gold = len(label_list)  # number of gold tools = number of "Tool i:" lines
    return node_f1, edge_f1, ned_ratio, n_gold


def eval(path):
    df = pd.read_json(path, lines=True)
    # compute accuracy
    node_f1_all, edge_f1_all, ned_all = 0, 0, 0

    for pred, label, idx in zip(df["pred"], df["label"], df["id"]):
        node_f1, edge_f1, ned_ratio, _ = _row_metrics(pred, label)
        node_f1_all += node_f1
        edge_f1_all += edge_f1
        ned_all += ned_ratio
    node_f1 = node_f1_all / len(df)
    edge_f1 = edge_f1_all / len(df)
    ned = 1 - (ned_all / len(df))
    return node_f1, edge_f1, ned


def _bucket_name(n):
    """Bucket by gold tool count: 2, 3, or 4+ (1 kept separate if it ever appears)."""
    if n <= 1:
        return "tool=1"
    if n == 2:
        return "tool=2"
    if n == 3:
        return "tool=3"
    return "tool>=4"


# Fixed display order for the per-bucket breakdown.
BUCKET_ORDER = ["tool=1", "tool=2", "tool=3", "tool>=4", "overall"]


def eval_grouped(path):
    """Evaluate, broken down by gold tool count, plus an overall score.

    Returns ``{bucket_name: {"node_f1", "edge_f1", "ned", "count"}}`` where
    bucket_name is one of ``tool=2 / tool=3 / tool>=4`` (and ``tool=1`` if any
    single-tool sample exists) plus ``overall``.
    """
    df = pd.read_json(path, lines=True)
    groups = {}  # name -> [node_sum, edge_sum, ned_ratio_sum, count]

    def add(name, nf, ef, r):
        g = groups.setdefault(name, [0.0, 0.0, 0.0, 0])
        g[0] += nf; g[1] += ef; g[2] += r; g[3] += 1

    for pred, label in zip(df["pred"], df["label"]):
        node_f1, edge_f1, ned_ratio, n_gold = _row_metrics(pred, label)
        add("overall", node_f1, edge_f1, ned_ratio)
        add(_bucket_name(n_gold), node_f1, edge_f1, ned_ratio)

    result = {}
    for name, (ns, es, rs, c) in groups.items():
        result[name] = {
            "node_f1": ns / c,
            "edge_f1": es / c,
            "ned": 1 - (rs / c),
            "count": c,
        }
    return result


def format_grouped(result):
    """Pretty one-line-per-bucket table for printing."""
    lines = [f"{'group':10s} {'count':>7s} {'node_f1':>9s} {'edge_f1':>9s} {'ned':>9s}"]
    for name in BUCKET_ORDER:
        if name not in result:
            continue
        m = result[name]
        lines.append(f"{name:10s} {m['count']:>7d} {m['node_f1']:>9.4f} {m['edge_f1']:>9.4f} {m['ned']:>9.4f}")
    return "\n".join(lines)
