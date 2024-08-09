import json
from glob import glob
from os import makedirs


def read_jsonl(fname):
    with open(fname) as jsonl:
        for line in jsonl:
            yield json.loads(line)

def load_responses(response_fname_lis):
    response_dct = {}
    for response_fname in response_fname_lis:
        with open(response_fname) as in_f:
            response_dct.update(json.load(in_f))
    return response_dct

def build_dataset(context_fname, response_fname_lis, out_fname):
    response_dct = load_responses(response_fname_lis)
    with open(out_fname,"w") as o_f:
        for js in read_jsonl(context_fname):
            response = response_dct[js["utterances"][-1]] if js["utterances"][-1] in response_dct else "[[TBA]]"
            js["utterances"] = js["utterances"][:-1] + [response]
            json.dump(js, o_f)
            o_f.write("\n")


if __name__=="__main__":
    makedirs("./dataset/indomain", exist_ok=True)
    makedirs("./dataset/outdomain_daily", exist_ok=True)
    makedirs("./dataset/outdomain_topical", exist_ok=True)
    response_fname_lis = glob("./responses/*/*.json")

    # 1. Build our large dataset
    context_fname = "./contexts/indomain/contexts_indomain_all.jsonl"
    out_fname = "./dataset/indomain/indomain_all.jsonl"
    build_dataset(context_fname, response_fname_lis, out_fname)

    # 2. Build indomain testset
    for model in ["blender2-3B", "blender3-3B", "blender3-30B", "blender3B", "opt-60B", "plato2", "platoxl", "chatgpt"]:
        context_fname = f"./contexts/indomain/contexts_indomain_test_{model}.jsonl"
        out_fname = f"./dataset/indomain/indomain_test_{model}.jsonl"
        build_dataset(context_fname, response_fname_lis, out_fname)

    # 3. Build outdomain(topicalchat) testset
    for model in ["blender2-3B", "blender3-3B", "blender3-30B", "blender3B", "opt-60B", "plato2", "platoxl"]:
        context_fname = f"./contexts/outdomain_topical/contexts_outdomain_{model}.jsonl"
        out_fname = f"./dataset/outdomain_topical/topical_test_{model}.jsonl"
        build_dataset(context_fname, response_fname_lis, out_fname)

    # 4. Build outdomain(dailydialog) testset
    for model in ["blender2-3B", "blender3-3B", "blender3-30B", "blender3B", "opt-60B", "plato2", "platoxl"]:
        context_fname = f"./contexts/outdomain_daily/contexts_outdomain_{model}.jsonl"
        out_fname = f"./dataset/outdomain_daily/daily_test_{model}.jsonl"
        build_dataset(context_fname, response_fname_lis, out_fname)