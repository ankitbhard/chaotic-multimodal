"""Patch-embedding ROI selection pilot; not raw WSI segmentation.

Input: per-slide .h5 with 'features' [N,D] and optional 'coords' [N,2].
Output: selected-patch pooled vector [D] and selected indices.
Unsupervised, label-free feature-centrality and diversity heuristics.
"""
import argparse,json
from pathlib import Path
import numpy as np
import h5py


def select_roi(features,k=64,diversity=0.25):
    x=np.asarray(features,dtype=np.float32)
    if x.ndim!=2 or len(x)==0: raise ValueError("Expected nonempty [N,D] patch features")
    if not np.isfinite(x).all(): raise ValueError("Nonfinite patch features")
    k=min(int(k),len(x))
    if k<1: raise ValueError("k must be >= 1")
    norm=x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)
    center=norm.mean(axis=0)
    relevance=norm@center
    selected=[int(np.argmax(relevance))]
    # Greedy relevance/diversity tradeoff; O(N*k*D), suitable for a pilot.
    max_similarity=norm@norm[selected[0]]
    for _ in range(1,k):
        score=relevance-diversity*max_similarity
        score[selected]=-np.inf
        j=int(np.argmax(score))
        selected.append(j)
        max_similarity=np.maximum(max_similarity,norm@norm[j])
    return np.asarray(selected,dtype=np.int64),x[selected].mean(axis=0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,help="Directory of per-slide HDF5 patch features")
    p.add_argument("--output",required=True)
    p.add_argument("--top-k",type=int,default=64)
    p.add_argument("--diversity",type=float,default=0.25)
    a=p.parse_args()
    src=Path(a.input);dst=Path(a.output)
    if not src.is_dir(): raise FileNotFoundError(src)
    files=sorted(src.glob("*.h5"))
    if not files: raise RuntimeError("No .h5 files found; raw WSIs require feature extraction first")
    dst.mkdir(parents=True,exist_ok=True)
    records=[]
    for path in files:
        with h5py.File(path) as h:
            if "features" not in h:
                raise ValueError(f"{path}: missing 'features' [N,D]")
            features=h["features"][()]
            coords=h["coords"][()] if "coords" in h else None
        indices,pooled=select_roi(features,a.top_k,a.diversity)
        np.save(dst/(path.stem+".npy"),pooled)
        np.save(dst/(path.stem+"_indices.npy"),indices)
        if coords is not None:
            np.save(dst/(path.stem+"_coords.npy"),coords[indices])
        records.append({"slide":path.name,"patches":len(features),"selected":len(indices),
                        "has_coords":coords is not None})
    (dst/"manifest.json").write_text(json.dumps(records,indent=2))
    print(f"Processed {len(records)} slides; output={dst}")
    print("WARNING: These .npy pooled features are NOT consumed by the existing survival loader.")
    print("Use a validated patient-matching integration before survival training.")


if __name__=="__main__":main()
