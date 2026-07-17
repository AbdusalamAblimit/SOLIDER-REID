# encoding: utf-8
"""
AG-ReID.v2 OFFICIAL two-protocol meta-dataset, CARGO-compatible.

This is the SECOND benchmark for the AIRL / OVLI method paper (cross-dataset main
table, paired with CARGO's A<->G). It wraps the *validated* official-protocol
loader `agreid_v2_dataset.AGReIDV2` (which selects the exact images listed in the
official `exp{N}_*.txt` files) and combines the two cross-platform directions that
mirror CARGO's A<->G:

    A->G  ==  exp1  aerial_to_cctv   (query C0 UAV     -> gallery C3 CCTV)
    G->A  ==  exp4  cctv_to_aerial   (query C3 CCTV    -> gallery C0 UAV)

WHY a combined object (and not just AGReIDV2(protocol=...) twice)
----------------------------------------------------------------
The existing CARGO training/eval loop (cargo_cvpb/afd_train.py) keys EVERYTHING
off a single `dataset` with `.train / .query / .gallery` and recovers the two
cross-view directions with `filter_by_view(dataset.query|gallery, 'Aerial'|'Ground')`.
We make the official per-protocol numbers fall out of that SAME machinery with
zero changes to the eval / AIRL-iso code by laying the two protocols out so the
view filter selects exactly the right official subset:

    self.query   = exp1.query (all view=='Aerial')  +  exp4.query (all view=='Ground')
    self.gallery = exp1.gallery (all view=='Ground') +  exp4.gallery (all view=='Aerial')

    filter_by_view(query,   'Aerial') == exp1 query  (UAV)    \\  A->G == official exp1
    filter_by_view(gallery, 'Ground') == exp1 gallery (CCTV)  /
    filter_by_view(query,   'Ground') == exp4 query  (CCTV)   \\  G->A == official exp4
    filter_by_view(gallery, 'Aerial') == exp4 gallery (UAV)   /

So run_cross_view_eval()'s
    A->G : q_aerial vs g_ground   -> official exp1
    G->A : q_ground vs g_aerial   -> official exp4
and the [mean] is the AG-ReID.v2 cross-platform mean, the analogue of CARGO's
A<->G mean. eval_market's same-(pid,camid) junk removal is a no-op across a
cross-platform direction (query and gallery never share a camid: 0 vs 3), so the
combined gallery does not perturb either direction's official number.

pid handling
------------
The validated AGReIDV2 emits the raw FOLDER-NAME string as the query/gallery pid
(relabel=False). The downstream extract_features() does
`torch.cat(pids).numpy()`, which needs INT pids. exp1 and exp4 share the SAME test
identity folders, so we build ONE folder-name -> contiguous-int map over BOTH
protocols' query+gallery and relabel every test sample through it. A query and its
gallery matches therefore carry the SAME int pid in either direction (string ==
guaranteed identical folder name -> same int). Train pids come straight from the
exp1 AGReIDV2 (already relabeled to [0, num_train_pids); train is protocol-
independent -- the whole train_all set -- so exp1 vs exp4 give the identical train).

The on-disk image selection, the field parsing, and the official counts are all
inherited verbatim from agreid_v2_dataset.AGReIDV2 (already verified by the user
against the official spec: exp1 2356q/6347g, exp4 1811q/14362g).
"""
import os

from agreid_v2_dataset import AGReIDV2


class AGReIDV2Combined(object):
    """Official AG-ReID.v2 exp1(A->G) + exp4(G->A), CARGO-compatible.

    Attributes (each a list of dicts {img_path, pid:int, camid:int, view}):
        self.train, self.query, self.gallery
    plus the CARGO-style counts:
        self.num_train_pids / num_train_imgs / num_train_cams
        self.num_query_pids / num_gallery_pids
        self.pid2label       (train folder-name -> [0, num_train_pids))
        self.test_pid2label  (test  folder-name -> [0, num_test_pids))
    """

    # the two official directions that pair with CARGO's A<->G
    AERIAL_TO_GROUND = 'exp1'   # aerial_to_cctv  (A->G)
    GROUND_TO_AERIAL = 'exp4'   # cctv_to_aerial  (G->A)

    def __init__(self, root='/root/work/SOLIDER-REID/data', verbose=True):
        # Build both official directions from the validated loader. verbose=False
        # on the inner loads -> we print one combined summary instead of two.
        ag = AGReIDV2(root=root, protocol=self.AERIAL_TO_GROUND, verbose=False)
        ga = AGReIDV2(root=root, protocol=self.GROUND_TO_AERIAL, verbose=False)

        # ---- train: protocol-independent (full train_all); take exp1's verbatim.
        # Sanity: the two protocols must agree on the train split (same folder set,
        # same relabel) -- otherwise the loader changed under us.
        assert ag.num_train_pids == ga.num_train_pids, (
            f"train pid count differs across protocols "
            f"({ag.num_train_pids} vs {ga.num_train_pids}); train must be "
            f"protocol-independent")
        assert ag.num_train_imgs == ga.num_train_imgs, (
            f"train img count differs across protocols "
            f"({ag.num_train_imgs} vs {ga.num_train_imgs})")
        self.train = ag.train
        self.pid2label = ag.pid2label
        self.num_train_pids = ag.num_train_pids
        self.num_train_imgs = ag.num_train_imgs
        self.num_train_cams = ag.num_train_cams

        # ---- test: one shared folder-name -> int map over BOTH protocols' q+g.
        # (exp1 and exp4 list the same test identities; a query and its gallery
        # match must get the same int pid in either direction.)
        test_names = set()
        for split in (ag.query, ag.gallery, ga.query, ga.gallery):
            for d in split:
                test_names.add(d['pid'])     # raw folder-name string
        self.test_pid2label = {name: i for i, name in enumerate(sorted(test_names))}

        # A->G (exp1): query Aerial(UAV) -> gallery Ground(CCTV)
        # G->A (exp4): query Ground(CCTV) -> gallery Aerial(UAV)
        # Concatenate so filter_by_view recovers each official direction.
        self.query = (self._relabel(ag.query, expect_view='Aerial')
                      + self._relabel(ga.query, expect_view='Ground'))
        self.gallery = (self._relabel(ag.gallery, expect_view='Ground')
                        + self._relabel(ga.gallery, expect_view='Aerial'))

        self.num_query_pids = len({d['pid'] for d in self.query})
        self.num_gallery_pids = len({d['pid'] for d in self.gallery})

        # keep the per-direction views around for sanity / debugging
        self._n_exp1_q = len(ag.query)
        self._n_exp1_g = len(ag.gallery)
        self._n_exp4_q = len(ga.query)
        self._n_exp4_g = len(ga.gallery)

        if verbose:
            self._print_stats()

    def _relabel(self, split, expect_view=None):
        """Copy split dicts, mapping the folder-name pid -> shared test int.

        expect_view, if given, asserts every sample carries that view (guards the
        A->G/G->A layout contract that filter_by_view depends on).
        """
        out = []
        for d in split:
            if expect_view is not None and d['view'] != expect_view:
                raise AssertionError(
                    f"expected view={expect_view!r} but got {d['view']!r} for "
                    f"{d['img_path']}; AG-ReID.v2 protocol layout broke")
            out.append({
                'img_path': d['img_path'],
                'pid': self.test_pid2label[d['pid']],   # str -> shared int
                'camid': d['camid'],
                'view': d['view'],
            })
        return out

    def _print_stats(self):
        def cnt(split):
            pids = len({d['pid'] for d in split})
            cams = len({d['camid'] for d in split})
            a = sum(d['view'] == 'Aerial' for d in split)
            g = sum(d['view'] == 'Ground' for d in split)
            return len(split), pids, cams, a, g

        print("=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, "
              "CARGO-aligned)")
        print(f"   A->G = exp1 aerial_to_cctv ({self._n_exp1_q} q / "
              f"{self._n_exp1_g} g)")
        print(f"   G->A = exp4 cctv_to_aerial ({self._n_exp4_q} q / "
              f"{self._n_exp4_g} g)")
        print("  -----------------------------------------------------------")
        print("  subset   | # imgs | # pids | # cams | aerial | ground")
        print("  -----------------------------------------------------------")
        for name, split in (('train', self.train), ('query', self.query),
                            ('gallery', self.gallery)):
            n, p, c, a, g = cnt(split)
            print(f"  {name:8s} | {n:6d} | {p:6d} | {c:6d} | {a:6d} | {g:6d}")
        print("  -----------------------------------------------------------")


if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else '/root/work/SOLIDER-REID/data'
    ds = AGReIDV2Combined(root=root, verbose=True)
    # the four official subsets the eval will actually use
    from agreid_v2_dataset import AGReIDV2  # noqa: F401 (already imported above)

    def fbv(split, v):
        return [s for s in split if s['view'] == v]

    print("num_train_pids:", ds.num_train_pids,
          "num_test_pids:", len(ds.test_pid2label))
    print("A->G  q_aerial:", len(fbv(ds.query, 'Aerial')),
          " g_ground:", len(fbv(ds.gallery, 'Ground')),
          "  (expect 2356 / 6347)")
    print("G->A  q_ground:", len(fbv(ds.query, 'Ground')),
          " g_aerial:", len(fbv(ds.gallery, 'Aerial')),
          "  (expect 1811 / 14362)")
