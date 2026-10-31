import math
import unittest
from experiments.samo.censor_aware_controller import CensorAwareController, survival_information, MODES

class TestSurvivalInformation(unittest.TestCase):
    def test_pairs_censored(self):
        x=survival_information([1,2,3],[1,0,1])
        self.assertAlmostEqual(x["event_rate"],2/3)
        self.assertAlmostEqual(x["pair_density"],1/3)

    def test_all_censored(self):
        x=survival_information([1,2,3],[0,0,0])
        self.assertAlmostEqual(x["event_rate"],0.)
        self.assertAlmostEqual(x["pair_density"],0.)

    def test_all_events(self):
        x=survival_information([1,2,3],[1,1,1])
        self.assertGreater(x["pair_density"],0.)
        self.assertAlmostEqual(x["event_rate"],1.)

    def test_tied_event_times(self):
        x=survival_information([1,1,2],[1,1,1])
        self.assertGreater(x["pair_density"],0.)

    def test_single_patient(self):
        x=survival_information([5],[1])
        self.assertAlmostEqual(x["pair_density"],0.)


class TestV3Controller(unittest.TestCase):
    def test_exact_zero_baseline(self):
        c=CensorAwareController("ra_samo_v3_0")
        self.assertEqual(c.state.weights,(0.,0.,0.))
        self.assertEqual(c.state.mode,"cox")

    def test_all_variants_construct(self):
        for i in range(4):
            c=CensorAwareController(f"ra_samo_v3_{i}",warmup=0,switch_margin=0,min_dwell=0)
            state=c.update(0,[2.,2.,2.,2.],{"event_rate":.4,"pair_density":.2})
            self.assertIn(state.mode,MODES)
            self.assertTrue(all(v>=0 for v in state.weights))

    def test_rank_mode_exact_gates(self):
        """When rank progress dominates, mode=rank with img/gene weights=0."""
        c=CensorAwareController("ra_samo_v3_0",warmup=0,switch_margin=0,min_dwell=0)
        c.update(0,[2.,2.,2.,2.],{})
        state=c.update(1,[1.9,2.,2.,.5],{})
        self.assertEqual(state.mode,"rank")
        self.assertEqual(state.weights[:2],(0.,0.))
        self.assertGreater(state.weights[2],0.)

    def test_aux_mode_exact_gates(self):
        """When aux progress dominates, mode=aux with rank weight=0."""
        c=CensorAwareController("ra_samo_v3_0",warmup=0,switch_margin=0,min_dwell=0)
        c.update(0,[2.,2.,2.,2.],{})
        state=c.update(1,[1.9,.5,.5,1.9],{})
        self.assertEqual(state.mode,"aux")
        self.assertEqual(state.weights[2],0.)
        self.assertGreater(state.weights[0],0.)
        self.assertGreater(state.weights[1],0.)

    def test_cox_mode_zero_weights(self):
        """Cox mode: ALL secondary weights are exactly zero."""
        c=CensorAwareController("ra_samo_v3_0",warmup=0)
        state=c.update(0,[2.,2.,2.,2.],{})
        self.assertEqual(state.mode,"cox")
        self.assertEqual(state.weights,(0.,0.,0.))

    def test_switch_hysteresis(self):
        """min_dwell prevents rapid switching."""
        c=CensorAwareController("ra_samo_v3_0",warmup=0,switch_margin=0,min_dwell=5)
        c.update(0,[2.,2.,2.,2.],{})
        c.update(1,[1.9,2.,2.,.5],{})  # rank wants to switch
        self.assertEqual(c.mode,"rank")
        # Try to switch back immediately — blocked by min_dwell
        c.update(2,[1.8,.5,.5,1.8],{})
        self.assertEqual(c.mode,"rank")  # still rank due to dwell
        # After dwell period
        for ep in range(3,7):
            c.update(ep,[1.8,.5,.5,1.8],{})
        self.assertEqual(c.mode,"aux")  # now switches

    def test_v3_1_censoring_features_affect_score(self):
        """v3.1 uses event_rate and pair_density; v3.0 does not."""
        c0=CensorAwareController("ra_samo_v3_0",warmup=0,switch_margin=0,min_dwell=0)
        c1=CensorAwareController("ra_samo_v3_1",warmup=0,switch_margin=0,min_dwell=0)
        losses=[2.,2.,2.,2.]
        info={"event_rate":.1,"pair_density":.3}
        c0.update(0,losses,{}); c1.update(0,losses,info)
        s0=c0.update(1,[1.9,1.5,1.5,1.5],{})
        s1=c1.update(1,[1.9,1.5,1.5,1.5],info)
        # v3.1 should produce different scores than v3.0 given same losses
        self.assertNotEqual(s0.reason,s1.reason)

    def test_v3_2_alignment_affects_score(self):
        """v3.2 uses gradient alignment; v3.1 does not."""
        c1=CensorAwareController("ra_samo_v3_1",warmup=0,switch_margin=0,min_dwell=0)
        c2=CensorAwareController("ra_samo_v3_2",warmup=0,switch_margin=0,min_dwell=0)
        losses=[2.,2.,2.,2.]
        info={"event_rate":.3,"pair_density":.2}
        align={"aux":0.9,"rank":-0.5,"aux_reliability":0.8,"rank_reliability":0.2}
        c1.update(0,losses,info); c2.update(0,losses,info)
        s1=c1.update(1,[1.9,1.5,1.5,1.5],info)
        s2=c2.update(1,[1.9,1.5,1.5,1.5],info,alignment=align)
        self.assertNotEqual(s1.reason,s2.reason)

    def test_v3_3_validation_scores_affect_mode(self):
        """v3.3 uses inner-validation rollout scores."""
        c=CensorAwareController("ra_samo_v3_3",warmup=0,switch_margin=0,min_dwell=0)
        losses=[2.,2.,2.,2.]
        info={"event_rate":.3,"pair_density":.2}
        c.update(0,losses,info)
        # Without validation scores — behaves like v3.2
        s_no=c.update(1,[1.9,1.5,1.5,1.5],info)
        # Reset and try with strong rank validation signal
        c2=CensorAwareController("ra_samo_v3_3",warmup=0,switch_margin=0,min_dwell=0)
        c2.update(0,losses,info)
        val_scores={"cox":0.60,"aux":0.55,"rank":0.75}
        s_with=c2.update(1,[1.9,1.5,1.5,1.5],info,validation_scores=val_scores)
        self.assertEqual(s_with.mode,"rank")

    def test_no_leakage_initial_mode(self):
        """All variants start in cox mode with zero weights."""
        for i in range(4):
            c=CensorAwareController(f"ra_samo_v3_{i}")
            self.assertEqual(c.mode,"cox")
            self.assertEqual(c.state.weights,(0.,0.,0.))

    def test_invalid_variant_rejected(self):
        with self.assertRaises(ValueError):
            CensorAwareController("ra_samo_v3_4")
        with self.assertRaises(ValueError):
            CensorAwareController("ra_samo_v2")

    def test_nonfinite_losses_preserve_state(self):
        c=CensorAwareController("ra_samo_v3_0",warmup=0,switch_margin=0,min_dwell=0)
        c.update(0,[2.,2.,2.,2.],{})
        c.update(1,[1.9,2.,2.,.5],{})
        self.assertEqual(c.mode,"rank")
        # NaN loss should not change state
        c.update(2,[float('nan'),1.,1.,1.],{})
        self.assertEqual(c.mode,"rank")


if __name__=="__main__":
    unittest.main()
